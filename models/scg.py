import os, sys
import numpy as np
import pickle
import types
import warnings
from tqdm import tqdm
import math
import copy

import torch as tc

from .base import *
from .util import *

from transformers.data.metrics.squad_metrics import compute_exact, compute_f1, normalize_answer

from models.util import compute_inclusion, compute_EM

import time

from datasets import concatenate_datasets

from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
import random
from scipy.special import logsumexp
from scipy.stats import beta

def inject_feedback_noise(labels, flip_pos, flip_neg, seed):
    """Corrupt the binary feedback the learner observes, leaving `labels` (used for evaluation) alone.

    `flip_pos` is p(observed=0 | true=1) and `flip_neg` is p(observed=1 | true=0); the two are
    kept separate because an LLM judge is not a symmetric noise channel -- it tends to be
    lenient, i.e. flip_neg > flip_pos, which biases the FDR the learner perceives.
    Noise is drawn from its own RNG so a run is comparable across seeds of the learner itself.
    """
    if not flip_pos and not flip_neg:
        return labels
    rng = np.random.default_rng(seed)
    rates = np.where(labels == 1, flip_pos, flip_neg)
    flip = rng.random(labels.shape[0]) < rates
    return np.where(flip, 1 - labels, labels)


def extract_scores(rd, kind):
    """The raw (un-tiled) confidence score stream for one score function."""
    if kind == 'fM1':
        # a generation with no token logprobs has no fM1 score; nan would silently never clear any
        # threshold, so it is pinned to 0 -- least confident, which is the honest reading.
        return np.hstack([np.exp(np.mean(lp['logprobs'])) if len(lp['logprobs']) else 0.0
                          for lp in rd])
    elif kind == 'fM2':
        return np.array(rd['sample_score'])
    else:
        raise ValueError(f"Unknown confidence_score: {kind}")


def perturb_scores(scores, rd, params, labels, T, repeats):
    """Degrade the confidence score the learner thresholds, leaving `labels` alone.

    Unlike `inject_feedback_noise` this corrupts the *input* rather than the feedback, which is the
    setting where FDR control should be untouched by construction: the guarantee is over the
    threshold class, not over the score being informative. What must move is the efficiency -- an
    uninformative score leaves no threshold that both controls FDR and commits often.

    Its RNG is offset from the feedback-noise RNG so the two perturbations stay independent when a
    run enables both.
    """
    mode = getattr(params, 'score_mode', None) or 'clean'
    if mode == 'clean':
        return scores

    rng = np.random.default_rng(getattr(params, 'seed', 42) + 10_000)
    if mode == 'shuffle':
        # marginal score distribution preserved, score-label association destroyed. The honest
        # 'random score' control: a uniform draw would also change the scale of the score.
        out = rng.permutation(scores)
    elif mode == 'uniform':
        out = rng.random(T)
    elif mode == 'mixed_bern':
        # fM1 and fM2 live on different scales, so alternating them per round makes the optimal
        # threshold jump -- no single fixed threshold covers the stream.
        other = 'fM1' if params.confidence_score == 'fM2' else 'fM2'
        alt = np.tile(extract_scores(rd, other), repeats)[:T]
        p = params.score_mix_p
        keep = rng.random(T) < p
        out = np.where(keep, scores, alt)
        print(f"# score mix: {params.confidence_score} w.p. {p} / {other} otherwise, "
              f"realised fraction={keep.mean():.4f}")
    elif mode == 'switch':
        # An abrupt changepoint rather than a per-round mixture: --confidence_score for the first
        # --score_switch_at fraction of the stream, the other score function afterwards. The two
        # score functions differ in scale *and* in quality, so the post-switch segment must be read
        # against a run on the other score alone, never against the pre-switch segment.
        other = 'fM1' if params.confidence_score == 'fM2' else 'fM2'
        alt = np.tile(extract_scores(rd, other), repeats)[:T]
        k = int(round(params.score_switch_at * T))
        out = np.concatenate([scores[:k], alt[k:]])
        print(f"# score switch: {params.confidence_score} for t<{k}, {other} for t>={k} (T={T})")
    elif mode == 'mixed_unif':
        # a fraction eps of the rounds get a score carrying no information at all -- the partial
        # version of 'uniform', so the degradation can be traced as eps grows rather than only at
        # the extreme.
        eps = params.score_unif_p
        corrupt = rng.random(T) < eps
        out = np.where(corrupt, rng.random(T), scores)
        print(f"# score mix: U(0,1) w.p. {eps} / {params.confidence_score} otherwise, "
              f"realised fraction={corrupt.mean():.4f}")
    else:
        raise ValueError(f"Unknown score_mode: {mode}")

    corr_before = np.corrcoef(scores, labels)[0, 1]
    corr_after = np.corrcoef(out, labels)[0, 1]
    print(f"# score_mode={mode}: score-label corr {corr_before:.4f} -> {corr_after:.4f}, "
          f"mean {scores.mean():.4f} -> {out.mean():.4f}")
    return out


def compute_per_dist_metrics(dist_ids, commits, true_labels):
    """Running FDR / inefficiency restricted to each source distribution.

    Curves are indexed by the global round t and forward-filled over rounds where the
    distribution did not appear, so they can be overlaid on the overall FDR plot. `n` is the
    distribution's own round count: report it next to the FDR, because a minority
    distribution's running FDR is very noisy while its denominator `n_commit` is still small.
    """
    out = {}
    for d in np.unique(dist_ids):
        mask = dist_ids == d
        c = np.cumsum(np.where(mask, commits, 0))
        mc = np.cumsum(np.where(mask & (true_labels == 0), commits, 0))
        n = np.cumsum(mask)
        out[int(d)] = {
            'fdrs': (mc / np.maximum(c, 1)).tolist(),
            'inefficiencies': (1 - c / np.maximum(n, 1)).tolist(),
            'n': int(n[-1]),
            'n_commit': int(c[-1]),
        }
    return out


# INFO: Online selective generation
class OSGLearner:
    def __init__(self, params=None, name_postfix=None):
        self.params = params
        self.name_postfix = name_postfix
        self.alpha = params.alpha
        self.h_size = params.h_size
        self.T = params.T
        
        self.tau_h = np.linspace(0, 1, self.h_size)
        self.weights = np.ones_like(self.tau_h) / self.h_size
        # self.lambda_ = (1/(1-self.alpha)) * (self.T)**(1/4) # a special? case of \ell_max, when lambda >= (1/(1-alpha))
        # self.ell_max = self.lambda_
        # self.lambda_ = self.T
        # if params.dlambda is not None:
        #     self.lambda_ = (self.T) ** (params.dlambda)
        # self.ell_max = self.lambda_ + 1
        
        if params.overLambda:
            if params.lambda_T == 1:
                self.lambda_ = (self.T)
            elif params.lambda_T == 2:
                self.lambda_ = (self.T)**(1/2)
            elif params.lambda_T == 3:
                self.lambda_ = (self.T)**(1/4)
            elif params.lambda_T == 4:
                self.lambda_ = (self.T)**(1/8)
            elif params.lambda_T == 5:
                self.lambda_ = (self.T)**(1/6)
        else:
            self.lambda_ = (self.T)**(1/2)
            # self.lambda_ = (self.T)**(1/6)
            # self.lambda_ = (self.T)**(1/4)
            # self.lambda_ = (self.T)**(1/2)
        
        self.err = np.zeros_like(self.tau_h)
        self.fgap = np.zeros_like(self.tau_h)
        self.ineff = np.zeros_like(self.tau_h)
        
        self.lr = np.sqrt(8*np.log(self.h_size)/(self.T))#/self.ell_max
        
        self.mc_sum = 0
        self.c_sum = 0
        self.em_sum = 0
        
        
        self.efficiencies = []
        self.fdrs = []
        self.t = 1
        # self.k = 0 # INFO: for doubling trick
        # self.doub_step = 0
        self.emp_fdrs = []
        
        
        ########### regret calculation ###########
        self.reg_fgap = np.zeros_like(self.tau_h)
        self.reg_ineff = np.zeros_like(self.tau_h)
        self.loss = np.zeros(self.h_size)       # L_t(i)
        self.prev_loss = np.zeros(self.h_size)  # L_{t-1}(i)
        
        self.cum_loss_all_tau = np.zeros(self.h_size)  # Σ ℓ_s(i)
        self.cum_loss_learner = 0.0                    # Σ ℓ_s(i_s)
        self.cum_loss_all_tau_hist = []   # shape: [t][h_size]
        self.cum_loss_learner_hist = []   # shape: [t]
        self.regrets_static = []          # R_t for hindsight tau*
        self.regrets_online = []          # R_t for online best
        ##########################################
        
    def train(self, rd, updated_params=None):
        # init params
        params = copy.deepcopy(self.params)
        if updated_params:
            for a in dir(updated_params):
                if a[:2] != '__' and a[-2:] != '__':
                    setattr(params, a, getattr(updated_params, a))
        self.params = params
        n = params.n
        self.alpha = params.alpha

        if self.name_postfix is None:
            self.name_postfix = ''    
        self.name_postfix = self.name_postfix + f'n-{n}-alpha-{self.alpha:e}'
        verbose = params.verbose
        
        print(f"# learn a prediction: n = {n}, alpha = {self.alpha:.2e}")

        # scores = np.hstack([np.exp(np.array(lp['logprobs']).mean()) for lp in rd]) \
        #     if params.confidence_score == 'fM1' \
        #     else np.array(rd['confidence_score'])
        #     # else np.hstack([(1 - np.array(ss['samples_scores'])[..., 0]).mean() for ss in rd])
        
        # # scores_m3 = tc.hstack([(1 - tc.tensor(lp['entail_scores'])[0]) for lp in rd])

        scores, labels, fed_labels, dist_ids = self._prepare_stream(rd, params)

        tau_history = []  # for data shift
        commits = np.empty(self.T, dtype=bool)  # per-round commit decision, for the per-distribution metrics

        for t, (score, label, fed_label) in enumerate(zip(scores, labels, fed_labels)):
            # self.lambda_ = 1/(self.t*self.alpha)
            index = self.predict()
            self.update(score, index, fed_label, true_label=label)
            # print('q:', rd[t]['question'], 'a:', rd[t]['answer'], 'g:', rd[t]['generated_sequence'], label, f"IDK: {self.tau_h[index] >= score}")
            tau_history.append(self.tau_h[index]) # for data shift
            commits[t] = score >= self.tau_h[index]

        eff_guaranteed = []
        for tau in self.tau_h:
            fdr = ((labels[(tau <= scores) & (labels == 0)].shape[0]) / ((tau <= scores).sum())) if (tau <= scores).sum() > 0 else 0
            eff = (tau <= scores).sum()
            if fdr <= self.alpha:
                eff_guaranteed.append(eff)

        best_efficiency = (max(eff_guaranteed) / self.T) if len(eff_guaranteed) > 0 else 0
        
        ########### regret calculation ###########
        final_cum_loss_all_tau = self.cum_loss_all_tau_hist[-1]
        best_idx = np.argmin(final_cum_loss_all_tau)

        self.regrets_static = []
        for t in range(len(self.cum_loss_learner_hist)):
            L_learn_t = self.cum_loss_learner_hist[t]
            L_best_t  = self.cum_loss_all_tau_hist[t][best_idx]
            self.regrets_static.append((L_learn_t - L_best_t)/t)
        ##########################################
        
        # print(self.mc_sum, self.c_sum)
        
        return {
            'weights': self.weights,
            'fdrs': self.fdrs,
            'efficiencies': self.efficiencies,
            'tau_h': self.tau_h,
            'best_efficiency': best_efficiency,
            'tau_history': tau_history,  # for data shift
            'empirical_fdrs': self.emp_fdrs,
            'regrets_static': self.regrets_static,
            'regrets_online': self.regrets_online,
            'per_dist': compute_per_dist_metrics(dist_ids, commits, labels),
            'round_log': {
                'score': scores, 'true_label': labels, 'fed_label': fed_labels,
                'dist_id': dist_ids, 'tau': np.array(tau_history), 'commit': commits,
            },
        }

    def _prepare_stream(self, rd, params):
        """Turn the dataset into the four aligned per-round streams the learner consumes.

        `labels` is the ground truth the reported FDR is measured against; `fed_labels` is what
        the learner actually observes (they differ only under injected feedback noise).
        All four are tiled identically -- if they ever drift apart the per-distribution
        aggregation silently mismatches, so keep the tiling in this one place.
        """
        scores = extract_scores(rd, params.confidence_score)

        labels = np.hstack([np.array(lp['label']) for lp in rd])
        # datasets without provenance (the single-distribution experiments) collapse to one group
        dist_ids = np.array(rd['dist_id']) if 'dist_id' in rd.column_names else np.zeros(len(labels), dtype=int)

        repeats = math.ceil(self.T / len(scores))
        scores = np.tile(scores, repeats)[:self.T]
        labels = np.tile(labels, repeats)[:self.T]
        dist_ids = np.tile(dist_ids, repeats)[:self.T]

        # after tiling, for the same reason as the feedback noise below: a repeated example must not
        # carry the identical perturbation on every pass through the dataset.
        scores = perturb_scores(scores, rd, params, labels, self.T, repeats)

        # under a score changepoint the two sides of the switch are the groups worth separating, so
        # reuse the per-distribution machinery to get pre/post FDR and inefficiency curves for free.
        # nq/triviaqa are single-distribution, so nothing is overwritten that carried information.
        if getattr(params, 'score_mode', None) == 'switch':
            k = int(round(params.score_switch_at * self.T))
            dist_ids = (np.arange(self.T) >= k).astype(int)
            for seg in (0, 1):
                m = dist_ids == seg
                print(f"#   segment {seg} (n={m.sum()}): score mean {scores[m].mean():.4f}, "
                      f"corr with label {np.corrcoef(scores[m], labels[m])[0, 1]:.4f}")

        # after tiling, so a repeated example does not carry the same corrupted feedback every pass.
        # `is None` rather than `or`: an explicit rate of 0.0 must override the symmetric shorthand,
        # which is exactly how a one-sided condition is expressed (flip_rate_pos=0, flip_rate_neg=0.2).
        symmetric = getattr(params, 'feedback_flip_rate', None)
        flip_pos = getattr(params, 'flip_rate_pos', None)
        flip_neg = getattr(params, 'flip_rate_neg', None)
        flip_pos = symmetric if flip_pos is None else flip_pos
        flip_neg = symmetric if flip_neg is None else flip_neg
        flip_pos, flip_neg = flip_pos or 0.0, flip_neg or 0.0
        fed_labels = inject_feedback_noise(labels, flip_pos, flip_neg, getattr(params, 'seed', 42))
        if flip_pos or flip_neg:
            print(f"# feedback noise: flip_pos={flip_pos}, flip_neg={flip_neg}, "
                  f"observed flip fraction={np.mean(fed_labels != labels):.4f}")

        return scores, labels, fed_labels, dist_ids

    def predict(self):
        return np.random.choice(np.arange(self.tau_h.size), p=self.weights)

    def update(self, score, index, label, true_label=None):
        # `label` is the (possibly noisy) feedback that drives the weight update; `true_label` is
        # the ground truth the reported FDR is measured against. Mixing the two would make the
        # feedback-noise experiment vacuous -- the metric would be corrupted along with the signal.
        if true_label is None:
            true_label = label
        tau = self.tau_h[index]
        weight = self.weights[index]

        # the analysis assumes f_t in [0,1), which is what makes tau=1 the "always abstain"
        # hypothesis the guarantee leans on. Real scores land on exactly 1.0 often (68.9% of
        # gpt/triviaqa), and under `score >= tau` that would instead turn tau=1 into "commit on
        # all of them".
        score = min(score, np.nextafter(1.0, 0.0))

        pred_c = (score >= tau)
        pred_m = (label == 0)
        pred_m_true = (true_label == 0)

        self.mc_sum += pred_m_true * pred_c
        self.c_sum += pred_c
        self.em_sum += pred_m_true

        c = score >= self.tau_h
        m = label == 0

        self.compute_loss(c, m, index, pred_c, pred_m, score)
        self.err = ((self.ineff + self.fgap * self.lambda_) / (self.lambda_ + 1)) * self.lr
        self.weights = np.exp(-self.err - logsumexp(-self.err))

        self.efficiencies.append(self.c_sum/(self.t))
        self.fdrs.append(self.mc_sum / (self.c_sum if self.c_sum != 0 else 1))
        self.emp_fdrs.append(self.em_sum / self.t)
            
        ########### regret calculation ###########
        self.cum_loss_learner += (self.reg_ineff[index] + self.reg_fgap[index] * self.lambda_) / (self.lambda_ + 1)
        self.cum_loss_all_tau += (self.reg_ineff + self.reg_fgap * self.lambda_) / (self.lambda_ + 1)
        ##########################################
        
        ########### regret calculation ###########
        self.cum_loss_learner_hist.append(self.cum_loss_learner)
        self.cum_loss_all_tau_hist.append(self.cum_loss_all_tau.copy())
        ##########################################
        
        ########### regret calculation ###########
        regret_t = self.cum_loss_learner - np.min(self.cum_loss_all_tau)
        self.regrets_online.append(regret_t/self.t)
        ##########################################
        
        self.t += 1
        
    def compute_loss(self, c, m, index, pred_c, pred_m, score):
        # if self.params.unk_T:
        #     self.lr = np.sqrt(4*np.log(self.h_size)/(self.T))#/self.ell_max
        #     self.lambda_ = (self.t)**(1/4)
        
        self.fgap = self.fgap + ((m - self.alpha)*c + self.alpha)# / (self.lambda_+1)
        self.ineff = self.ineff + (1-c)# / (self.lambda_+1)
        
        self.reg_fgap = (m - self.alpha)*c + self.alpha
        self.reg_ineff = 1-c
        
        
# INFO: This it the class for the adversarial bandit setting sg. (Exp3-IX)
class OSGAdvBanditLearner(OSGLearner):
    def __init__(self, params=None, name_postfix=None):
        super().__init__(params=params, name_postfix=name_postfix)
        # self.lr = np.sqrt(8*np.log(self.h_size)/(self.T))#/self.ell_max
        self.lr = np.sqrt(2*np.log(self.h_size)/(self.T * self.h_size))#/self.ell_max
        self.gamma = self.lr / 2
    
    def compute_loss(self, c, m, index, pred_c, pred_m, score):
        # if self.params.unk_T:
        #     self.lr = np.sqrt(np.log(self.h_size)/(self.t * self.h_size))#/self.ell_max
        #     self.gamma = self.lr / 2
        #     self.lambda_ = (self.t)**(1/4)
        
        self.fgap[index] = self.fgap[index] + ((m - self.alpha)*c[index] + self.alpha) / (self.weights[index] + self.gamma)# / (self.lambda_+1)
        self.ineff[index] = self.ineff[index] + ((1-c[index])) / (self.weights[index] + self.gamma)# / (self.lambda_+1)
        
        self.reg_fgap = (m - self.alpha)*c + self.alpha
        self.reg_ineff = 1-c
        
class Exp3Learner(OSGLearner):
    def __init__(self, params=None, name_postfix=None):
        super().__init__(params=params, name_postfix=name_postfix)
        self.lr = np.sqrt(2*np.log(self.h_size)/(self.T * self.h_size))
    
    def compute_loss(self, c, m, index, pred_c, pred_m, score):
        self.fgap[index] = self.fgap[index] + ((m - self.alpha)*c[index] + self.alpha) / (self.weights[index])
        self.ineff[index] = self.ineff[index] + ((1-c[index])) / (self.weights[index])
        
        self.reg_fgap = (m - self.alpha)*c + self.alpha
        self.reg_ineff = 1-c
        
class Exp3PLearner(OSGLearner):
    def __init__(self, params=None, name_postfix=None):
        super().__init__(params=params, name_postfix=name_postfix)
        self.lr = 0.95 * np.sqrt(np.log(self.h_size)/(self.T * self.h_size))
        self.beta = np.sqrt(np.log(self.h_size)/(self.T * self.h_size))
        self.gamma = 1.05 * np.sqrt((np.log(self.h_size) * self.h_size)/(self.T))
    
    def compute_loss(self, c, m, index, pred_c, pred_m, score):
        self.fgap[index] = self.fgap[index] + ((m - self.alpha)*c[index] + self.alpha + self.beta) / ((1 - self.gamma) * self.weights[index] + self.gamma / self.h_size)
        self.ineff[index] = self.ineff[index] + ((1-c[index]) + self.beta) / ((1 - self.gamma) * self.weights[index] + self.gamma / self.h_size)
        
        self.reg_fgap = (m - self.alpha)*c + self.alpha
        self.reg_ineff = 1-c    
        
# INFO: ExAUL
class OSGSemiAdvBanditLearner(OSGLearner):
    def __init__(self, params=None, name_postfix=None):
        super().__init__(params=params, name_postfix=name_postfix)
        
        self.lr = np.sqrt(np.log(self.h_size)/(self.T))
        self.gamma = self.lr / 2
        
    def compute_loss(self, c, m, index, pred_c, pred_m, score):
        cum = np.cumsum(self.weights)
        below_indices = np.where(self.tau_h <= score)[0]
        above_indices = np.where(self.tau_h > score)[0]

        i_below = below_indices[-1] if len(below_indices) > 0 else 0
        i_above = above_indices[0] if len(above_indices) > 0 else 0
        if pred_c: # not IDK
            mask = self.tau_h <= score
            estimate_weight = cum[i_below]
        else:
            mask = self.tau_h > score
            estimate_weight = cum[-1] - (cum[i_above-1] if i_above > 0 else 0)
            
        denom = self.gamma + estimate_weight
        
        self.fgap[mask] += ((m - self.alpha) * c[mask] + self.alpha) / (denom)
        self.ineff[mask] += (1 - c[mask]) / (denom)
        
        self.reg_fgap = (m - self.alpha)*c + self.alpha
        self.reg_ineff = 1-c

class OSGAblationLearner(OSGSemiAdvBanditLearner):
    def __init__(self, params=None, name_postfix=None):
        super().__init__(params=params, name_postfix=name_postfix)

    def train(self, rd, updated_params=None):
        params = copy.deepcopy(self.params)
        if updated_params:
            for a in dir(updated_params):
                if a[:2] != '__' and a[-2:] != '__':
                    setattr(params, a, getattr(updated_params, a))
        self.params = params
        n = params.n
        self.alpha = params.alpha

        if self.name_postfix is None:
            self.name_postfix = ''
        self.name_postfix = self.name_postfix + f'n-{n}-alpha-{self.alpha:e}'

        print(f"# learn a prediction: n = {n}, alpha = {self.alpha:.2e}")

        scores, labels, fed_labels, dist_ids = self._prepare_stream(rd, params)

        tau_history = []
        commits = np.empty(self.T, dtype=bool)

        for t, (score, label, fed_label) in enumerate(zip(scores, labels, fed_labels)):
            index = self.predict()
            self.update(score, index, fed_label, true_label=label)
            tau_history.append(self.tau_h[index])
            commits[t] = score >= self.tau_h[index]

        eff_guaranteed = []
        for tau in self.tau_h:
            fdr = ((labels[(tau <= scores) & (labels == 0)].shape[0]) / ((tau <= scores).sum())) if (tau <= scores).sum() > 0 else 0
            eff = (tau <= scores).sum()
            if fdr <= self.alpha:
                eff_guaranteed.append(eff)

        best_efficiency = (max(eff_guaranteed) / self.T) if len(eff_guaranteed) > 0 else 0

        return {
            'weights': self.weights,
            'fdrs': self.fdrs,
            'efficiencies': self.efficiencies,
            'tau_h': self.tau_h,
            'best_efficiency': best_efficiency,
            'tau_history': tau_history,
            'empirical_fdrs': self.emp_fdrs,
            'regrets_static': [],
            'regrets_online': [],
            'per_dist': compute_per_dist_metrics(dist_ids, commits, labels),
            'round_log': {
                'score': scores, 'true_label': labels, 'fed_label': fed_labels,
                'dist_id': dist_ids, 'tau': np.array(tau_history), 'commit': commits,
            },
        }

    def update(self, score, index, label, true_label=None):
        if true_label is None:
            true_label = label
        tau = self.tau_h[index]
        weight = self.weights[index]

        # see OSGLearner.update: keeps tau=1 as the always-abstain hypothesis
        score = min(score, np.nextafter(1.0, 0.0))

        pred_c = (score >= tau)
        pred_m = (label == 0)
        pred_m_true = (true_label == 0)

        self.mc_sum += pred_m_true * pred_c
        self.c_sum += pred_c
        self.em_sum += pred_m_true

        c = score >= self.tau_h
        m = label == 0

        self.compute_loss(c, m, index, pred_c, pred_m, score)
        # self.err = (self.fgap * self.lambda_) * self.lr # T
        self.err = self.fgap * self.lr
        self.weights = np.exp(-self.err - logsumexp(-self.err))
            
        self.efficiencies.append(self.c_sum/(self.t))
        self.fdrs.append(self.mc_sum / (self.c_sum if self.c_sum != 0 else 1))
        self.emp_fdrs.append(self.em_sum / self.t)
    
        self.t += 1
        

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
        self.lambda_ = (self.T)**(1/4)
        self.ell_max = self.lambda_ + 1
        
        self.err = np.zeros_like(self.tau_h)
        self.fgap = np.zeros_like(self.tau_h)
        self.ineff = np.zeros_like(self.tau_h)
        
        self.lr = np.sqrt(8*np.log(self.h_size)/(self.T))/self.ell_max
        
        self.mc_sum = 0
        self.c_sum = 0
        
        
        self.efficiencies = []
        self.fdrs = []
        self.t = 0
        
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

        if params.confidence_score == 'fM1':
            scores = np.hstack([np.exp(np.array(lp['logprobs']).mean()) for lp in rd])
        elif params.confidence_score == 'fM2':
            scores = np.array(rd['sample_score'])
        else:
            raise ValueError(f"Unknown confidence_score: {params.confidence_score}") 

        labels = np.hstack([np.array(lp['label']) for lp in rd])
        
        tau_history = []  # for data shift

        repeats = math.ceil(self.T / len(scores))
        scores = np.tile(scores, repeats)[:self.T]
        labels = np.tile(labels, repeats)[:self.T]

        for t, (score, label) in enumerate(zip(scores, labels)):
            index = self.predict()
            self.update(score, index, label)

            tau_history.append(self.tau_h[index]) # for data shift
        
        # print(self.tau_h[np.argmin(self.err)])
        best_efficiency = (
            (self.tau_h[np.argmin(self.err)] < scores).sum() \
            + self.lambda_ * (((self.tau_h[np.argmin(self.err)] >= scores)*(labels==0)).sum() - self.alpha * (self.tau_h[np.argmin(self.err)] >= scores).sum())
        )/self.T
        
        return {
            'weights': self.weights,
            'fdrs': self.fdrs,
            'efficiencies': self.efficiencies,
            'tau_h': self.tau_h,
            'best_efficiency': best_efficiency,
            'tau_history': tau_history,  # for data shift
        }
        
    def predict(self):
        return np.random.choice(np.arange(self.tau_h.size), p=self.weights)
        
    def update(self, score, index, label):
        tau = self.tau_h[index]
        weight = self.weights[index]

        pred_c = (score >= tau)
        pred_m = (label == 0)

        self.mc_sum += pred_m * pred_c
        self.c_sum += pred_c
        
        c = score >= self.tau_h
        m = label == 0
        
        self.compute_loss(c, m, index, pred_c, pred_m, score)
        self.err = (self.ineff + self.fgap * self.lambda_) * self.lr # T


        self.weights = np.exp(-self.err - logsumexp(-self.err))
            
        self.t += 1
        self.efficiencies.append(self.c_sum/(self.t))
        self.fdrs.append(self.mc_sum / (self.c_sum if self.c_sum != 0 else 1))
    
    def compute_loss(self, c, m, index, pred_c, pred_m, score):
        self.fgap = self.fgap + (m - self.alpha)*c + self.alpha
        self.ineff = self.ineff + (1-c)
        
        
# INFO: This it the class for the adversarial bandit setting sg.
class OSGAdvBanditLearner(OSGLearner):
    def __init__(self, params=None, name_postfix=None):
        super().__init__(params=params, name_postfix=name_postfix)
        # self.lr = np.sqrt(8*np.log(self.h_size)/(self.T))/self.ell_max
        self.lr = np.sqrt(2*np.log(self.h_size)/(self.T * self.h_size))/self.ell_max
    
    def compute_loss(self, c, m, index, pred_c, pred_m, score):
        self.fgap[index] = self.fgap[index] + ((m - self.alpha)*c[index] + self.alpha) / (self.weights[index] + np.finfo(float).eps)
        self.ineff[index] = self.ineff[index] + ((1-c[index])) / (self.weights[index] + np.finfo(float).eps)
        

class OSGSemiAdvBanditLearner(OSGLearner):
    def __init__(self, params=None, name_postfix=None):
        super().__init__(params=params, name_postfix=name_postfix)
        # self.lr = np.sqrt(8*np.log(self.h_size)/(self.T))/self.ell_max
        self.lr = np.sqrt(np.log(self.h_size)/(self.T))/self.ell_max
        
    def compute_loss(self, c, m, index, pred_c, pred_m, score):
        
        cum = np.cumsum(self.weights)
        below_indices = np.where(self.tau_h < score)[0]
        above_indices = np.where(self.tau_h > score)[0]

        i_below = below_indices[-1] if len(below_indices) > 0 else 0 # Will not be used?
        i_above = above_indices[0] if len(above_indices) > 0 else 0 # Will not be used?
        if pred_c:  # good prediction
            mask = self.tau_h <= score
            estimate_weight = cum[i_below]
        else:
            mask = self.tau_h > score
            estimate_weight = cum[-1] - (cum[i_above-1] if i_above > 0 else 0)
        
        self.fgap[mask] += ((m - self.alpha) * c[mask] + self.alpha) / (estimate_weight)#(self.h_size * np.max(self.weights))#(self.weights[mask]+np.finfo(float).eps)#np.max(self.weights)
        self.ineff[mask] += (1 - c[mask]) / (estimate_weight)#(self.h_size * np.max(self.weights))#np.max(self.weights+np.finfo(float).eps)


class Exp3EwLearner(OSGSemiAdvBanditLearner):
    def __init__(self, params=None, name_postfix=None):
        super().__init__(params=params, name_postfix=name_postfix)
             
    def compute_loss(self, c, m, index, pred_c, pred_m, score):
        if pred_c:
            mask = np.arange(self.h_size) < self.h_size
            self.fgap += ((m - self.alpha) * c + self.alpha)#(self.h_size * np.max(self.weights))#(self.weights[mask]+np.finfo(float).eps)#np.max(self.weights)
            self.ineff += (1 - c)#(self.h_size * np.max(self.weights))#np.max(self.weights+np.finfo(float).eps)
        else:
            self.fgap[index] = self.fgap[index] + ((m - self.alpha)*c[index] + self.alpha) / (np.max(self.weights))
            self.ineff[index] = self.ineff[index] + ((1-c[index])) / (np.max(self.weights))

        
        
        
        
        
    # def plot(self, rd, updated_params=None):
        
    #     # init params
    #     params = copy.deepcopy(self.params)
    #     if updated_params:
    #         for a in dir(updated_params):
    #             if a[:2] != '__' and a[-2:] != '__':
    #                 setattr(params, a, getattr(updated_params, a))
    #     n = params.n
    #     self.n_e = n_e = params.n_e
    #     eps = params.eps
    #     delta = params.delta

    #     eps_e = params.eps_e
    #     delta_p = params.delta_p
        
    #     delta_s = (delta-delta_p)/2
    #     delta_e = (delta-delta_p)/2

    #     fer = params.fer
    #     K = params.K

    #     if self.name_postfix is None:
    #         self.name_postfix = ''    
    #     self.name_postfix = self.name_postfix + f'n-{n}-eps-{eps:e}-delta-{delta:e}'
    #     verbose = params.verbose
        
    #     print(f"# learn a prediction set: n = {n}, eps = {eps:.2e}, delta = {delta:.2e}")

    #     if self.params.cache_cal_fn:
    #         cache_fn = os.path.join(
    #             self.params.cache_root,
    #             f'{self.params.cache_cal_fn}-CAL2',
    #         )
    #         cache_fn_e = os.path.join(
    #             self.params.cache_root,
    #             f'{self.params.cache_cal_fn}-CAL2_E',
    #         )
    #         os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
    #     else:
    #         cache_fn = None
    #         cache_fn_e = None

    #     if os.path.exists(f'{params.output_dir}/results.txt'):
    #         with open(f'{self.params.output_dir}/results.txt', 'w') as f:
    #             f.truncate(0)


    #     scores_m1 = tc.hstack([tc.tensor(lp['logprobs']).mean().exp() for lp in rd])
    #     scores_m2 = tc.hstack([(1 - tc.tensor(ss['samples_scores'])[..., 0]).mean() for ss in rd])
    #     scores_m3 = tc.hstack([(1 - tc.tensor(lp['entail_scores'])[0]) for lp in rd])

    #     # print(scores_m1.sort(descending=True))
    #     # print(scores_m2.sort(descending=True))

    #     tau_h = np.linspace(0, 1, 10000)
    #     weights = np.ones_like(tau_h) / len(tau_h)
    #     # print(tau_h)
    #     labels = tc.hstack([tc.tensor(lp['label']) for lp in rd])
        
    #     A_t = []
    #     f_t = []
    #     t_t = []
    #     DD_t = []
        
    #     for idx, score in enumerate(scores_m2):
    #         if labels[idx] == 0:
    #             f_t.append(score)
    #         elif labels[idx] == 1:
    #             t_t.append(score)

    #     for tau in tau_h:
    #         A_t.append(np.mean((scores_m2 < tau).cpu().numpy(), 0))
    #         # print(
    #         #     tc.maximum(
    #         #         tc.where((scores_m2 >= tau).float() == 0, -0.25, (((scores_m2 >= tau).float() * (labels == 0).float()).mean()) / (((scores_m2 >= tau).float()).mean()) - 0.25),
    #         #         tc.zeros_like(scores_m2)
    #         #     ).shape
    #         # )
    #         DD_t.append(
    #             (len(rd))**(1/4)*
    #             tc.maximum(
    #                 tc.where((scores_m2 >= tau).float() == 0, -0.25, (((scores_m2 >= tau).float() * (labels == 0).float()).mean()) / (((scores_m2 >= tau).float()).mean()) - 0.25),
    #                 tc.zeros_like(scores_m2)
    #             ).mean().cpu().numpy()
    #         )
    #         # temp = ((scores_m2 >= tau).float() * (labels == 0).float() - (scores_m2 >= tau).float() * 0.25 + 0.25)
    #         # DD_t.append(np.mean((len(rd))**(1/4) * temp.cpu().numpy(), 0))
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(tau_h, A_t, label="A_t")
    #     plt.scatter(f_t, np.zeros_like(f_t), label="D_t", color='red', s=5)
    #     plt.scatter(t_t, np.ones_like(t_t), label="T_t", color='blue', s=5)
    #     plt.plot(tau_h, DD_t, label="DD_t")
    #     plt.plot(tau_h, np.array(A_t) + np.array(DD_t), label="A_t + DD_t")
    #     # plt.axhline(0, color='red', linestyle='--', label='y=0')
    #     plt.xlabel("tau")
    #     plt.ylabel("Value")
    #     plt.title("Plot of A_t, D_t and A_t + D_t vs tau")
    #     plt.legend()
    #     plt.savefig('proof.png')
    #     # plt.show()
    #     exit()
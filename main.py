import logging
#import math
import os
import sys
#from itertools import chain
import warnings
import pickle
import io
import json
import numpy as np
import time
import types
import random

import datasets
from datasets import concatenate_datasets, load_dataset, load_from_disk#, load_metric

import transformers
from transformers import (
    #CONFIG_MAPPING,
    #MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    #TrainingArguments,
    default_data_collator,
    # is_torch_tpu_available,
    set_seed,
)
from transformers.utils import check_min_version

# prompt learning
# from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, PeftModelForCausalLM

import torch as tc
from torch.utils.data import DataLoader

import models

# from nltk.corpus import wordnet

from transformers.testing_utils import CaptureLogger
# from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version, send_example_telemetry
# from transformers.utils.versions import require_version

from args import *
import util

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from tqdm import tqdm

def init_datasets_qna(data_args, model_args, training_args):  
    # INFO: Deprecated, this is for nli dataset
    if data_args.dataset_config_name:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            # cache_dir=model_args.cache_dir,
            trust_remote_code=True,
        )
        return raw_datasets['validation']
    
    if training_args.method.startswith('Adversary'):
        # Adversary dialog: the adversary picks questions from the pooled NQ + TriviaQA (GPT-3.5)
        # generations, so --dataset_name is ignored here.
        raw_datasets = concatenate_datasets([
            load_from_disk('data/nq/gpt-3.5'),
            load_from_disk('data/triviaqa/gpt-3.5'),
        ])
    else:
        try:
            raw_datasets = load_from_disk(
                data_args.dataset_name,
            )
        except:
            if not training_args.method.startswith('Dialog'):
                raise FileNotFoundError(f"Dataset {data_args.dataset_name} not found in disk.")
            return None

    # if not training_args.method.startswith('Dialog'):
    #     raw_datasets = raw_datasets.shuffle(seed=training_args.seed)

    if data_args.contradict_ratio is not None:
        ratio = data_args.contradict_ratio
        labels = raw_datasets["label"]
        idx0 = [i for i, l in enumerate(labels) if l == 0]
        idx1 = [i for i, l in enumerate(labels) if l == 1]
        n_max = min(len(idx0) / ratio, len(idx1) / (1 - ratio))
        n0 = round(n_max * ratio)
        n1 = round(n_max * (1 - ratio))
        rng = random.Random(training_args.seed)
        idx0_s = rng.sample(idx0, n0)
        idx1_s = rng.sample(idx1, n1)
        raw_datasets = raw_datasets.select(sorted(idx0_s + idx1_s))
        print(f"Resampled dataset: n={len(raw_datasets)}, contradict={n0} ({n0/len(raw_datasets):.3f}), entail={n1}")

    raw_datasets = raw_datasets.shuffle(seed=training_args.seed, keep_in_memory=True)

    # FIXME: will be deleted
    if training_args.method == 'Static-Dialog-OSG-Semi-Bandit':
        raw_datasets = raw_datasets.select(range(training_args.static_dialog_chunk, training_args.static_dialog_chunk + 500))
        print(f"Static Dialog dataset chunk: {training_args.static_dialog_chunk}~{training_args.static_dialog_chunk+499}, size: {len(raw_datasets)}")
    return raw_datasets

# provenance ids written into the `dist_id` column, consumed by models.scg.compute_per_dist_metrics
DIST_NQ, DIST_TRI = 0, 1


def _sample_chunk(ds, chunk_size, base_seed):
    """Draw `chunk_size` rows from `ds`, up-sampling by concatenating independent shuffles if needed."""
    if chunk_size <= len(ds):
        return ds.shuffle(seed=base_seed, keep_in_memory=True).select(range(chunk_size))
    reps = (chunk_size + len(ds) - 1) // len(ds)
    pieces = [ds.shuffle(seed=base_seed + r, keep_in_memory=True) for r in range(reps)]
    return concatenate_datasets(pieces).select(range(chunk_size))


def _tag_dist(ds, dist_id):
    """Attach the provenance column so per-distribution FDR is recoverable from the stream."""
    return ds.add_column('dist_id', [dist_id] * len(ds))


def build_shifted_dataset(nq_ds, trivia_ds, chunk_size, num_chunks, first_flag, seed):
    chunks = []
    for idx in range(num_chunks):
        base_seed = seed + idx
        nq_chunk = _tag_dist(_sample_chunk(nq_ds, chunk_size, base_seed), DIST_NQ)
        tri_chunk = _tag_dist(_sample_chunk(trivia_ds, chunk_size, base_seed), DIST_TRI)

        if (first_flag=='tri'):
            chunks.extend([tri_chunk, nq_chunk])
        else:
            chunks.extend([nq_chunk, tri_chunk])
    return concatenate_datasets(chunks)


def build_imbalanced_shifted_dataset(nq_ds, trivia_ds, chunk_size_nq, chunk_size_tri, num_chunks, first_flag, seed):
    """Discrete shift whose two arms get different block lengths (e.g. 27000 NQ / 3000 TriviaQA).

    Kept separate from `build_shifted_dataset` so the balanced setup used by the existing
    experiments keeps its exact behaviour. Because the blocks differ in length the shift points
    are no longer multiples of a single chunk size, so they are returned rather than re-derived
    by the caller.
    """
    chunks, boundaries, pos = [], [], 0
    for idx in range(num_chunks):
        base_seed = seed + idx
        nq_chunk = _tag_dist(_sample_chunk(nq_ds, chunk_size_nq, base_seed), DIST_NQ)
        tri_chunk = _tag_dist(_sample_chunk(trivia_ds, chunk_size_tri, base_seed), DIST_TRI)

        ordered = [tri_chunk, nq_chunk] if first_flag == 'tri' else [nq_chunk, tri_chunk]
        for chunk in ordered:
            chunks.append(chunk)
            pos += len(chunk)
            boundaries.append(pos)
    # drop the trailing boundary: it is the end of the stream, not a shift
    return concatenate_datasets(chunks), boundaries[:-1]


def build_mixture_dataset(nq_ds, trivia_ds, T, p_major, major_dist, seed):
    """Steady-state imbalanced stream: each round is drawn i.i.d. from the major distribution
    w.p. `p_major` and from the minority one otherwise.

    Unlike the shift builders the mixing ratio is constant over the horizon, so the stream is
    stationary but imbalanced (e.g. 90:10). This is the regime where the per-arm information
    the feedback unlocking relies on becomes scarce for the minority distribution.
    """
    if major_dist == 'nq':
        major_ds, minor_ds, major_id, minor_id = nq_ds, trivia_ds, DIST_NQ, DIST_TRI
    elif major_dist == 'tri':
        major_ds, minor_ds, major_id, minor_id = trivia_ds, nq_ds, DIST_TRI, DIST_NQ
    else:
        raise ValueError(f"major_dist must be 'nq' or 'tri', got {major_dist}")

    # re-seeding per round matches the convention of the gradual-shift builder below
    picks = []
    for t in range(T):
        rng = random.Random(seed + t)
        if rng.random() < p_major:
            picks.append((True, rng.randrange(len(major_ds))))
        else:
            picks.append((False, rng.randrange(len(minor_ds))))

    # one vectorised select over the concatenated pool instead of T single-row lookups
    pool = concatenate_datasets([_tag_dist(major_ds, major_id), _tag_dist(minor_ds, minor_id)])
    minor_offset = len(major_ds)
    order = [idx if is_major else minor_offset + idx for is_major, idx in picks]

    n_minor = sum(1 for is_major, _ in picks if not is_major)
    print(f"# mixture stream: T={T}, major={major_dist} p={p_major}, "
          f"minority rounds={n_minor} ({n_minor/T:.4f})")
    return pool.select(order)


# (label, colour) in the order the three learners are constructed: EW-CA, Exp3-IX-CA, ExAUL
LEARNER_LABELS = [
    (r'$\mathtt{EW\text{-}CA}$', 'blue'),
    (r'$\mathtt{Exp3\text{-}IX\text{-}CA}$', 'red'),
    (r'$\mathtt{ExAUL}$ (ours)', 'green'),
]
DIST_NAMES = {DIST_NQ: 'NQ', DIST_TRI: 'TriviaQA'}


def save_round_logs(results_list, output_dir):
    """Persist the per-round streams so a new metric can be computed without re-running the sweep."""
    payload = {}
    for name, res in zip(('ew', 'exp3ix', 'exaul'), results_list):
        for key, value in res['round_log'].items():
            payload[f'{name}__{key}'] = np.asarray(value)
    np.savez_compressed(os.path.join(output_dir, 'rounds.npz'), **payload)


def save_per_dist_summary(results_list, training_args, output_dir):
    """Final overall and per-distribution FDR / inefficiency as JSON.

    `n_commit` is the FDR denominator for that distribution -- it belongs next to the number,
    because a minority FDR computed over a handful of commits is noise, not a violation.
    """
    summary = {'alpha': training_args.alpha, 'T': training_args.T, 'seed': training_args.seed}
    for name, res in zip(('EW-CA', 'Exp3-IX-CA', 'ExAUL'), results_list):
        summary[name] = {
            'overall': {
                'fdr': float(res['fdrs'][-1]),
                'inefficiency': float(1 - res['efficiencies'][-1]),
            },
            'per_dist': {
                DIST_NAMES.get(d, str(d)): {
                    'fdr': float(pd['fdrs'][-1]),
                    'inefficiency': float(pd['inefficiencies'][-1]),
                    'n': pd['n'],
                    'n_commit': pd['n_commit'],
                }
                for d, pd in res['per_dist'].items()
            },
        }
    with open(os.path.join(output_dir, 'per_dist_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def save_curve_plot(curves, ylabel, filename, training_args, shift_positions, ylim,
                    alpha_line=True, legend_loc='upper right'):
    """One figure from a list of (label, colour, linestyle, y) curves."""
    with PdfPages(os.path.join(training_args.output_dir, filename + '.pdf')) as pdf:
        plt.figure()
        for label, color, style, y in curves:
            plt.plot(y, label=label, color=color, linestyle=style)
        plt.xlabel(r'$t$', fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        plt.ylim(*ylim)
        if alpha_line:
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed',
                        label=r'$\alpha$=' + f'{training_args.alpha}')
        for pos in shift_positions:
            plt.axvline(x=pos, color='red', linestyle=':', linewidth=1, alpha=0.5)
        plt.legend(loc=legend_loc, fontsize=8)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
        plt.tight_layout()
        plt.savefig(os.path.join(training_args.output_dir, filename + '.png'))
        pdf.savefig(bbox_inches='tight')
        plt.close()


def save_per_dist_outputs(results_list, training_args, shift_positions):
    """Per-distribution FDR / inefficiency plots, the round logs, and the JSON summary.

    Called from every two-distribution method, not just the imbalanced ones: the overall FDR can
    hide a violation on one distribution in any of them.
    """
    out = training_args.output_dir
    os.makedirs(out, exist_ok=True)

    # colour identifies the learner, linestyle the distribution
    dist_ids = sorted(results_list[0]['per_dist'].keys())
    styles = dict(zip(dist_ids, ['-', '--', ':']))
    fdr_curves, ineff_curves = [], []
    for (lbl, col), res in zip(LEARNER_LABELS, results_list):
        for d in dist_ids:
            name = DIST_NAMES.get(d, str(d))
            fdr_curves.append((f'{lbl} / {name}', col, styles[d], res['per_dist'][d]['fdrs']))
            ineff_curves.append((f'{lbl} / {name}', col, styles[d], res['per_dist'][d]['inefficiencies']))
    save_curve_plot(fdr_curves, 'SR', 'fdrs_per_dist_plot', training_args, shift_positions, (0, 0.5))
    save_curve_plot(ineff_curves, 'Inefficiency', 'inefficiencies_per_dist_plot', training_args,
                    shift_positions, (0, 1), alpha_line=False)

    if getattr(training_args, 'save_round_log', True):
        save_round_logs(results_list, out)
    return save_per_dist_summary(results_list, training_args, out)


def save_imbalance_outputs(results_list, training_args, shift_positions):
    """Overall and per-distribution FDR / inefficiency plots, round logs, and the JSON summary."""
    out = training_args.output_dir
    os.makedirs(out, exist_ok=True)

    overall_fdr = [(lbl, col, '-', res['fdrs']) for (lbl, col), res in zip(LEARNER_LABELS, results_list)]
    overall_fdr.append((r'$\mathtt{No\text{-}CA}$', 'orange', '-', results_list[-1]['empirical_fdrs']))
    save_curve_plot(overall_fdr, 'SR', 'fdrs_plot', training_args, shift_positions, (0, 0.5))

    overall_ineff = [(lbl, col, '-', 1 - np.array(res['efficiencies']))
                     for (lbl, col), res in zip(LEARNER_LABELS, results_list)]
    save_curve_plot(overall_ineff, 'Inefficiency', 'inefficiencies_plot', training_args,
                    shift_positions, (0, 1), alpha_line=False)

    return save_per_dist_outputs(results_list, training_args, shift_positions)


def build_gradual_shifted_dataset(nq_ds, trivia_ds, T, first_flag, seed):
    """Gradual shift: the probability of drawing from the `first_flag` dataset decays 1 -> 0.

    Same draw order as the original inline loop, so runs reproduce; the only addition is that
    the source of each round is now recorded in `dist_id` instead of being discarded, which is
    what makes per-distribution FDR computable under a continuous shift.
    """
    if first_flag == 'nq':
        first_ds, second_ds, first_id, second_id = nq_ds, trivia_ds, DIST_NQ, DIST_TRI
    else:
        first_ds, second_ds, first_id, second_id = trivia_ds, nq_ds, DIST_TRI, DIST_NQ

    picks = []
    for t in range(T):
        rng = random.Random(seed + t)
        p = 1.0 - t / (T - 1)
        if rng.random() < p:
            picks.append((True, rng.randrange(len(first_ds))))
        else:
            picks.append((False, rng.randrange(len(second_ds))))

    pool = concatenate_datasets([_tag_dist(first_ds, first_id), _tag_dist(second_ds, second_id)])
    second_offset = len(first_ds)
    order = [idx if is_first else second_offset + idx for is_first, idx in picks]
    return pool.select(order)


def stream_error_rate(results):
    """Error rate of the exact stream a run consumed: fraction of rounds whose true label is 0.

    This equals the SR of accepting everything, so it is the empirical upper bound on SR.
    """
    labels = np.asarray(results['round_log']['true_label'])
    return float(np.mean(labels == 0))


def save_alpha_sweep_plot(alpha_results, output_dir, filename, error_rate=None):
    """`error_rate`: optional scalar drawn as a horizontal dotted line (empirical SR upper bound)."""
    alphas = sorted(alpha_results.keys())
    fdr_lists = [alpha_results[alpha] for alpha in alphas]
    if len(alphas) > 1:
        width = max(0.01, min(np.diff(alphas)) * 0.6)
    else:
        width = 0.04

    with PdfPages(os.path.join(output_dir, f"{filename}.pdf")) as pdf:
        plt.figure()
        parts = plt.violinplot(
            fdr_lists,
            positions=alphas,
            widths=width,
            showmeans=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor("lightgreen")
            body.set_alpha(0.7)

        plt.plot(alphas, alphas, linestyle="--", color="k", linewidth=1, label=r"$y=x$")
        if error_rate is not None:
            plt.axhline(y=error_rate, linestyle=":", color="tab:red", linewidth=1.2,
                        label=f"Empirical SR upper-bound = {error_rate:.3f}")
        plt.xlabel(r"Target $\alpha$", fontsize=15)
        plt.ylabel("Empirical SR", fontsize=15)
        plt.xlim(min(alphas) - width, max(alphas) + width)
        plt.ylim(0, max(1.0, max(alphas) * 1.05))
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, zorder=1)
        plt.legend(loc="upper left", fontsize=10)
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f"{filename}.png"))
        pdf.savefig(bbox_inches="tight")
        plt.close()

#==================================================
# training andevaluation code
#==================================================
def main():       
    # read args
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UncertaintyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Use the configured output directory as the single run directory for
    # logs and artifacts so later saves do not fail on a missing path.
    if training_args.output_dir is None:
        training_args.output_dir = os.path.join(
            training_args.snapshot_root, training_args.method, training_args.exp_name
        )
    os.makedirs(training_args.output_dir, exist_ok=True)
    sys.stdout = util.Logger(os.path.join(training_args.output_dir, 'out'))
    sys.stderr = util.Logger(os.path.join(training_args.output_dir, 'out_err'))

    if data_args.log_args is True:
        print('==================================================')
        print(model_args)
        print(training_args)
        if 'data_args' in locals():
            print(data_args)
        
    logger = logging.getLogger(__name__)

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # init datasets
    # the two-distribution methods load NQ/TriviaQA themselves rather than a single dataset
    if training_args.method.startswith(('Shift', 'Mixture', 'Imbalanced_Shift')):
        nq_dataset_path = data_args.nq_dataset_name
        triviaqa_dataset_path = data_args.triviaqa_dataset_name
    else:
        raw_dataset = init_datasets_qna(data_args, model_args, training_args)

    np.random.seed(training_args.seed) # for unified restuls, will not affect much.
    
    # FIXME: Deprecated?
    if training_args.method.startswith('IID'):
        # INFO: IID experiment
        if training_args.method == 'IID-OSG':
            l = models.OSGLearner(
                params=training_args,
                name_postfix='osg'
            )
        elif training_args.method == 'IID-OSG-Bandit':
            l = models.OSGAdvBanditLearner(
                params=training_args,
                name_postfix='adv_sg'
            )
        elif training_args.method == 'IID-OSG-Semi-Bandit':
            l = models.OSGSemiAdvBanditLearner(
                params=training_args,
                name_postfix='adv_sg_bandit'
            )
        else:
            raise NotImplementedError
        
        l.train(
            raw_dataset,
            updated_params=types.SimpleNamespace(n=len(raw_dataset))
        )
    
    elif training_args.method == 'Plot':
        # INFO: Violin plot experiment
        fdrs1, fdrs2, fdrs3 = [], [], []
        
        for idx in tqdm(range(100)):
            base_seed = training_args.seed + idx
            size = len(raw_dataset)
            T = training_args.T

            if T <= size:
                dataset = raw_dataset.shuffle(seed=base_seed, keep_in_memory=True).select(range(T))
            else:
                reps = (T + size - 1) // size 
                pieces = [raw_dataset.shuffle(seed=base_seed + r, keep_in_memory=True) for r in range(reps)]
                big = concatenate_datasets(pieces)
                dataset = big.select(range(T))            
            # dataset = raw_dataset.shuffle(seed=training_args.seed+idx).select(range(training_args.T))
            # np.random.seed(training_args.seed+idx)
            
            l1, l2, l3 = models.OSGLearner(params=training_args, name_postfix='osg'), models.OSGAdvBanditLearner(params=training_args, name_postfix='adv_sg'), models.OSGSemiAdvBanditLearner(params=training_args, name_postfix='adv_sg_bandit')
            
            results1 = l1.train(dataset, updated_params=types.SimpleNamespace(n=len(dataset), seed=training_args.seed+idx))
            results2 = l2.train(dataset, updated_params=types.SimpleNamespace(n=len(dataset), seed=training_args.seed+idx))
            results3 = l3.train(dataset,updated_params=types.SimpleNamespace(n=len(dataset), seed=training_args.seed+idx))
            
            fdrs1.append(results1['fdrs'][-1])
            fdrs2.append(results2['fdrs'][-1])
            fdrs3.append(results3['fdrs'][-1])
            
        json.dump(fdrs1, open(os.path.join(training_args.output_dir, 'fdrs1.json'), 'w'))
        json.dump(fdrs2, open(os.path.join(training_args.output_dir, 'fdrs2.json'), 'w'))
        json.dump(fdrs3, open(os.path.join(training_args.output_dir, 'fdrs3.json'), 'w'))
             
        with PdfPages(os.path.join(training_args.output_dir, 'violin.pdf')) as pdf:
            os.makedirs(training_args.output_dir, exist_ok=True)
            plt.figure()
            vi = plt.violinplot([fdrs1, fdrs2, fdrs3], showmeans=True, showextrema=False)
            plt.xlabel('Method', fontsize=15)
            plt.xticks(np.arange(1,4), [r'$\mathtt{EW\text{-}CA}$', r'$\mathtt{Exp3\text{-}IX\text{-}CA}$', r'$\mathtt{ExAUL}$ (ours)'])
            plt.ylabel('SR', fontsize=15)
            
            if training_args.model == 'llama3.1-8B' and 'triviaqa' in data_args.dataset_name:
                ylim = 0.300
            elif training_args.model == 'gpt-5.4' and 'triviaqa' in data_args.dataset_name:
                ylim = 0.150
            elif training_args.model == 'llama3.1-8B' and 'nq' in data_args.dataset_name:
                ylim = 0.500
            elif training_args.model == 'gpt-5.4' and 'nq' in data_args.dataset_name:
                ylim = 0.300
            
            plt.ylim(0, ylim)
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')

            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            plt.rcParams.update({'font.size': 14})
            
            vi['bodies'][0].set_facecolor('aqua')
            vi['bodies'][1].set_facecolor('red')
            vi['bodies'][2].set_facecolor('lightgreen')
            
            plt.savefig(os.path.join(training_args.output_dir, 'violin.png'))
            pdf.savefig(bbox_inches='tight')
            
    elif training_args.method == 'Plot-over-alpha-parallel':
        # INFO: Violin plot of fdr, efficiency
            
        # FIXME trivial implementation
        # if training_args.model == 'llama3.1-8B' and 'triviaqa' in data_args.dataset_name:
        #     alphas = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
        # elif training_args.model == 'gpt-3.5' and 'triviaqa' in data_args.dataset_name:
        #     alphas = [0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
        # elif training_args.model == 'llama3.1-8B' and 'nq' in data_args.dataset_name:
        #     alphas = [0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
        # elif training_args.model == 'gpt-3.5' and 'nq' in data_args.dataset_name:
        #     alphas = [0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
        alphas = training_args.ablation_alphas
        
        # fdr1_total = []
        # fdr2_total = []
        # fdr3_total = []
        # fdr4_total = []
        # ineff1_total = []
        # ineff2_total = []
        # ineff3_total = []
        # ineff4_total = []
        # for aidx, alpha in enumerate(alphas):
        fdrs1 = []
        fdrs2 = []
        fdrs3 = []
        ineff1 = []
        ineff2 = []
        ineff3 = []
        
        if os.path.exists(os.path.join(training_args.output_dir, f'fdr3_{training_args.alpha}.json')):
            return
        
        for idx in tqdm(range(30)):
            dataset = raw_dataset.shuffle(seed=training_args.seed+idx, keep_in_memory=True).select(range(training_args.T))
            np.random.seed(training_args.seed+idx)
            
            l3 = models.OSGLearner(
                params=training_args,
                name_postfix='osg'
            )
            l2 = models.OSGAdvBanditLearner(
                params=training_args,
                name_postfix='adv_sg'
            )
            l1 = models.OSGSemiAdvBanditLearner(
                params=training_args,
                name_postfix='adv_sg_bandit'
            )
            
            results1 = l1.train(
                dataset,
                updated_params=types.SimpleNamespace(n=len(dataset), seed=training_args.seed+idx, alpha=training_args.alpha)
            )
            results2 = l2.train(
                dataset,
                updated_params=types.SimpleNamespace(n=len(dataset), seed=training_args.seed+idx, alpha=training_args.alpha)
            )
            results3 = l3.train(
                dataset,
                updated_params=types.SimpleNamespace(n=len(dataset), seed=training_args.seed+idx, alpha=training_args.alpha)
            )
            
            fdrs1.append(results1['fdrs'][-1])
            fdrs2.append(results2['fdrs'][-1])
            fdrs3.append(results3['fdrs'][-1])
            ineff1.append(1-results1['efficiencies'][-1])
            ineff2.append(1-results2['efficiencies'][-1])
            ineff3.append(1-results3['efficiencies'][-1])
        
            # fdr1_total.append(fdrs1)
            # fdr2_total.append(fdrs2)
            # fdr3_total.append(fdrs3)
            # fdr4_total.append(fdrs4)
            # ineff1_total.append(ineff1)
            # ineff2_total.append(ineff2)
            # ineff3_total.append(ineff3)
            # ineff4_total.append(ineff4)
            
        json.dump(fdrs1, open(os.path.join(training_args.output_dir, f'fdr1_{training_args.alpha}.json'), 'w'))
        json.dump(fdrs2, open(os.path.join(training_args.output_dir, f'fdr2_{training_args.alpha}.json'), 'w'))
        json.dump(fdrs3, open(os.path.join(training_args.output_dir, f'fdr3_{training_args.alpha}.json'), 'w'))
        json.dump(ineff1, open(os.path.join(training_args.output_dir, f'ineff1_{training_args.alpha}.json'), 'w'))
        json.dump(ineff2, open(os.path.join(training_args.output_dir, f'ineff2_{training_args.alpha}.json'), 'w'))
        json.dump(ineff3, open(os.path.join(training_args.output_dir, f'ineff3_{training_args.alpha}.json'), 'w'))
        
        # with PdfPages(os.path.join(training_args.output_dir, 'violin_fdr.pdf')) as pdf:
        #     plt.figure()
        #     plt.xlabel(r'$\alpha$', fontsize=15)
        #     plt.xticks(np.arange(1,7), alphas)
        #     plt.ylabel('FDR', fontsize=15)
        #     # plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')

        #     # plt.title('FDR')
        #     plt.legend(loc='upper right', fontsize=10)
        #     plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
        #     plt.tight_layout()
        #     # plt.rcParams.update({'font.size': 14})
        #     vi = plt.violinplot(fdr3_total, showmeans=True, showextrema=False)
                
        #     plt.plot([0.2, 0.35], [0.2, 0.35], linestyle='--', color='k', linewidth=0.5, label=r'$y=x$')
        #     # vi['bodies'][0].set_facecolor('aqua')
        #     # vi['bodies'][1].set_facecolor('red')
        #     [d.set_facecolor('lightgreen') for d in vi['bodies']]
    
        #     plt.savefig(os.path.join(training_args.output_dir, 'violin_fdr.png'))
        #     pdf.savefig(bbox_inches='tight')
        
        # with PdfPages(os.path.join(training_args.output_dir, 'violin_ineff.pdf')) as pdf:
        #     plt.figure()
        #     plt.xlabel(r'$\alpha$', fontsize=15)
        #     plt.xticks(np.arange(1,7), alphas)
        #     plt.ylabel('Inefficiency', fontsize=15)
        #     # plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')

        #     # plt.title('FDR')
        #     plt.legend(loc='upper right', fontsize=10)
        #     plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
        #     plt.tight_layout()
        #     # plt.rcParams.update({'font.size': 14})
        #     vi = plt.violinplot(ineff3_total, showmeans=True, showextrema=False)
                
        #     # vi['bodies'][0].set_facecolor('aqua')
        #     # vi['bodies'][1].set_facecolor('red')
        #     [d.set_facecolor('lightgreen') for d in vi['bodies']]
    
        #     plt.savefig(os.path.join(training_args.output_dir, 'violin_ineff.png'))
        #     pdf.savefig(bbox_inches='tight')
            
    # elif training_args.method == 'Plot-over-alpha':
    #     # INFO: Violin plot of fdr, efficiency
            
    #     # FIXME trivial implementation
    #     # if training_args.model == 'llama3.1-8B' and 'triviaqa' in data_args.dataset_name:
    #     #     alphas = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
    #     # elif training_args.model == 'gpt-3.5' and 'triviaqa' in data_args.dataset_name:
    #     #     alphas = [0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
    #     # elif training_args.model == 'llama3.1-8B' and 'nq' in data_args.dataset_name:
    #     #     alphas = [0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
    #     # elif training_args.model == 'gpt-3.5' and 'nq' in data_args.dataset_name:
    #     #     alphas = [0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
    #     alphas = training_args.ablation_alphas
        
    #     fdr1_total = []
    #     fdr2_total = []
    #     fdr3_total = []
    #     fdr4_total = []
    #     ineff1_total = []
    #     ineff2_total = []
    #     ineff3_total = []
    #     ineff4_total = []
    #     for aidx, alpha in enumerate(alphas):
    #         fdrs1 = []
    #         fdrs2 = []
    #         fdrs3 = []
    #         fdrs4 = []
    #         ineff1 = []
    #         ineff2 = []
    #         ineff3 = []
    #         ineff4 = []
    #         for idx in tqdm(range(30)):
    #             dataset = raw_dataset.shuffle(seed=training_args.seed+idx+100*aidx, keep_in_memory=True).select(range(training_args.T))
    #             np.random.seed(training_args.seed+idx+100*aidx)
                
    #             l1 = models.OSGLearner(
    #                 params=training_args,
    #                 name_postfix='osg'
    #             )
    #             l2 = models.OSGAdvBanditLearner(
    #                 params=training_args,
    #                 name_postfix='adv_sg'
    #             )
    #             l3 = models.OSGSemiAdvBanditLearner(
    #                 params=training_args,
    #                 name_postfix='adv_sg_bandit'
    #             )
    #             l4 = models.Exp3EwLearner(
    #                 params=training_args,
    #                 name_postfix='adv_sg_bandit_expew'
    #             )
                
    #             results1 = l1.train(
    #                 dataset,
    #                 updated_params=types.SimpleNamespace(n=len(dataset), seed=training_args.seed+idx+100*aidx, alpha=alpha)
    #             )
    #             results2 = l2.train(
    #                 dataset,
    #                 updated_params=types.SimpleNamespace(n=len(dataset), seed=training_args.seed+idx+100*aidx, alpha=alpha)
    #             )
    #             results3 = l3.train(
    #                 dataset,
    #                 updated_params=types.SimpleNamespace(n=len(dataset), seed=training_args.seed+idx+100*aidx, alpha=alpha)
    #             )
    #             results4 = l4.train(
    #                 dataset,
    #                 updated_params=types.SimpleNamespace(n=len(dataset), seed=training_args.seed+idx+100*aidx, alpha=alpha)
    #             )
                
    #             fdrs1.append(results1['fdrs'][-1])
    #             fdrs2.append(results2['fdrs'][-1])
    #             fdrs3.append(results3['fdrs'][-1])
    #             fdrs4.append(results4['fdrs'][-1])
    #             ineff1.append(1-results1['efficiencies'][-1])
    #             ineff2.append(1-results2['efficiencies'][-1])
    #             ineff3.append(1-results3['efficiencies'][-1])
    #             ineff4.append(1-results4['efficiencies'][-1])
            
    #         fdr1_total.append(fdrs1)
    #         fdr2_total.append(fdrs2)
    #         fdr3_total.append(fdrs3)
    #         fdr4_total.append(fdrs4)
    #         ineff1_total.append(ineff1)
    #         ineff2_total.append(ineff2)
    #         ineff3_total.append(ineff3)
    #         ineff4_total.append(ineff4)
            
    #     json.dump(fdr1_total, open(os.path.join(training_args.output_dir, 'fdr1_total.json'), 'w'))
    #     json.dump(fdr2_total, open(os.path.join(training_args.output_dir, 'fdr2_total.json'), 'w'))
    #     json.dump(fdr3_total, open(os.path.join(training_args.output_dir, 'fdr3_total.json'), 'w'))
    #     json.dump(fdr4_total, open(os.path.join(training_args.output_dir, 'fdr4_total.json'), 'w'))
    #     json.dump(ineff1_total, open(os.path.join(training_args.output_dir, 'ineff1_total.json'), 'w'))
    #     json.dump(ineff2_total, open(os.path.join(training_args.output_dir, 'ineff2_total.json'), 'w'))
    #     json.dump(ineff3_total, open(os.path.join(training_args.output_dir, 'ineff3_total.json'), 'w'))
    #     json.dump(ineff4_total, open(os.path.join(training_args.output_dir, 'ineff4_total.json'), 'w'))
        
    #     with PdfPages(os.path.join(training_args.output_dir, 'violin_fdr.pdf')) as pdf:
    #         plt.figure()
    #         plt.xlabel(r'$\alpha$', fontsize=15)
    #         plt.xticks(np.arange(1,7), alphas)
    #         plt.ylabel('FDR', fontsize=15)
    #         # plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')

    #         # plt.title('FDR')
    #         plt.legend(loc='upper right', fontsize=10)
    #         plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
    #         plt.tight_layout()
    #         # plt.rcParams.update({'font.size': 14})
    #         vi = plt.violinplot(fdr3_total, showmeans=True, showextrema=False)
                
    #         plt.plot([0.2, 0.35], [0.2, 0.35], linestyle='--', color='k', linewidth=0.5, label=r'$y=x$')
    #         # vi['bodies'][0].set_facecolor('aqua')
    #         # vi['bodies'][1].set_facecolor('red')
    #         [d.set_facecolor('lightgreen') for d in vi['bodies']]
    
    #         plt.savefig(os.path.join(training_args.output_dir, 'violin_fdr.png'))
    #         pdf.savefig(bbox_inches='tight')
        
    #     with PdfPages(os.path.join(training_args.output_dir, 'violin_ineff.pdf')) as pdf:
    #         plt.figure()
    #         plt.xlabel(r'$\alpha$', fontsize=15)
    #         plt.xticks(np.arange(1,7), alphas)
    #         plt.ylabel('Inefficiency', fontsize=15)
    #         # plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')

    #         # plt.title('FDR')
    #         plt.legend(loc='upper right', fontsize=10)
    #         plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
    #         plt.tight_layout()
    #         # plt.rcParams.update({'font.size': 14})
    #         vi = plt.violinplot(ineff3_total, showmeans=True, showextrema=False)
                
    #         # vi['bodies'][0].set_facecolor('aqua')
    #         # vi['bodies'][1].set_facecolor('red')
    #         [d.set_facecolor('lightgreen') for d in vi['bodies']]
    
    #         plt.savefig(os.path.join(training_args.output_dir, 'violin_ineff.png'))
    #         pdf.savefig(bbox_inches='tight')
        
    #     with PdfPages(os.path.join(training_args.output_dir, 'violin_fdr_expew.pdf')) as pdf:
    #         plt.figure()
    #         plt.xlabel(r'$\alpha$', fontsize=15)
    #         plt.xticks(np.arange(1,7), alphas)
    #         plt.ylabel('FDR', fontsize=15)
    #         plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')

    #         # plt.title('FDR')
    #         plt.legend(loc='upper right', fontsize=10)
    #         plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
    #         plt.tight_layout()
    #         # plt.rcParams.update({'font.size': 14})
    #         vi = plt.violinplot(fdr4_total, showmeans=True, showextrema=False)
                
                
    #         # vi['bodies'][0].set_facecolor('aqua')
    #         # vi['bodies'][1].set_facecolor('red')
    #         [d.set_facecolor('lightgreen') for d in vi['bodies']]
    
    #         plt.savefig(os.path.join(training_args.output_dir, 'violin_fdr_expew.png'))
    #         pdf.savefig(bbox_inches='tight')
        
    #     with PdfPages(os.path.join(training_args.output_dir, 'violin_ineff_expew.pdf')) as pdf:
    #         plt.figure()
    #         plt.xlabel(r'$\alpha$', fontsize=15)
    #         plt.xticks(np.arange(1,7), alphas)
    #         plt.ylabel('Inefficiency', fontsize=15)
    #         # plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')

    #         # plt.title('FDR')
    #         plt.legend(loc='upper right', fontsize=10)
    #         plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
    #         plt.tight_layout()
    #         # plt.rcParams.update({'font.size': 14})
    #         vi = plt.violinplot(ineff4_total, showmeans=True, showextrema=False)
                
    #         # vi['bodies'][0].set_facecolor('aqua')
    #         # vi['bodies'][1].set_facecolor('red')
    #         [d.set_facecolor('lightgreen') for d in vi['bodies']]
    
    #         plt.savefig(os.path.join(training_args.output_dir, 'violin_ineff_expew.png'))
    #         pdf.savefig(bbox_inches='tight')

    elif training_args.method == 'Baseline':
        l1, l2, l3 = models.OSGAdvBanditLearner(params=training_args, name_postfix='exp3-ix'), models.Exp3Learner(params=training_args, name_postfix='exp3'), models.Exp3Learner(params=training_args, name_postfix='exp3-p')
        
        results1, results2, results3 = l1.train(raw_dataset, updated_params=types.SimpleNamespace(n=len(raw_dataset), seed=training_args.seed)), l2.train(raw_dataset, updated_params=types.SimpleNamespace(n=len(raw_dataset), seed=training_args.seed)), l3.train(raw_dataset, updated_params=types.SimpleNamespace(n=len(raw_dataset), seed=training_args.seed))

        print(f"\n=== Final Results at T={len(results1['fdrs'])} ===")
        print(f"{'Method':<25} {'FDR':>8} {'Inefficiency':>14}")
        print("-" * 50)
        print(f"{'Exp3-IX':<25} {results1['fdrs'][-1]:>8.4f} {1-results1['efficiencies'][-1]:>14.4f}")
        print(f"{'Exp3-P':<25} {results3['fdrs'][-1]:>8.4f} {1-results3['efficiencies'][-1]:>14.4f}")
        print(f"{'Exp3':<25} {results2['fdrs'][-1]:>8.4f} {1-results2['efficiencies'][-1]:>14.4f}")
        print("=" * 50)

        with PdfPages(os.path.join(training_args.output_dir, 'fdrs_plot.pdf')) as pdf:
            plt.figure()
            plt.plot(results1['fdrs'], label=r'$\mathtt{Exp3-IX\text{-}SG}$', color='green')
            plt.plot(results3['fdrs'], label=r'$\mathtt{Exp3-P\text{-}SG}$', color='red')
            plt.plot(results2['fdrs'], label=r'$\mathtt{Exp3\text{-}SG}$', color='blue')
            plt.plot(results1['empirical_fdrs'], label=r'$\mathtt{No\text{-}SG}$', color='orange')

            ylim = max(max(results1['fdrs']), max(results2['fdrs']),  max(results3['fdrs']), max(results1['empirical_fdrs']))
            ylo = min(min(results1['fdrs']), min(results2['fdrs']), min(results3['fdrs']), min(results1['empirical_fdrs']), training_args.alpha)
            margin = (ylim - ylo) * 0.1 + 1e-4

            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('SR', fontsize=15)
            plt.ylim(ylo - margin, ylim + margin)
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(training_args.output_dir, 'fdrs_plot.png'))
            pdf.savefig(bbox_inches='tight')

        with PdfPages(os.path.join(training_args.output_dir, 'inefficiencies_plot.pdf')) as pdf:
            plt.figure()
            plt.plot(1-np.array(results1['efficiencies']), label=r'$\mathtt{Exp3-IX\text{-}SG}$', color='green')
            plt.plot(1-np.array(results3['efficiencies']), label=r'$\mathtt{Exp3-P\text{-}SG}$', color='red')
            plt.plot(1-np.array(results2['efficiencies']), label=r'$\mathtt{Exp3\text{-}SG}$', color='blue')
            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Inefficiency', fontsize=15)
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(training_args.output_dir, 'inefficiencies_plot.png'))
            pdf.savefig(bbox_inches='tight')
         
    elif training_args.method == 'ExSUL-Lambda':
        LAMBDA_CONFIGS = {
            r'$\lambda=T^{1/2}$':  types.SimpleNamespace(overLambda=True, lambda_T=2),
            r'$\lambda=T^{1/4}$':  types.SimpleNamespace(overLambda=True, lambda_T=3),
            r'$\lambda=T^{1/6}$':  types.SimpleNamespace(overLambda=True, lambda_T=5),
            r'$\lambda=T^{1/8}$':  types.SimpleNamespace(overLambda=True, lambda_T=4),
        }
        COLORS = ['purple', 'green', 'pink', 'skyblue']

        all_results = {}
        for label, lam_cfg in LAMBDA_CONFIGS.items():
            merged_dict = {k: v for k, v in vars(training_args).items()}
            merged_dict.update(n=len(raw_dataset), overLambda=lam_cfg.overLambda, lambda_T=lam_cfg.lambda_T)
            merged = types.SimpleNamespace(**merged_dict)
            l = models.OSGSemiAdvBanditLearner(params=merged, name_postfix=f'adv_sg_bandit_lam{lam_cfg.lambda_T}')
            all_results[label] = l.train(raw_dataset, updated_params=merged)

        print(f"\n=== Final Results at T={len(list(all_results.values())[0]['fdrs'])} ===")
        print(f"{'Method':<25} {'FDR':>8} {'Inefficiency':>14}")
        print("-" * 50)
        for label, res in all_results.items():
            print(f"{label:<25} {res['fdrs'][-1]:>8.4f} {1-res['efficiencies'][-1]:>14.4f}")
        print("=" * 50)

        with PdfPages(os.path.join(training_args.output_dir, 'fdrs_plot.pdf')) as pdf:
            plt.figure()
            for (label, res), color in zip(all_results.items(), COLORS):
                plt.plot(res['fdrs'], label=label, color=color)

            plt.plot(list(all_results.values())[0]['empirical_fdrs'], label=r'$\mathtt{No\text{-}CA}$', color='orange')
            
            ylim = max(max(res['fdrs']) for res in all_results.values())
            ylo = min(min(res['fdrs']) for res in all_results.values())
            margin = (ylim - ylo) * 0.1 + 1e-4
            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('SR', fontsize=15)
            plt.ylim(ylo - margin, ylim + margin)
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(training_args.output_dir, 'fdrs_plot.png'))
            pdf.savefig(bbox_inches='tight')

        with PdfPages(os.path.join(training_args.output_dir, 'inefficiencies_plot.pdf')) as pdf:
            plt.figure()
            for (label, res), color in zip(all_results.items(), COLORS):
                plt.plot(1-np.array(res['efficiencies']), label=label, color=color)
            best_eff = list(all_results.values())[0]['best_efficiency']
            plt.axhline(y=1-best_eff, color='k', linestyle='dashed', label='Best Efficiency')
            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Inefficiency', fontsize=15)
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(training_args.output_dir, 'inefficiencies_plot.png'))
            pdf.savefig(bbox_inches='tight')

    elif training_args.method == 'ExSUL':
        l = models.OSGSemiAdvBanditLearner(params=training_args, name_postfix='adv_sg_bandit')
        results = l.train(raw_dataset, updated_params=types.SimpleNamespace(n=len(raw_dataset), seed=training_args.seed))

        print(f"\n=== Final Results at T={len(results['fdrs'])} ===")
        print(f"{'Method':<25} {'FDR':>8} {'Inefficiency':>14}")
        print("-" * 50)
        print(f"{'ExSUL (ours)':<25} {results['fdrs'][-1]:>8.4f} {1-results['efficiencies'][-1]:>14.4f}")
        print("=" * 50)

        with PdfPages(os.path.join(training_args.output_dir, 'fdrs_plot.pdf')) as pdf:
            plt.figure()
            plt.plot(results['fdrs'], label=r'$\mathtt{ExSUL}$ (ours)', color='green')
            plt.plot(results['empirical_fdrs'], label=r'$\mathtt{No\text{-}SG}$', color='orange')
            ylim = max(max(results['fdrs']), max(results['empirical_fdrs']))
            ylo = min(min(results['fdrs']), min(results['empirical_fdrs']), training_args.alpha)
            margin = (ylim - ylo) * 0.1 + 1e-4
            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('SR', fontsize=15)
            plt.ylim(ylo - margin, ylim + margin)
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(training_args.output_dir, 'fdrs_plot.png'))
            pdf.savefig(bbox_inches='tight')

        with PdfPages(os.path.join(training_args.output_dir, 'inefficiencies_plot.pdf')) as pdf:
            plt.figure()
            plt.plot(1-np.array(results['efficiencies']), label=r'$\mathtt{ExSUL}$ (ours)', color='green')
            plt.axhline(y=1-results['best_efficiency'], color='k', linestyle='dashed', label='Best Efficiency')
            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Inefficiency', fontsize=15)
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(training_args.output_dir, 'inefficiencies_plot.png'))
            pdf.savefig(bbox_inches='tight')

    elif training_args.method == 'All':
        l1, l2, l3 = models.OSGLearner(params=training_args, name_postfix='osg'), models.OSGAdvBanditLearner(params=training_args, name_postfix='exp3_ew'), models.OSGSemiAdvBanditLearner(params=training_args, name_postfix='adv_sg_bandit')

        results1, results2, results3 = l1.train(raw_dataset, updated_params=types.SimpleNamespace(n=len(raw_dataset), seed=training_args.seed)), l2.train(raw_dataset, updated_params=types.SimpleNamespace(n=len(raw_dataset), seed=training_args.seed)), l3.train(raw_dataset, updated_params=types.SimpleNamespace(n=len(raw_dataset), seed=training_args.seed))

        print(f"\n=== Final Results at T={len(results3['fdrs'])} ===")
        print(f"{'Method':<25} {'FDR':>8} {'Inefficiency':>14}")
        print("-" * 50)
        print(f"{'EW-SG':<25} {results1['fdrs'][-1]:>8.4f} {1-results1['efficiencies'][-1]:>14.4f}")
        print(f"{'Exp3-SG':<25} {results2['fdrs'][-1]:>8.4f} {1-results2['efficiencies'][-1]:>14.4f}")
        print(f"{'ExAUL (ours)':<25} {results3['fdrs'][-1]:>8.4f} {1-results3['efficiencies'][-1]:>14.4f}")
        print("=" * 50)

        final_results = {
            'seed': training_args.seed,
            'T': len(results3['fdrs']),
            'alpha': training_args.alpha,
            'EW-SG': {
                'fdr': float(results1['fdrs'][-1]),
                'inefficiency': float(1 - results1['efficiencies'][-1]),
            },
            'Exp3-SG': {
                'fdr': float(results2['fdrs'][-1]),
                'inefficiency': float(1 - results2['efficiencies'][-1]),
            },
            'ExAUL': {
                'fdr': float(results3['fdrs'][-1]),
                'inefficiency': float(1 - results3['efficiencies'][-1]),
            },
        }
        json.dump(final_results, open(os.path.join(training_args.output_dir, 'final_results.json'), 'w'), indent=4)

        # Under a score changepoint the overall numbers average the two score functions together and
        # say nothing about how fast the learner recovers, so the segments (dist_id 0 = pre-switch,
        # 1 = post-switch) and the threshold trajectory are written out alongside.
        if getattr(training_args, 'score_mode', None) == 'switch':
            seg_results = {'alpha': training_args.alpha, 'T': training_args.T,
                           'seed': training_args.seed, 'switch_at': training_args.score_switch_at}
            for name, res in zip(('EW-SG', 'Exp3-SG', 'ExAUL'), (results1, results2, results3)):
                seg_results[name] = {
                    ('pre' if d == 0 else 'post'): {
                        'fdr': float(pd['fdrs'][-1]),
                        'inefficiency': float(pd['inefficiencies'][-1]),
                        'n': pd['n'], 'n_commit': pd['n_commit'],
                    }
                    for d, pd in res['per_dist'].items()
                }
            json.dump(seg_results, open(os.path.join(training_args.output_dir,
                                                     'segment_results.json'), 'w'), indent=4)
            if getattr(training_args, 'save_round_log', True):
                save_round_logs([results1, results2, results3], training_args.output_dir)

        # TODO: Reduce lines
        with PdfPages(os.path.join(training_args.output_dir, 'fdrs_plot.pdf')) as pdf:
            plt.figure()
            plt.plot(results3['fdrs'], label=r'$\mathtt{ExAUL}$ (ours)', color='green')
            plt.plot(results1['fdrs'], label=r'$\mathtt{EW\text{-}CA}$', color='blue')
            plt.plot(results2['fdrs'], label=r'$\mathtt{Exp3\text{-}CA}$', color='red')

            plt.plot(results3['empirical_fdrs'], label=r'$\mathtt{No\text{-}CA}$', color='orange')

            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('SR', fontsize=15)
            # plt.title('FDR')
            # Data-driven top: alpha alone is not enough (e.g. nq with lenient-judge feedback noise
            # pushes the FDR well above alpha). The first few hundred steps are ignored because a
            # single early mistake sends the running FDR to 1.0 and would flatten every curve.
            _warmup = min(300, len(results3['fdrs']) // 10)
            _fdr_curves = [results1['fdrs'], results2['fdrs'], results3['fdrs'], results3['empirical_fdrs']]
            _top = max([v for c in _fdr_curves for v in c[_warmup:] if np.isfinite(v)]
                       + [training_args.alpha])
            plt.ylim(0, _top * 1.15 + 1e-4)
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            # plt.rcParams.update({'font.size': 14})
            plt.savefig(os.path.join(training_args.output_dir, 'fdrs_plot.png'))
            pdf.savefig(bbox_inches='tight')

        with PdfPages(os.path.join(training_args.output_dir, 'inefficiencies_plot.pdf')) as pdf:
            plt.figure()
            plt.plot(1-np.array(results3['efficiencies']), label=r'$\mathtt{ExAUL}$ (ours)', color='green')
            plt.plot(1-np.array(results1['efficiencies']), label=r'$\mathtt{EW\text{-}CA}$', color='blue')
            plt.plot(1-np.array(results2['efficiencies']), label=r'$\mathtt{Exp3\text{-}CA}$', color='red')
            # plt.axhline(y=1-results1['best_efficiency'], color='k', linestyle='dashed', label='Best Efficiency')
            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Inefficiency', fontsize=15)
            plt.ylim(0, 1)
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            # plt.rcParams.update({'font.size': 14})
            plt.savefig(os.path.join(training_args.output_dir, 'inefficiencies_plot.png'))
            pdf.savefig(bbox_inches='tight')

        with PdfPages(os.path.join(training_args.output_dir, 'static_regret.pdf')) as pdf:
            plt.figure()
            plt.plot(results3['regrets_static'], label=r'$\mathtt{ExAUL}$ (ours)', color='green')
            plt.plot(results1['regrets_static'], label=r'$\mathtt{EW\text{-}CA}$', color='blue')

            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Regret', fontsize=15)
            _rs = [v for v in results3['regrets_static'] + results1['regrets_static'] if np.isfinite(v)]
            plt.ylim(0, (max(_rs) if _rs else 0.5) * 1.1 + 1e-4)
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()

            plt.savefig(os.path.join(training_args.output_dir, 'static_regret.png'))
            pdf.savefig(bbox_inches='tight')

        with PdfPages(os.path.join(training_args.output_dir, 'online_regret.pdf')) as pdf:
            plt.figure()
            plt.plot(results3['regrets_online'], label=r'$\mathtt{ExAUL}$ (ours)', color='green')
            plt.plot(results1['regrets_online'], label=r'$\mathtt{EW\text{-}CA}$', color='blue')

            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Regret', fontsize=15)
            _ro = [v for v in results3['regrets_online'] + results1['regrets_online'] if np.isfinite(v)]
            plt.ylim(0, (max(_ro) if _ro else 0.5) * 1.1 + 1e-4)
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()

            plt.savefig(os.path.join(training_args.output_dir, 'online_regret.png'))
            pdf.savefig(bbox_inches='tight')

        # with PdfPages(os.path.join(training_args.output_dir, 'weights_plot_osg.pdf')) as pdf:
        #     plt.figure()
        #     plt.plot(results1['weights'], label=r'$\mathtt{EW\text{-}SG}$', color='blue')
        #     plt.xlabel(r'$\tau$',fontsize=15)
        #     plt.ylabel('Weight',fontsize=15)
        #     # plt.title('FDR')
        #     plt.legend(loc='upper right', fontsize=10)
        #     plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
        #     plt.tight_layout()
        #     # plt.rcParams.update({'font.size': 14})
        #     plt.savefig(os.path.join(training_args.output_dir, 'weights_plot_osg.png'))
        #     pdf.savefig(bbox_inches='tight')
        # with PdfPages(os.path.join(training_args.output_dir, 'weights_plot_adv.pdf')) as pdf:
        #     plt.figure()
        #     plt.plot(results2['weights'], label=r'$\mathtt{Exp3\text{-}IX\text{-}SG}$', color='red')
        #     plt.xlabel(r'$\tau$',fontsize=15)
        #     plt.ylabel('Weight',fontsize=15)
        #     # plt.title('FDR')
        #     plt.legend(loc='upper right', fontsize=10)
        #     plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
        #     plt.tight_layout()
        #     # plt.rcParams.update({'font.size': 14})
        #     plt.savefig(os.path.join(training_args.output_dir, 'weights_plot_adv.png'))
        #     pdf.savefig(bbox_inches='tight')
        # with PdfPages(os.path.join(training_args.output_dir, 'weights_plot_semi.pdf')) as pdf:
        #     plt.figure()
        #     plt.plot(results3['weights'], label=r'$\mathtt{ExSUL}$ (ours)', color='green')
        #     plt.xlabel(r'$\tau$',fontsize=15)
        #     plt.ylabel('Weight',fontsize=15)
        #     # plt.title('FDR')
        #     plt.legend(loc='upper right', fontsize=10)
        #     plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
        #     plt.tight_layout()
        #     # plt.rcParams.update({'font.size': 14})
        #     plt.savefig(os.path.join(training_args.output_dir, 'weights_plot_semi.png'))
        #     pdf.savefig(bbox_inches='tight')    
        
    elif training_args.method == 'Ablation':
        l1, l2= models.OSGSemiAdvBanditLearner(params=training_args, name_postfix='adv_sg_bandit'), models.OSGAblationLearner(params=training_args, name_postfix='adv_sg_bandit_ablation')
        
        results1, results2 = l1.train(raw_dataset, updated_params=types.SimpleNamespace(n=len(raw_dataset), seed=training_args.seed)), l2.train(raw_dataset, updated_params=types.SimpleNamespace(n=len(raw_dataset), seed=training_args.seed))
        
        # TODO: Reduce lines
        with PdfPages(os.path.join(training_args.output_dir, 'fdrs_plot.pdf')) as pdf:
            plt.figure()
            plt.plot(results1['fdrs'], label=r'$\mathtt{ExAUL}$ (ours)', color='green')
            plt.plot(results2['fdrs'], label=r'$\mathtt{ExAUL}$ w/o $a_t$', color='purple')
            assert results1['empirical_fdrs'] == results2['empirical_fdrs'], "Empirical FDRs should be the same across methods."

            ylim = max(max(results1['fdrs']), max(results2['fdrs']), max(results1['empirical_fdrs']))
            ylo = min(min(results1['fdrs']), min(results2['fdrs']), min(results1['empirical_fdrs']), training_args.alpha)
            margin = (ylim - ylo) * 0.1 + 1e-4

            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('SR', fontsize=15)
            # plt.title('FDR')
            plt.ylim(ylo - margin, ylim + margin)
            # plt.ylim(0, 0.2) if 'triviaqa' in data_args.dataset_name else plt.ylim(0, 0.5)
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            # plt.axhline(y=ylim, color='k', alpha=0.7, linestyle='dotted', label='Empirical FDR upper-bound')
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            # plt.rcParams.update({'font.size': 14})
            plt.savefig(os.path.join(training_args.output_dir, 'fdrs_plot.png'))
            pdf.savefig(bbox_inches='tight')
        
        with PdfPages(os.path.join(training_args.output_dir, 'inefficiencies_plot.pdf')) as pdf:
            plt.figure()
            plt.plot(1-np.array(results1['efficiencies']), label=r'$\mathtt{ExAUL}$ (ours)', color='green')
            plt.plot(1-np.array(results2['efficiencies']), label=r'$\mathtt{ExAUL}$ w/o $a_t$', color='purple')
            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Inefficiency', fontsize=15)
            # plt.axhline(y=results1['best_efficiency'], color='k', linestyle='dashed', label='Best Efficiency')
            # plt.title('FDR')
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            # plt.rcParams.update({'font.size': 14})
            plt.savefig(os.path.join(training_args.output_dir, 'inefficiencies_plot.png'))
            pdf.savefig(bbox_inches='tight')
        
    elif training_args.method.startswith('Dialog'):
        # INFO: Dialog simulation
        
        if training_args.method == 'Dialog-OSG':
            G = models.OSGLearner(
                params=training_args,
                name_postfix='osg'
            )
        elif training_args.method == 'Dialog-OSG-Bandit':
            G = models.OSGAdvBanditLearner(
                params=training_args,
                name_postfix='adv_sg'
            )
        elif training_args.method == 'Dialog-OSG-Semi-Bandit':
            G = models.OSGSemiAdvBanditLearner(
                params=training_args,
                name_postfix='adv_sg_bandit'
            )
        # elif training_args.method == 'Dialog-OSG-Semi-Bandit-Exp3Ew':
        #     G = models.Exp3EwLearner(
        #         params=training_args,
        #         name_postfix='adv_sg_bandit_expew'
        #     )
        else:
            raise NotImplementedError
        
        l = models.DialogLearner(
            G,
            params=training_args,
        )
        l.simulate(
            raw_dataset,
            updated_params=types.SimpleNamespace(n=len(raw_dataset))
        )
    
    elif training_args.method.startswith('Static-Dialog'):
        
        if training_args.method == 'Static-Dialog-OSG':
            G = models.OSGLearner(
                params=training_args,
                name_postfix='osg'
            )
        elif training_args.method == 'Static-Dialog-OSG-Bandit':
            G = models.OSGAdvBanditLearner(
                params=training_args,
                name_postfix='adv_sg'
            )
        elif training_args.method == 'Static-Dialog-OSG-Semi-Bandit':
            G = models.OSGSemiAdvBanditLearner(
                params=training_args,
                name_postfix='adv_sg_bandit'
            )
        else:
            raise NotImplementedError
        
        l = models.StaticDialogLearner(
            G,
            params=training_args,
        )
        l.simulate(
            raw_dataset,
            updated_params=types.SimpleNamespace(n=len(raw_dataset))
        )

    elif training_args.method.startswith('Adversary-Dialog'):
        # INFO: Adversarial dialog simulation (an LLM adversary picks the next question)

        if training_args.method == 'Adversary-Dialog-OSG':
            G = models.OSGLearner(
                params=training_args,
                name_postfix='osg'
            )
        elif training_args.method == 'Adversary-Dialog-OSG-Bandit':
            G = models.OSGAdvBanditLearner(
                params=training_args,
                name_postfix='adv_sg'
            )
        elif training_args.method == 'Adversary-Dialog-OSG-Semi-Bandit':
            G = models.OSGSemiAdvBanditLearner(
                params=training_args,
                name_postfix='adv_sg_bandit'
            )
        else:
            raise NotImplementedError

        l = models.DialogAdversaryLearner(
            G,
            params=training_args,
        )
        l.simulate(
            raw_dataset,
            updated_params=types.SimpleNamespace(n=len(raw_dataset))
        )

    elif training_args.method == 'Shift':
        # Distribution shift dataset
        nq_ds = load_from_disk(data_args.nq_dataset_name)
        trivia_ds = load_from_disk(data_args.triviaqa_dataset_name)        
        full_ds = build_shifted_dataset(nq_ds, trivia_ds, data_args.chunk_size, data_args.num_chunks, training_args.first_flag, training_args.seed)
        # Known Horizon
        training_args.T = len(full_ds)
        
        l1, l2, l3 = models.OSGLearner(params=training_args, name_postfix='osg'), models.OSGAdvBanditLearner(params=training_args, name_postfix='adv_sg'), models.OSGSemiAdvBanditLearner(params=training_args, name_postfix='adv_sg_bandit')
    
        results1, results2, results3 = l1.train(full_ds, updated_params=types.SimpleNamespace(n=len(full_ds), seed=training_args.seed)), l2.train(full_ds, updated_params=types.SimpleNamespace(n=len(full_ds), seed=training_args.seed)), l3.train(full_ds, updated_params=types.SimpleNamespace(n=len(full_ds), seed=training_args.seed))
    
        # Shift positions
        shift_positions = [(i+1) * data_args.chunk_size for i in range(data_args.num_chunks * 2 - 1) ]

        with PdfPages(os.path.join(training_args.output_dir, 'fdrs_plot.pdf')) as pdf:
            os.makedirs(training_args.output_dir, exist_ok=True)
            plt.figure()
            plt.plot(results3['fdrs'], label=r'$\mathtt{ExAUL}$ (ours)', color='green')
            plt.plot(results2['fdrs'], label=r'$\mathtt{Exp3\text{-}IX\text{-}CA}$', color='red')
            plt.plot(results1['fdrs'], label=r'$\mathtt{EW\text{-}CA}$', color='blue')
            assert results1['empirical_fdrs'] == results2['empirical_fdrs'] == results3['empirical_fdrs'], "Empirical FDRs should be the same across methods."
            plt.plot(results3['empirical_fdrs'], label=r'$\mathtt{No\text{-}CA}$', color='orange')
            
            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('SR', fontsize=15)
            plt.ylim(0, 0.5) # fixed ylim for shift setup  
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            # Shift positions
            for pos in shift_positions:
                plt.axvline(x=pos, color='red', linestyle=':', linewidth=1, alpha=0.5)
                
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(training_args.output_dir, 'fdrs_plot.png'))
            pdf.savefig(bbox_inches='tight')
            
        with PdfPages(os.path.join(training_args.output_dir, 'inefficiencies_plot.pdf')) as pdf:
            plt.figure()
            plt.plot(1-np.array(results3['efficiencies']), label=r'$\mathtt{ExAUL}$ (ours)', color='green')
            plt.plot(1-np.array(results2['efficiencies']), label=r'$\mathtt{Exp3\text{-}IX\text{-}CA}$', color='red')
            plt.plot(1-np.array(results1['efficiencies']), label=r'$\mathtt{EW\text{-}CA}$', color='blue')
            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Inefficiency', fontsize=15)
            plt.ylim(0, 1)
            # Shift positions
            for pos in shift_positions:
                plt.axvline(x=pos, color='red', linestyle=':', linewidth=1, alpha=0.5)
            
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(training_args.output_dir, 'inefficiencies_plot.png'))
            pdf.savefig(bbox_inches='tight')

        with PdfPages(os.path.join(training_args.output_dir, 'static_regret.pdf')) as pdf:
            plt.figure()
            plt.plot(results3['regrets_static'], label=r'$\mathtt{ExAUL}$ (ours)', color='green')
            plt.plot(results2['regrets_static'], label=r'$\mathtt{Exp3\text{-}IX\text{-}CA}$', color='red')
            plt.plot(results1['regrets_static'], label=r'$\mathtt{EW\text{-}CA}$', color='blue')

            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Regret', fontsize=15)
            plt.ylim(0, 0.1) # fixed ylim for shift setup  
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            # Shift positions
            for pos in shift_positions:
                plt.axvline(x=pos, color='red', linestyle=':', linewidth=1, alpha=0.5)
                
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()

            plt.savefig(os.path.join(training_args.output_dir, 'static_regret.png'))
            pdf.savefig(bbox_inches='tight')

        with PdfPages(os.path.join(training_args.output_dir, 'online_regret.pdf')) as pdf:
            plt.figure()
            plt.plot(results3['regrets_online'], label=r'$\mathtt{ExAUL}$ (ours)', color='green')
            plt.plot(results2['regrets_online'], label=r'$\mathtt{Exp3\text{-}IX\text{-}CA}$', color='red')
            plt.plot(results1['regrets_online'], label=r'$\mathtt{EW\text{-}CA}$', color='blue')

            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Regret', fontsize=15)
            plt.ylim(0, 0.1) # fixed ylim for shift setup  
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            # Shift positions
            for pos in shift_positions:
                plt.axvline(x=pos, color='red', linestyle=':', linewidth=1, alpha=0.5)
                
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()

            plt.savefig(os.path.join(training_args.output_dir, 'online_regret.png'))
            pdf.savefig(bbox_inches='tight')

        save_per_dist_outputs([results1, results2, results3], training_args, shift_positions)

    elif training_args.method == 'Mixture':
        # Q3: steady-state imbalanced stream. The mixing ratio is constant, so there is no shift
        # to adapt to -- what is scarce is the per-round information about the minority distribution.
        nq_ds = load_from_disk(data_args.nq_dataset_name)
        trivia_ds = load_from_disk(data_args.triviaqa_dataset_name)
        full_ds = build_mixture_dataset(
            nq_ds, trivia_ds, training_args.T, data_args.p_major, data_args.major_dist, training_args.seed
        )

        l1, l2, l3 = models.OSGLearner(params=training_args, name_postfix='osg'), models.OSGAdvBanditLearner(params=training_args, name_postfix='adv_sg'), models.OSGSemiAdvBanditLearner(params=training_args, name_postfix='adv_sg_bandit')

        updated = types.SimpleNamespace(n=len(full_ds), seed=training_args.seed)
        results1, results2, results3 = l1.train(full_ds, updated_params=updated), l2.train(full_ds, updated_params=updated), l3.train(full_ds, updated_params=updated)

        # no shift markers: the stream is stationary
        save_imbalance_outputs([results1, results2, results3], training_args, [])

    elif training_args.method == 'Imbalanced_Shift':
        # Q3: discrete shift where the two distributions get different block lengths, so the
        # minority distribution's block is short relative to the horizon.
        nq_ds = load_from_disk(data_args.nq_dataset_name)
        trivia_ds = load_from_disk(data_args.triviaqa_dataset_name)
        chunk_nq = data_args.chunk_size_nq or data_args.chunk_size
        chunk_tri = data_args.chunk_size_tri or data_args.chunk_size
        full_ds, shift_positions = build_imbalanced_shifted_dataset(
            nq_ds, trivia_ds, chunk_nq, chunk_tri, data_args.num_chunks, training_args.first_flag, training_args.seed
        )
        # Known horizon; must be set before the learners read params.T
        training_args.T = len(full_ds)
        print(f"# imbalanced shift: nq_block={chunk_nq}, tri_block={chunk_tri}, "
              f"num_chunks={data_args.num_chunks}, T={training_args.T}")

        l1, l2, l3 = models.OSGLearner(params=training_args, name_postfix='osg'), models.OSGAdvBanditLearner(params=training_args, name_postfix='adv_sg'), models.OSGSemiAdvBanditLearner(params=training_args, name_postfix='adv_sg_bandit')

        updated = types.SimpleNamespace(n=len(full_ds), seed=training_args.seed)
        results1, results2, results3 = l1.train(full_ds, updated_params=updated), l2.train(full_ds, updated_params=updated), l3.train(full_ds, updated_params=updated)

        save_imbalance_outputs([results1, results2, results3], training_args, shift_positions)

    elif training_args.method == 'Shift_Linear':
        # Distribution shift dataset
        nq_ds = load_from_disk(data_args.nq_dataset_name)
        trivia_ds = load_from_disk(data_args.triviaqa_dataset_name)
        full_ds = build_gradual_shifted_dataset(nq_ds, trivia_ds, training_args.T, training_args.first_flag, training_args.seed)

        l1, l2, l3 = models.OSGLearner(params=training_args, name_postfix='osg'), models.OSGAdvBanditLearner(params=training_args, name_postfix='adv_sg'), models.OSGSemiAdvBanditLearner(params=training_args, name_postfix='adv_sg_bandit')
    
        results1, results2, results3 = l1.train(full_ds, updated_params=types.SimpleNamespace(n=len(full_ds), seed=training_args.seed)), l2.train(full_ds, updated_params=types.SimpleNamespace(n=len(full_ds), seed=training_args.seed)), l3.train(full_ds, updated_params=types.SimpleNamespace(n=len(full_ds), seed=training_args.seed))
    
        with PdfPages(os.path.join(training_args.output_dir, 'fdrs_plot.pdf')) as pdf:
            os.makedirs(training_args.output_dir, exist_ok=True)
            plt.figure()
            plt.plot(results3['fdrs'], label=r'$\mathtt{ExAUL}$ (ours)', color='green')
            plt.plot(results2['fdrs'], label=r'$\mathtt{Exp3\text{-}IX\text{-}CA}$', color='red')
            plt.plot(results1['fdrs'], label=r'$\mathtt{EW\text{-}CA}$', color='blue')

            assert results1['empirical_fdrs'] == results2['empirical_fdrs'] == results3['empirical_fdrs'], "Empirical FDRs should be the same across methods."
            plt.plot(results3['empirical_fdrs'], label=r'$\mathtt{No\text{-}CA}$', color='orange')
                        
            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('SR', fontsize=15)
            plt.ylim(0, 0.5) # fixed ylim for shift setup  
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
   
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(training_args.output_dir, 'fdrs_plot.png'))
            pdf.savefig(bbox_inches='tight')
            
        with PdfPages(os.path.join(training_args.output_dir, 'inefficiencies_plot.pdf')) as pdf:
            plt.figure()
            plt.plot(1-np.array(results3['efficiencies']), label=r'$\mathtt{ExAUL}$ (ours)', color='green')
            plt.plot(1-np.array(results2['efficiencies']), label=r'$\mathtt{Exp3\text{-}IX\text{-}CA}$', color='red')
            plt.plot(1-np.array(results1['efficiencies']), label=r'$\mathtt{EW\text{-}CA}$', color='blue')
            
            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Inefficiency', fontsize=15)
            plt.ylim(0, 1)
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(training_args.output_dir, 'inefficiencies_plot.png'))
            pdf.savefig(bbox_inches='tight')

        with PdfPages(os.path.join(training_args.output_dir, 'static_regret.pdf')) as pdf:
            plt.figure()
            plt.plot(results3['regrets_static'], label=r'$\mathtt{ExAUL}$ (ours)', color='green')
            plt.plot(results2['regrets_static'], label=r'$\mathtt{Exp3\text{-}IX\text{-}CA}$', color='red')
            plt.plot(results1['regrets_static'], label=r'$\mathtt{EW\text{-}CA}$', color='blue')

            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Regret', fontsize=15)
            plt.ylim(0, 0.1) # fixed ylim for shift setup  

            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()

            plt.savefig(os.path.join(training_args.output_dir, 'static_regret.png'))
            pdf.savefig(bbox_inches='tight')

        with PdfPages(os.path.join(training_args.output_dir, 'online_regret.pdf')) as pdf:
            plt.figure()
            plt.plot(results3['regrets_online'], label=r'$\mathtt{ExAUL}$ (ours)', color='green')
            plt.plot(results2['regrets_online'], label=r'$\mathtt{Exp3\text{-}IX\text{-}CA}$', color='red')
            plt.plot(results1['regrets_online'], label=r'$\mathtt{EW\text{-}CA}$', color='blue')

            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Regret', fontsize=15)
            plt.ylim(0, 0.1) # fixed ylim for shift setup  
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()

            plt.savefig(os.path.join(training_args.output_dir, 'online_regret.png'))
            pdf.savefig(bbox_inches='tight')

        # the gradual shift has no block boundaries, so there are no shift markers to draw
        save_per_dist_outputs([results1, results2, results3], training_args, [])

    elif training_args.method == 'Shift_Plot':
        fdrs1, fdrs2, fdrs3 = [], [], []
        
        for idx in tqdm(range(100)):
            # Distribution shift dataset
            nq_ds = load_from_disk(data_args.nq_dataset_name) 
            trivia_ds = load_from_disk(data_args.triviaqa_dataset_name)
            full_ds = build_shifted_dataset(nq_ds, trivia_ds, data_args.chunk_size, data_args.num_chunks, training_args.first_flag, training_args.seed+idx)
            np.random.seed(training_args.seed+idx)
            # Known Horizon
            training_args.T = len(full_ds)
            
            l1, l2, l3 = models.OSGLearner(params=training_args, name_postfix='osg'), models.OSGAdvBanditLearner(params=training_args, name_postfix='adv_sg'), models.OSGSemiAdvBanditLearner(params=training_args, name_postfix='adv_sg_bandit')
        
            results1, results2, results3 = l1.train(full_ds, updated_params=types.SimpleNamespace(n=len(full_ds), seed=training_args.seed+idx)), l2.train(full_ds, updated_params=types.SimpleNamespace(n=len(full_ds), seed=training_args.seed+idx)), l3.train(full_ds, updated_params=types.SimpleNamespace(n=len(full_ds), seed=training_args.seed+idx))
    
            fdrs1.append(results1['fdrs'][-1])
            fdrs2.append(results2['fdrs'][-1])
            fdrs3.append(results3['fdrs'][-1])
            
        json.dump(fdrs1, open(os.path.join(training_args.output_dir, 'fdrs1.json'), 'w'))
        json.dump(fdrs2, open(os.path.join(training_args.output_dir, 'fdrs2.json'), 'w'))
        json.dump(fdrs3, open(os.path.join(training_args.output_dir, 'fdrs3.json'), 'w'))
            
        with PdfPages(os.path.join(training_args.output_dir, 'violin.pdf')) as pdf:
            os.makedirs(training_args.output_dir, exist_ok=True)
            plt.figure()
            vi = plt.violinplot([fdrs1, fdrs2, fdrs3], showmeans=True, showextrema=False)
            plt.xlabel('Method', fontsize=15)
            plt.xticks(np.arange(1,4), [r'$\mathtt{EW\text{-}CA}$', r'$\mathtt{Exp3\text{-}IX\text{-}CA}$', r'$\mathtt{ExAUL}$ (ours)'])
            plt.ylabel('SR', fontsize=15)
            
            if training_args.model == 'llama3.1-8B':
                ylim = 0.500
            elif training_args.model == 'gpt-5.4':
                ylim = 0.300
            
            plt.ylim(0, ylim)
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            plt.rcParams.update({'font.size': 14})
            
            vi['bodies'][0].set_facecolor('aqua')
            vi['bodies'][1].set_facecolor('red')
            vi['bodies'][2].set_facecolor('lightgreen')
            
            plt.savefig(os.path.join(training_args.output_dir, 'violin.png'))
            pdf.savefig(bbox_inches='tight')

    elif training_args.method == 'Shift_Plot_Linear':
        fdrs1, fdrs2, fdrs3 = [], [], []
        
        for idx in tqdm(range(100)):
            # Distribution shift dataset
            nq_ds = load_from_disk(data_args.nq_dataset_name)
            trivia_ds = load_from_disk(data_args.triviaqa_dataset_name)
            # NOTE: the stream is seeded with `seed + idx` so the 100 repeats actually differ.
            # The previous inline loop reused `idx` as a row index, which clobbered the repeat
            # counter and left every repeat on the same stream.
            full_ds = build_gradual_shifted_dataset(nq_ds, trivia_ds, training_args.T, training_args.first_flag, training_args.seed + idx)

            l1, l2, l3 = models.OSGLearner(params=training_args, name_postfix='osg'), models.OSGAdvBanditLearner(params=training_args, name_postfix='adv_sg'), models.OSGSemiAdvBanditLearner(params=training_args, name_postfix='adv_sg_bandit')
        
            results1, results2, results3 = l1.train(full_ds, updated_params=types.SimpleNamespace(n=len(full_ds), seed=training_args.seed+idx)), l2.train(full_ds, updated_params=types.SimpleNamespace(n=len(full_ds), seed=training_args.seed+idx)), l3.train(full_ds, updated_params=types.SimpleNamespace(n=len(full_ds), seed=training_args.seed+idx))
    
            fdrs1.append(results1['fdrs'][-1])
            fdrs2.append(results2['fdrs'][-1])
            fdrs3.append(results3['fdrs'][-1])
            
        json.dump(fdrs1, open(os.path.join(training_args.output_dir, 'fdrs1.json'), 'w'))
        json.dump(fdrs2, open(os.path.join(training_args.output_dir, 'fdrs2.json'), 'w'))
        json.dump(fdrs3, open(os.path.join(training_args.output_dir, 'fdrs3.json'), 'w'))
            
        with PdfPages(os.path.join(training_args.output_dir, 'violin.pdf')) as pdf:
            os.makedirs(training_args.output_dir, exist_ok=True)
            plt.figure()
            vi = plt.violinplot([fdrs1, fdrs2, fdrs3], showmeans=True, showextrema=False)
            plt.xlabel('Method', fontsize=15)
            plt.xticks(np.arange(1,4), [r'$\mathtt{EW\text{-}CA}$', r'$\mathtt{Exp3\text{-}IX\text{-}CA}$', r'$\mathtt{ExAUL}$ (ours)'])
            plt.ylabel('SR', fontsize=15)
            
            if training_args.model == 'llama3.1-8B':
                ylim = 0.300
            elif training_args.model == 'gpt-5.4':
                ylim = 0.500
            
            plt.ylim(0, ylim)
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            plt.rcParams.update({'font.size': 14})
            
            vi['bodies'][0].set_facecolor('aqua')
            vi['bodies'][1].set_facecolor('red')
            vi['bodies'][2].set_facecolor('lightgreen')
            
            plt.savefig(os.path.join(training_args.output_dir, 'violin.png'))
            pdf.savefig(bbox_inches='tight')
          
    elif training_args.method == 'Plot-over-alpha':
        alphas = training_args.ablation_alphas or [training_args.alpha]
        alpha_results = {}
        used_error_rates = {}  # alpha -> per-seed error rate of the stream actually used

        for alpha in alphas:
            training_args.alpha = alpha
            fdrs = []

            for idx in tqdm(range(30), desc=f"alpha={alpha:.2f}"):
                base_seed = training_args.seed + idx
                size = len(raw_dataset)
                T = training_args.T

                if T <= size:
                    dataset = raw_dataset.shuffle(seed=base_seed, keep_in_memory=True).select(range(T))
                else:
                    reps = (T + size - 1) // size
                    pieces = [raw_dataset.shuffle(seed=base_seed + r, keep_in_memory=True) for r in range(reps)]
                    big = concatenate_datasets(pieces)
                    dataset = big.select(range(T))

                l = models.OSGSemiAdvBanditLearner(params=training_args, name_postfix='adv_sg_bandit')
                results = l.train(dataset, updated_params=types.SimpleNamespace(n=len(dataset), seed=training_args.seed+idx))
                fdrs.append(results['fdrs'][-1])
                used_error_rates.setdefault(alpha, []).append(stream_error_rate(results))

            alpha_results[alpha] = fdrs
            json.dump(
                fdrs,
                open(os.path.join(training_args.output_dir, f"fdrs_alpha_{alpha:.2f}.json"), 'w')
            )

        error_rate = float(np.mean([r for rs in used_error_rates.values() for r in rs]))
        print(f"# error rate of used streams (empirical SR upper-bound): {error_rate:.4f}")
        json.dump({'mean': error_rate, 'per_alpha': used_error_rates},
                  open(os.path.join(training_args.output_dir, 'error_rates.json'), 'w'))
        json.dump(alpha_results, open(os.path.join(training_args.output_dir, 'fdrs_by_alpha.json'), 'w'))
        save_alpha_sweep_plot(alpha_results, training_args.output_dir, 'fdr_control_over_alpha', error_rate=error_rate)
    
    elif training_args.method == 'Shift_Plot_over_alpha':
        alphas = training_args.ablation_alphas or [training_args.alpha]
        alpha_results = {}
        used_error_rates = {}  # alpha -> per-seed error rate of the shifted stream actually used

        for alpha in alphas:
            training_args.alpha = alpha
            fdrs = []

            for idx in tqdm(range(30), desc=f"alpha={alpha:.2f}"):
                nq_ds = load_from_disk(data_args.nq_dataset_name)
                trivia_ds = load_from_disk(data_args.triviaqa_dataset_name)
                full_ds = build_shifted_dataset(
                    nq_ds,
                    trivia_ds,
                    data_args.chunk_size,
                    data_args.num_chunks,
                    training_args.first_flag,
                    training_args.seed+idx,
                )
                np.random.seed(training_args.seed+idx)
                training_args.T = len(full_ds)
                l = models.OSGSemiAdvBanditLearner(params=training_args, name_postfix='adv_sg_bandit')
                results = l.train(full_ds, updated_params=types.SimpleNamespace(n=len(full_ds), seed=training_args.seed+idx))
                fdrs.append(results['fdrs'][-1])
                used_error_rates.setdefault(alpha, []).append(stream_error_rate(results))

            alpha_results[alpha] = fdrs
            json.dump(
                fdrs,
                open(os.path.join(training_args.output_dir, f"fdrs_alpha_{alpha:.2f}.json"), 'w')
            )

        error_rate = float(np.mean([r for rs in used_error_rates.values() for r in rs]))
        print(f"# error rate of used streams (empirical SR upper-bound): {error_rate:.4f}")
        json.dump({'mean': error_rate, 'per_alpha': used_error_rates},
                  open(os.path.join(training_args.output_dir, 'error_rates.json'), 'w'))
        json.dump(alpha_results, open(os.path.join(training_args.output_dir, 'fdrs_by_alpha.json'), 'w'))
        save_alpha_sweep_plot(alpha_results, training_args.output_dir, 'fdr_control_over_alpha', error_rate=error_rate)
     
    elif training_args.method == 'Plot_over_lambda':  
        import copy
        cfgs = [1, 2, 3, 4]
        results = []
    
        for lambda_T in cfgs:
            p = copy.deepcopy(training_args)
            p.lambda_T = lambda_T
            learner = models.OSGSemiAdvBanditLearner(params=p, name_postfix=f'adv_sg_bandit_lambda_{lambda_T}')
            result =  learner.train(raw_dataset, updated_params=types.SimpleNamespace(n=len(raw_dataset)))
            results.append(result)
        
        results1, results2, results3, results4 = results
        
        with PdfPages(os.path.join(training_args.output_dir, 'fdrs_plot.pdf')) as pdf:
            plt.figure()
            
            # plt.plot(results1['fdrs'], label=r'$\mathtt{ExSUL}(\lambda=T)$', color='tab:red')
            plt.plot(results2['fdrs'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/2})$', color='skyblue')
            plt.plot(results3['fdrs'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/4})$', color='green')
            plt.plot(results4['fdrs'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/8})$', color='pink')
            
            assert results1['empirical_fdrs'] == results2['empirical_fdrs'] == results3['empirical_fdrs'], "Empirical FDRs should be the same across methods."
            plt.plot(results3['empirical_fdrs'], label=r'$\mathtt{No\text{-}CA}$', color='orange')

            ylim = max(max(results2['fdrs']), max(results3['fdrs']), max(results4['fdrs']), max(results3['empirical_fdrs']))
            ylo = min(min(results2['fdrs']), min(results3['fdrs']), min(results4['fdrs']), min(results3['empirical_fdrs']), training_args.alpha)
            margin = (ylim - ylo) * 0.1 + 1e-4
                
            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('SR', fontsize=15)
            plt.ylim(ylo - margin, ylim + margin)
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()

            plt.savefig(os.path.join(training_args.output_dir, 'fdrs_plot.png'))
            pdf.savefig(bbox_inches='tight')
        
        with PdfPages(os.path.join(training_args.output_dir, 'inefficiencies_plot.pdf')) as pdf:
            plt.figure()
            
            # plt.plot(1-np.array(results1['efficiencies']), label=r'$\mathtt{ExSUL}(\lambda=T)$', color='tab:red')
            plt.plot(1-np.array(results2['efficiencies']), label=r'$\mathtt{ExAUL}(\lambda=T^{1/2})$', color='skyblue')
            plt.plot(1-np.array(results3['efficiencies']), label=r'$\mathtt{ExAUL}(\lambda=T^{1/4})$', color='green')
            plt.plot(1-np.array(results4['efficiencies']), label=r'$\mathtt{ExAUL}(\lambda=T^{1/8})$', color='pink')
            
            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Inefficiency', fontsize=15)

            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()

            plt.savefig(os.path.join(training_args.output_dir, 'inefficiencies_plot.png'))
            pdf.savefig(bbox_inches='tight')
    
        with PdfPages(os.path.join(training_args.output_dir, 'static_regret.pdf')) as pdf:
            plt.figure()
            plt.plot(results2['regrets_static'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/2})$', color='skyblue')
            plt.plot(results3['regrets_static'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/4})$', color='green')
            plt.plot(results4['regrets_static'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/8})$', color='pink')

            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Regret', fontsize=15)
            plt.ylim(0, 0.5)

            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()

            plt.savefig(os.path.join(training_args.output_dir, 'static_regret.png'))
            pdf.savefig(bbox_inches='tight')

        with PdfPages(os.path.join(training_args.output_dir, 'online_regret.pdf')) as pdf:
            plt.figure()
            
            plt.plot(results2['regrets_online'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/2})$', color='skyblue')
            plt.plot(results3['regrets_online'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/4})$', color='green')
            plt.plot(results4['regrets_online'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/8})$', color='pink')

            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Regret', fontsize=15)
            plt.ylim(0, 0.5)
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()

            plt.savefig(os.path.join(training_args.output_dir, 'online_regret.png'))
            pdf.savefig(bbox_inches='tight')
     
    elif training_args.method == 'Shift_Plot_over_lambda':
        import copy
        cfgs = [1, 2, 3, 4]
        results = []
    
        for lambda_T in cfgs:
            p = copy.deepcopy(training_args)
            p.lambda_T = lambda_T
            learner = models.OSGSemiAdvBanditLearner(params=p, name_postfix=f'adv_sg_bandit_lambda_{lambda_T}')
            
            # Distribution shift dataset
            nq_ds = load_from_disk(data_args.nq_dataset_name)
            trivia_ds = load_from_disk(data_args.triviaqa_dataset_name)        
            full_ds = build_shifted_dataset(nq_ds, trivia_ds, data_args.chunk_size, data_args.num_chunks, training_args.first_flag, training_args.seed)
            training_args.T = len(full_ds)
            
            result =  learner.train(full_ds, updated_params=types.SimpleNamespace(n=len(full_ds)))
            results.append(result)
        
        results1, results2, results3, results4 = results
        shift_positions = [(i+1) * data_args.chunk_size for i in range(data_args.num_chunks * 2 - 1) ]
        
        with PdfPages(os.path.join(training_args.output_dir, 'fdrs_plot.pdf')) as pdf:
            plt.figure()
            
            # plt.plot(results1['fdrs'], label=r'$\mathtt{ExSUL}(\lambda=T)$', color='tab:red')
            plt.plot(results2['fdrs'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/2})$', color='skyblue')
            plt.plot(results3['fdrs'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/4})$', color='green')
            plt.plot(results4['fdrs'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/8})$', color='pink')
            
            # assert results1['empirical_fdrs'] == results2['empirical_fdrs'] == results3['empirical_fdrs'], "Empirical FDRs should be the same across methods."
            # plt.plot(results3['empirical_fdrs'], label=r'$\mathtt{No\text{-}SG}$', color='orange')
            
            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('SR', fontsize=15)
            plt.ylim(0, 0.5) # fixed ylim for shift setup  
            
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            # Shift positions
            for pos in shift_positions:
                plt.axvline(x=pos, color='red', linestyle=':', linewidth=1, alpha=0.5)
                
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(training_args.output_dir, 'fdrs_plot.png'))
            pdf.savefig(bbox_inches='tight')
        
        with PdfPages(os.path.join(training_args.output_dir, 'inefficiencies_plot.pdf')) as pdf:
            plt.figure()
            
            # plt.plot(1-np.array(results1['efficiencies']), label=r'$\mathtt{ExSUL}(\lambda=T)$', color='tab:red')
            plt.plot(1-np.array(results2['efficiencies']), label=r'$\mathtt{ExAUL}(\lambda=T^{1/2})$', color='skyblue')
            plt.plot(1-np.array(results3['efficiencies']), label=r'$\mathtt{ExAUL}(\lambda=T^{1/4})$', color='green')
            plt.plot(1-np.array(results4['efficiencies']), label=r'$\mathtt{ExAUL}(\lambda=T^{1/8})$', color='pink')
            
            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Inefficiency', fontsize=15)
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            # Shift positions
            for pos in shift_positions:
                plt.axvline(x=pos, color='red', linestyle=':', linewidth=1, alpha=0.5)
                
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()

            plt.savefig(os.path.join(training_args.output_dir, 'inefficiencies_plot.png'))
            pdf.savefig(bbox_inches='tight')
    
        with PdfPages(os.path.join(training_args.output_dir, 'static_regret.pdf')) as pdf:
            plt.figure()
            plt.plot(results2['regrets_static'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/2})$', color='skyblue')
            plt.plot(results3['regrets_static'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/4})$', color='green')
            plt.plot(results4['regrets_static'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/8})$', color='pink')

            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Regret', fontsize=15)
            plt.ylim(0, 0.5)
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            # Shift positions
            for pos in shift_positions:
                plt.axvline(x=pos, color='red', linestyle=':', linewidth=1, alpha=0.5)
                
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()

            plt.savefig(os.path.join(training_args.output_dir, 'static_regret.png'))
            pdf.savefig(bbox_inches='tight')

        with PdfPages(os.path.join(training_args.output_dir, 'online_regret.pdf')) as pdf:
            plt.figure()
            
            plt.plot(results2['regrets_online'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/2})$', color='skyblue')
            plt.plot(results3['regrets_online'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/4})$', color='green')
            plt.plot(results4['regrets_online'], label=r'$\mathtt{ExAUL}(\lambda=T^{1/8})$', color='pink')

            plt.xlabel(r'$t$', fontsize=15)
            plt.ylabel('Regret', fontsize=15)
            plt.ylim(0, 0.5)
            plt.axhline(y=training_args.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{training_args.alpha}')
            # Shift positions
            for pos in shift_positions:
                plt.axvline(x=pos, color='red', linestyle=':', linewidth=1, alpha=0.5)
                
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()

            plt.savefig(os.path.join(training_args.output_dir, 'online_regret.png'))
            pdf.savefig(bbox_inches='tight')
 
    else:
        raise NotImplementedError
    

    
if __name__ == "__main__":
    t_start = time.time()
    main()
    print(f'[total running time] {time.time() - t_start:.2f} sec')

# import os, sys
# from enum import Enum
# from dataclasses import asdict, dataclass, field, fields
# import json
# from typing import Optional
# from transformers import (
#     MODEL_FOR_CAUSAL_LM_MAPPING,
#     TrainingArguments,
# )

# import torch as tc


# MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# class Arguments:
#     def __str__(self):
#         self_as_dict = asdict(self)

#         # Remove deprecated arguments. That code should be removed once
#         # those deprecated arguments are removed from TrainingArguments. (TODO: v5)
#         # del self_as_dict["per_gpu_train_batch_size"]
#         # del self_as_dict["per_gpu_eval_batch_size"]

#         self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

#         attrs_as_str = [f"    {k}={v},\n" for k, v in sorted(self_as_dict.items())]
#         return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

#     __repr__ = __str__


# @dataclass
# class ModelArguments(Arguments):
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
#     """

#     model_name_or_path: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
#             )
#         },
#     )
#     model_type: Optional[str] = field(
#         default=None,
#         metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
#     )
#     config_overrides: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "Override some existing default config settings when a model is trained from scratch. Example: "
#                 "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
#             )
#         },
#     )
#     config_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
#     )
#     tokenizer_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
#     )
#     cache_dir: Optional[str] = field(
#         default=None,
#         metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
#     )
#     use_fast_tokenizer: bool = field(
#         default=True,
#         metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
#     )
#     model_revision: str = field(
#         default="main",
#         metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
#     )
#     use_auth_token: bool = field(
#         default=False,
#         metadata={
#             "help": (
#                 "Will use the token generated when running `transformers-cli login` (necessary to use this script "
#                 "with private models)."
#             )
#         },
#     )
#     entail_model_name_or_path: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
#             )
#         },
#     )

    
#     def __post_init__(self):
#         if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
#             raise ValueError(
#                 "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
#             )

#     # def __str__(self):
#     #     self_as_dict = asdict(self)

#     #     # Remove deprecated arguments. That code should be removed once
#     #     # those deprecated arguments are removed from TrainingArguments. (TODO: v5)
#     #     # del self_as_dict["per_gpu_train_batch_size"]
#     #     # del self_as_dict["per_gpu_eval_batch_size"]

#     #     self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

#     #     attrs_as_str = [f"    {k}={v},\n" for k, v in sorted(self_as_dict.items())]
#     #     return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

#     # __repr__ = __str__

    
#     # def to_dict(self):
#     #     """
#     #     Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
#     #     the token values by removing their value.
#     #     """
#     #     # filter out fields that are defined as field(init=False)
#     #     d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

#     #     for k, v in d.items():
#     #         if isinstance(v, Enum):
#     #             d[k] = v.value
#     #         if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
#     #             d[k] = [x.value for x in v]
#     #         if k.endswith("_token"):
#     #             d[k] = f"<{k.upper()}>"
#     #     return d

    
#     # def to_json_string(self):
#     #     """
#     #     Serializes this instance to a JSON string.
#     #     """
#     #     return json.dumps(self.to_dict(), indent=2)


# @dataclass
# class DataTrainingArguments(Arguments):
#     """
#     Arguments pertaining to what data we are going to input our model for training and eval.
#     """

#     dataset_name: Optional[str] = field(
#         default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
#     )
#     dataset_config_name: Optional[str] = field(
#         default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
#     )
#     train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
#     validation_file: Optional[str] = field(
#         default=None,
#         metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
#     )
#     max_train_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "For debugging purposes or quicker training, truncate the number of training examples to this "
#                 "value if set."
#             )
#         },
#     )
#     max_eval_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
#                 "value if set."
#             )
#         },
#     )

#     block_size: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "Optional input sequence length after tokenization. "
#                 "The training dataset will be truncated in block of this size for training. "
#                 "Default to the model max input length for single sentence inputs (take into account special tokens)."
#             )
#         },
#     )
#     overwrite_cache: bool = field(
#         default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
#     )
#     validation_split_percentage: Optional[int] = field(
#         default=5,
#         metadata={
#             "help": "The percentage of the train set used as validation set in case there's no validation split"
#         },
#     )
#     preprocessing_num_workers: Optional[int] = field(
#         default=None,
#         metadata={"help": "The number of processes to use for the preprocessing."},
#     )
#     keep_linebreaks: bool = field(
#         default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
#     )
#     log_args: bool = field(
#         default=False,
#         metadata={
#             "help": (
#                 "Whether to log the arguments to wandb or not. If True, the arguments will be logged to wandb."
#             )
#         },
#     )

#     def __post_init__(self):
#         if self.dataset_name is None and self.train_file is None and self.validation_file is None:
#             raise ValueError("Need either a dataset name or a training/validation file.")
#         else:
#             if self.train_file is not None:
#                 extension = self.train_file.split(".")[-1]
#                 assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
#             if self.validation_file is not None:
#                 extension = self.validation_file.split(".")[-1]
#                 assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

                
# @dataclass
# class UncertaintyTrainingArguments(Arguments, TrainingArguments):
#     distributed_state: Optional[str] = field(default=None)

#     dbg: Optional[bool] = field(default=False)

#     tag: Optional[str] = field(default='')

    
#     method: Optional[str] = field(default='Baseline', metadata={"help": "uncertainty learning"})
#     snapshot_root: Optional[str] = field(default='snapshots')
#     prompt_model_path: Optional[str] = field(default=None)
    
#     cache_root: Optional[str] = field(default='snapshots/cache')
#     cache_cal_fn: Optional[str] = field(default=None)
#     cache_eval_fn: Optional[str] = field(default=None)
#     cache_ent_fn: Optional[str] = field(default=None)
#     cache_ent_eval_fn: Optional[str] = field(default=None)
    
#     exp_name: Optional[str] = field(default='DBG')
    
#     rerun: Optional[bool] = field(default=False)
#     resume: Optional[bool] = field(default=False)
#     load_final: Optional[bool] = field(default=True)
#     verbose: Optional[bool] = field(default=False)
#     device: Optional[str] = field(default='cuda')
    
#     # n_cal: Optional[int] = field(default=None) #1_000_000
#     # eps: Optional[float] = field(default=0.1)
#     # eps_e: Optional[float] = field(default=0.1)
#     # delta: Optional[float] = field(default=1e-5)
#     # delta_p: Optional[float] = field(default=1e-5)
#     # n_train_max: Optional[int] = field(default=20_000) #100_000
#     # n_test_max: Optional[int] = field(default=1_000) #10_000
#     # topk: Optional[int] = field(default=50)
#     exp_method: Optional[str] = field(default='SSL')
#     # entail_model: Optional[str] = field(default=None)
#     alpha: Optional[float] = field(default=0.2)
#     h_size: Optional[int] = field(default=10000)
#     T: Optional[int] = field(default=10000)
#     confidence_score: Optional[str] = field(default=None)

#     # prompt learning
#     # prompt_learning: Optional[str] = field(default='prompt_tuning')
#     # n_virtual_tokens: Optional[int] = field(default=50)
    
#     # histogram binning
#     # n_bins: Optional[int] = field(default=20)

#     # optimizer: Optional[str] = field(default='Adam') ##TODO
#     # n_epochs: Optional[int] = field(default=1)
#     # lr: Optional[float] = field(default=1e-4)
#     # #momentum: Optional[float] = field(default=0.0)
#     # weight_decay: Optional[float] = field(default=0.0)
#     # lr_decay_epoch: Optional[int] = field(default=1)
#     # lr_decay_rate: Optional[float] = field(default=0.5)
#     # lr_gamma: Optional[float] = field(default=0.99)
#     # n_hidden_neurons: Optional[int] = field(default=4000)
#     # dropout_prob: Optional[float] = field(default=0.5)
#     # freeze_bias: Optional[bool] = field(default=False)
#     # use_logsigmoid: Optional[bool] = field(default=False)
#     # use_logspace: Optional[bool] = field(default=True)
    
    
#     # tau_step: Optional[float] = field(default=1e-16) 
#     # tau_end: Optional[float] = field(default=1.0) # 1.0: assume classification
#     # eps_tol: Optional[float] = field(default=1.25)
#     # tau_tol: Optional[float] = field(default=1e-16)
    
#     num_workers: Optional[int] = field(default=64)

#     # #TODO: create gen args
#     # gen: Optional[str] = field(default='learn_and_gen')
#     # gen_keywords: Optional[str] = field(default='')
#     gen_generation_type: Optional[str] = field(default='greedy')
#     gen_len: Optional[int] = field(default=50)
#     # gen_samples: Optional[int] = field(default=0)

#     # z_u: Optional[int] = field(default=30000)
#     # z_e: Optional[int] = field(default=10000)
#     # md_name: Optional[str] = field(default=None)
#     # fer: Optional[bool] = field(default=False)
#     # K: Optional[int] = field(default=5)
#     # pl: Optional[float] = field(default=0.9)

#     seed: Optional[int] = field(default=42)
#     model: Optional[str] = field(default=None)
#     def __post_init__(self):
#         if self.device == 'cpu':
#             self.device = tc.device('cpu')
#         elif self.device == 'cuda':
#             self.device = tc.device('cuda')
#         else:
#             raise NotImplementedError

import os
import sys
import json
from enum import Enum
from dataclasses import dataclass, field, asdict, fields
from typing import Optional, List
import torch as tc
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    TrainingArguments,
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

class Arguments:
    def __str__(self):
        self_as_dict = asdict(self)
        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}
        attrs_as_str = [f"    {k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"
    __repr__ = __str__

@dataclass
class ModelArguments(Arguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    entail_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

@dataclass
class DataTrainingArguments(Arguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    log_args: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to log the arguments to wandb or not. If True, the arguments will be logged to wandb."
            )
        },
    )
    # for shift
    nq_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Shift experiment: Disk path for Natural Questions dataset."}
    )
    triviaqa_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Shift experiment: Disk path for TriviaQA dataset."}
    )
    chunk_size: Optional[int] = field(
        default=10000,
        metadata={"help": "Shift chunk size."}
    )
    num_chunks: Optional[int] = field(
        default=10,
        metadata={"help": "# of chunks."}
    )
    contradict_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "If set, resample dataset so label==0 (contradict) fraction equals this value. "
                          "Both groups are randomly sampled to reduce selection bias."}
    )
    # for the imbalanced-stream experiments
    chunk_size_nq: Optional[int] = field(
        default=None,
        metadata={"help": "Imbalanced_Shift: NQ block length (e.g. 27000 against chunk_size_tri=3000)."}
    )
    chunk_size_tri: Optional[int] = field(
        default=None,
        metadata={"help": "Imbalanced_Shift: TriviaQA block length."}
    )
    p_major: Optional[float] = field(
        default=0.9,
        metadata={"help": "Mixture: per-round probability of drawing from the major distribution "
                          "(0.9 gives a steady-state 90:10 stream)."}
    )
    major_dist: Optional[str] = field(
        default='nq',
        metadata={"help": "Mixture: which distribution is the majority, 'nq' or 'tri'."}
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and not (self.nq_dataset_name and self.triviaqa_dataset_name)
        ):
            raise ValueError(
                "Need either a dataset_name, a train/validation file, or both nq_dataset_name and trivia_dataset_name for Shift experiments."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

                
@dataclass
class UncertaintyTrainingArguments(Arguments, TrainingArguments):
    distributed_state: Optional[str] = field(default=None)

    dbg: Optional[bool] = field(default=False)

    tag: Optional[str] = field(default='')

    
    method: Optional[str] = field(default='Baseline', metadata={"help": "uncertainty learning"})
    snapshot_root: Optional[str] = field(default='snapshots')
    prompt_model_path: Optional[str] = field(default=None)
    
    cache_root: Optional[str] = field(default='snapshots/cache')
    cache_cal_fn: Optional[str] = field(default=None)
    cache_eval_fn: Optional[str] = field(default=None)
    cache_ent_fn: Optional[str] = field(default=None)
    cache_ent_eval_fn: Optional[str] = field(default=None)
    
    exp_name: Optional[str] = field(default='DBG')
    
    rerun: Optional[bool] = field(default=False)
    resume: Optional[bool] = field(default=False)
    load_final: Optional[bool] = field(default=True)
    verbose: Optional[bool] = field(default=False)
    device: Optional[str] = field(default='cuda')
    
    exp_method: Optional[str] = field(default='SSL')
    # entail_model: Optional[str] = field(default=None)
    alpha: Optional[float] = field(default=0.2)
    h_size: Optional[int] = field(default=10000)
    T: Optional[int] = field(default=10000)
    confidence_score: Optional[str] = field(default=None)

    
    num_workers: Optional[int] = field(default=64)

    gen_generation_type: Optional[str] = field(default='greedy')
    gen_len: Optional[int] = field(default=50)

    # for data shift
    # bias_flag: Optional[str] = field(default=None)
    first_flag: Optional[str] = field(default='nq')
    
    seed: Optional[int] = field(default=42)
    model: Optional[str] = field(default=None)
    
    ablation_alphas: Optional[List[float]] = field(default=None)
    static_dialog_chunk: Optional[int] = field(default=None)
    
    unk_T: Optional[bool] = field(default=False)
    
    overLambda: Optional[bool] = field(default=False)
    lambda_T: Optional[int] = field(default=0)

    save_round_log: Optional[bool] = field(
        default=True,
        metadata={"help": "Dump the per-round streams to rounds.npz so new metrics can be computed "
                          "without re-running. Pass False for long-horizon runs (T in the millions), "
                          "where the archive gets large and slow to write."}
    )

    # feedback noise: corrupts only what the learner observes, never the labels the FDR is scored against
    feedback_flip_rate: Optional[float] = field(
        default=None,
        metadata={"help": "Symmetric feedback flip probability. Shorthand for setting both "
                          "flip_rate_pos and flip_rate_neg."}
    )
    flip_rate_pos: Optional[float] = field(
        default=None,
        metadata={"help": "p(observed=0 | true=1): the judge wrongly calls a correct answer wrong."}
    )
    flip_rate_neg: Optional[float] = field(
        default=None,
        metadata={"help": "p(observed=1 | true=0): the judge wrongly accepts a wrong answer. "
                          "Set above flip_rate_pos to model a lenient judge."}
    )

    # score robustness: degrades the confidence score itself, never the labels the FDR is scored
    # against. The claim under test is that FDR control does not rely on the score being informative.
    score_mode: Optional[str] = field(
        default='clean',
        metadata={"help": "How to perturb the confidence score. 'clean' uses --confidence_score as "
                          "is; 'shuffle' permutes it across rounds (marginal preserved, "
                          "score-label association destroyed); 'uniform' replaces it with U(0,1); "
                          "'mixed_bern' draws --confidence_score w.p. --score_mix_p and the other "
                          "score (fM1<->fM2) otherwise, so the optimal threshold moves per round; "
                          "'mixed_unif' replaces the score with U(0,1) on a --score_unif_p fraction "
                          "of rounds, a partial version of 'uniform'; 'switch' uses "
                          "--confidence_score up to --score_switch_at and the other score function "
                          "afterwards, an abrupt changepoint rather than a mixture."}
    )
    score_switch_at: Optional[float] = field(
        default=0.5,
        metadata={"help": "score_mode=switch: fraction of T at which the score function changes "
                          "over (0.5 switches at the midpoint of the stream)."}
    )
    score_mix_p: Optional[float] = field(
        default=0.5,
        metadata={"help": "score_mode=mixed_bern: per-round probability of using --confidence_score "
                          "rather than the other score function."}
    )
    score_unif_p: Optional[float] = field(
        default=0.1,
        metadata={"help": "score_mode=mixed_unif: per-round probability that the score is replaced "
                          "by a U(0,1) draw. 0 recovers the clean stream, 1 recovers score_mode=uniform."}
    )
        
    def __post_init__(self):
        if self.device == 'cpu':
            self.device = tc.device('cpu')
        elif self.device == 'cuda':
            self.device = tc.device('cuda')
        else:
            raise NotImplementedError

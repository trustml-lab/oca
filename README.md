

`./generation` contains scripts for generating answers or samples with models (API or local models).

`./models` contains `scg.py` which has classes for our basic algorithms.


## Environment Setup

```
# dependencies
uv sync
```

## Run a task

Run the scripts from the repository root:
```
uv run bash scripts/run_all.sh
```
or
```
source .venv/bin/activate
bash scripts/run_all.sh
```

Every script trains the three learners (EW-CA, Exp3-CA and ExAUL) on the same stream and writes results and plots to `snapshots/<method>/<run name>/`.
The models (`gpt-5.4`, `llama3.1-8B`), datasets, alpha and seeds are set at the top of each script.

### i.i.d. stream

| Script | Setting |
|---|---|
| `scripts/run_all.sh` | Main i.i.d. experiment: NQ or TriviaQA shuffled into one stream, T = 30,000, fM2 score. |
| `scripts/run_all_plot.sh` | Violin plot of the final FDR over 100 shuffles (NQ, fM1 score). |
| `scripts/run_all_std.sh`, `scripts/run_all_std_plot.sh` | The `run_all.sh` run and its violin plot on NQ with the fM1 score and seed 42. |
| `scripts/run_all_long.sh` | `run_all.sh` with T = 3,000,000. |
| `scripts/run_poa.sh` | Sweep of alpha from 0.05 to 1.0 (30 shuffles per alpha). |
| `scripts/run_lambda.sh` | Sweep of the ExAUL lambda over T, T^(1/2), T^(1/4) and T^(1/8). |

### Distribution shift (NQ and TriviaQA)

`FIRST_FLAG` in each script sets which dataset comes first (`nq` or `tri`).

| Script | Setting |
|---|---|
| `scripts/run_shift_single.sh` | One block of each dataset, 15,000 rounds each (T = 30,000). |
| `scripts/run_shift_alt.sh` | The datasets alternate in 5 blocks each, 3,000 rounds per block (T = 30,000). |
| `scripts/run_shift_linear.sh` | Gradual shift: the chance of drawing from the first dataset falls linearly from 1 to 0 over T = 30,000. |
| `scripts/run_shift_single_long.sh` | `run_shift_single.sh` with 1,500,000 rounds per block. |
| `scripts/run_shift_single_plot.sh`, `scripts/run_shift_alt_plot.sh`, `scripts/run_shift_linear_plot.sh` | Violin plot of the final FDR over 100 seeds of the matching setting. |
| `scripts/run_shift_poa.sh` | Sweep of alpha under the `run_shift_single.sh` shift (30 seeds per alpha). |
| `scripts/run_imbalanced_shift.sh` | One block of each dataset with different lengths (T = 60,000). |
| `scripts/run_mixture.sh` | No shift: every round comes from NQ with a fixed probability (T = 60,000). |
| `scripts/run_shift_medqa.sh`, `scripts/run_shift_medqa_plot.sh` | Shift between NQ or TriviaQA and MedQA-USMLE (GPT-5.4 only). |

### Robustness

| Script | Setting |
|---|---|
| `scripts/run_noise.sh` | Feedback labels flipped at random (symmetric, false-negative or false-positive); FDR is measured against the true labels. |
| `scripts/run_score_robust.sh` | The confidence score is shuffled, replaced by noise or mixed with another score. |
| `scripts/run_score_switch.sh` | The confidence score switches from fM2 to fM1 halfway through the stream. |

## Dialog experiments

These call the OpenAI API, so set `OPENAI_API_KEY` first (you are prompted for it if it is unset).
The other experiments never call the API and do not need a key.

| Script | Setting | Data |
|---|---|---|
| `scripts/run_dialog.sh` | A simulated user asks questions about a context; the chatbot answers and an LLM labeler judges each answer online. | `data/dialog/squad/contexts` |
| `scripts/run_dialog_static.sh` | The same dialog, run on fixed 500-row chunks of the contexts. | `data/dialog/squad/contexts` |
| `scripts/run_dialog_adversary.sh` | An LLM adversary sees the learner's recent decisions and picks the next question to make it fail. | `data/nq/gpt-3.5` + `data/triviaqa/gpt-3.5` |

An interrupted `run_dialog.sh` run saves `checkpoint.pkl` in its output directory and resumes from it when rerun.


## Datasets

* `data/nli/nq_gpt3.5`: This is the dataset from the original *selective generation* paper.
    * It contains `.jsonl` and the huggingface generator file `nli.py` that makes the dataset available by using `datasets.load_dataset()`

* `data/nq/llama3.1-8B`: This continas `generations` and `samples`.
    * Check out the form below and create a new arrow dataset which contains:
        1. `question`
        2. `answer`
        3. `generated_answer` (from `generated_sequence`)
        4. `transformed_answer` (from `transformed_sequence`)
        5. `label`
        6. `logprobs`, or `fM1` if you pre-computed
        7. `samples`
        8. `samples_scores`: list[float]


```
# generations

"generated_sequence": {
    "dtype": "string",
    "_type": "Value"
},
"question": {
    "dtype": "string",
    "_type": "Value"
},
"answer": {
    "dtype": "string",
    "_type": "Value"
},
"prompt": {
    "dtype": "string",
    "_type": "Value"
},
"transformed_sequence": {
    "dtype": "string",
    "_type": "Value"
},
"id": {
    "dtype": "int64",
    "_type": "Value"
},
"label": {
    "dtype": "null", # YJ, you fill this column.
    "_type": "Value"
},
"logprobs": {
    "feature": {
    "dtype": "float64",
    "_type": "Value"
    },
    "_type": "Sequence"
}
```


```
# samples
"generated_sequence": {
    "dtype": "null",
    "_type": "Value"
},
"transformed": {
    "dtype": "string",
    "_type": "Value"
},
"question": {
    "dtype": "string",
    "_type": "Value"
},
"answer": {
    "dtype": "string",
    "_type": "Value"
},
"prompt": {
    "dtype": "string",
    "_type": "Value"
},
"samples": {
    "feature": {
    "dtype": "string",
    "_type": "Value"
    },
    "_type": "Sequence"
}
# YJ, you should add a new column 'samples_scores'.
```

#!/bin/bash
set -e
# Rebuttal W1/Q1: FDR under noisy online feedback. Single-dataset setup of run_all.sh, but the
# feedback the learner observes is corrupted; the reported FDR is always measured against the
# true labels (models/scg.py: update(..., true_label=)).
#
# Rebuttal setting: T = 30,000 on NQ, alpha = 0.20, GPT-5.4 and Llama-3.1-8B,
# symmetric / false-negative / false-positive noise at 5%, 10%, 20%.
#
# Flag <-> rebuttal naming:
#   symmetric       --feedback_flip_rate r               both directions flipped w.p. r
#   false-negative  --flip_rate_pos r --flip_rate_neg 0  correct feedback flipped to incorrect,
#                                                        p(observed=0 | true=1). Strict judge:
#                                                        FDR goes down, inefficiency goes up.
#   false-positive  --flip_rate_pos 0 --flip_rate_neg r  incorrect feedback flipped to correct,
#                                                        p(observed=1 | true=0). Lenient judge:
#                                                        the direction that stresses FDR control.
# Results: snapshots/Noise/<run>/final_results.json, aggregated over seeds into
#          snapshots/Noise/noise_summary.json

TAG=NIPS25

HYPOTHESIS=1000
TS=(30000)

# the rebuttal reports NQ; TriviaQA (alpha=0.05) shows the same trend, add it to re-check
DATASETNAMELIST=(nq)
# DATASETNAMELIST=(nq triviaqa)
MDLNAMELIST=(gpt-5.4 llama3.1-8B)
METHODLIST=(All)
SCORELIST=(fM2)
UNK_T=False

# Every seed listed is reported (mean +- std below); do not pick the best seed.
SEEDLIST=(100)

# "<label>|<flags>". The label goes into the run directory name, so conditions stay separable.
NOISE_CONFIGS=(
  "clean|"
  "sym-0.05|--feedback_flip_rate 0.05"
  "sym-0.10|--feedback_flip_rate 0.10"
  "sym-0.20|--feedback_flip_rate 0.20"
  "fn-0.05|--flip_rate_pos 0.05 --flip_rate_neg 0.0"
  "fn-0.10|--flip_rate_pos 0.10 --flip_rate_neg 0.0"
  "fn-0.20|--flip_rate_pos 0.20 --flip_rate_neg 0.0"
  "fp-0.05|--flip_rate_pos 0.0 --flip_rate_neg 0.05"
  "fp-0.10|--flip_rate_pos 0.0 --flip_rate_neg 0.10"
  "fp-0.20|--flip_rate_pos 0.0 --flip_rate_neg 0.20"
)

for DATASETNAME in ${DATASETNAMELIST[*]};
do
	if [ "$DATASETNAME" = "triviaqa" ]; then
        ALPHAS=(0.05)
    elif [ "$DATASETNAME" = "nq" ]; then
		ALPHAS=(0.20)
	else
		ALPHAS=(0.25 0.2)
    fi
	for MDLNAME in ${MDLNAMELIST[*]};
	do
		for METHOD in ${METHODLIST[*]};
		do
			for SCORE in ${SCORELIST[*]};
			do
				for T in ${TS[*]};
				do
					for ALPHA in ${ALPHAS[*]};
					do
						for CONFIG in "${NOISE_CONFIGS[@]}";
						do
							NOISE_LABEL="${CONFIG%%|*}"
							NOISE_FLAGS="${CONFIG#*|}"
							for SEED in ${SEEDLIST[*]};
							do
					RUN_NAME=${DATASETNAME}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-alpha-${ALPHA}-noise-${NOISE_LABEL}-seed-${SEED}
					OUTPUT_DIR=snapshots/Noise/${RUN_NAME}
					echo "$TAG, $DATASETNAME, $MDLNAME, $METHOD, $SCORE, $HYPOTHESIS, $T, $ALPHA, noise-${NOISE_LABEL}, seed-${SEED}"

					# NOISE_FLAGS is deliberately unquoted so it expands into separate arguments
					.venv/bin/python main.py \
						--tag ${TAG} \
						--dataset_name data/${DATASETNAME}/${MDLNAME}/final \
						--model $MDLNAME \
						--exp_name "${RUN_NAME}" \
						--output_dir "${OUTPUT_DIR}" \
						--method $METHOD \
						--alpha ${ALPHA} \
						--h_size ${HYPOTHESIS} \
						--T ${T} \
						--confidence_score ${SCORE} \
						--seed ${SEED} \
						--unk_T ${UNK_T} \
						${NOISE_FLAGS}
							done
						done
					done
				done
			done
		done
	done
done

echo "All runs completed. Aggregating across seeds..."

# Mean +/- std over every seed, per noise condition.
.venv/bin/python - <<'PY'
import glob, json, os, re
from collections import defaultdict
import statistics as st

root = 'snapshots/Noise'
rows = defaultdict(list)
for path in sorted(glob.glob(f'{root}/*/final_results.json')):
    name = os.path.basename(os.path.dirname(path))
    m = re.match(r'(.+)-noise-(.+)-seed-(\d+)$', name)
    if not m:
        continue
    setting, noise, seed = m.groups()
    with open(path) as f:
        rows[(setting, noise)].append(json.load(f))

summary = {}
for (setting, noise), runs in sorted(rows.items()):
    entry = {'n_seeds': len(runs), 'alpha': runs[0]['alpha'], 'T': runs[0]['T']}
    for algo in ('EW-SG', 'Exp3-SG', 'ExAUL'):
        for metric in ('fdr', 'inefficiency'):
            vals = [r[algo][metric] for r in runs]
            entry[f'{algo}.{metric}.mean'] = st.mean(vals)
            entry[f'{algo}.{metric}.std'] = st.stdev(vals) if len(vals) > 1 else 0.0
    summary.setdefault(setting, {})[noise] = entry

os.makedirs(root, exist_ok=True)
with open(f'{root}/noise_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

for setting, per_noise in summary.items():
    print(f'\n=== {setting} (alpha={next(iter(per_noise.values()))["alpha"]}) ===')
    print(f'{"noise":20s} {"ExAUL FDR":>18s} {"ExAUL ineff":>18s} {"seeds":>6s}')
    for noise, e in per_noise.items():
        print(f'{noise:20s} {e["ExAUL.fdr.mean"]:9.4f} +-{e["ExAUL.fdr.std"]:6.4f} '
              f'{e["ExAUL.inefficiency.mean"]:9.4f} +-{e["ExAUL.inefficiency.std"]:6.4f} '
              f'{e["n_seeds"]:6d}')
print(f'\nWritten: {root}/noise_summary.json')
PY

echo "Done."

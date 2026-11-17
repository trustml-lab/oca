#!/bin/bash
set -e
# Is FDR control robust when the confidence score itself is uninformative or non-stationary?
#
# Unlike run_noise.sh, which corrupts the feedback the learner observes, this corrupts the learner's
# *input*. The guarantee is over the threshold class, not over the score being informative, so the
# expected outcome is: FDR stays at or below alpha in every condition while inefficiency rises
# relative to the clean fM2 run. That contrast -- same FDR, worse efficiency -- is the whole result.
#
# Conditions (--score_mode):
#   clean       fM2 as is. The reference point every other row is compared against.
#   shuffle     fM2 permuted across rounds. Marginal distribution preserved, score-label
#               association destroyed. The honest 'random score' control.
#   uniform     score ~ U(0,1), independent of the label. Also changes the score's scale.
#   mixed_bern  fM2 w.p. --score_mix_p, fM1 otherwise, drawn per round. fM1 and fM2 live on
#               different scales, so the optimal threshold jumps from round to round and no single
#               fixed threshold covers the stream. p sweeps 0.5 -> 0.9, i.e. from an even mix to
#               a nearly clean stream, so the degradation should be monotone in p.
#
# T matters here: the FDR bound is asymptotic (lr ~ sqrt(log N / T), lambda = sqrt(T)), and at
# T=3000 even the clean run overshoots alpha. Keep T at 30000 as in run_noise.sh.
#

TAG=NIPS25

HYPOTHESIS=1000
TS=(30000)

DATASETNAMELIST=(nq triviaqa)
MDLNAMELIST=(gpt-5.4 llama3.1-8B)
METHODLIST=(All)
SCORELIST=(fM2)
UNK_T=False

# Every seed is reported. Do NOT select the best seed the way run_all.sh does.
# SEEDLIST=(42 100 123 777 2024)
SEEDLIST=(42)

# "<label>|<flags>". The label goes into the run directory name, so conditions stay separable.
# SCORE_CONFIGS=(
#   "clean|--score_mode clean"
#   # uninformative score: FDR should hold, inefficiency should climb
#   "shuffle|--score_mode shuffle"
#   "uniform|--score_mode uniform"
#   # non-stationary score: fM2 with probability p, fM1 otherwise
#   "mixbern-0.5|--score_mode mixed_bern --score_mix_p 0.5"
#   "mixbern-0.6|--score_mode mixed_bern --score_mix_p 0.6"
#   "mixbern-0.7|--score_mode mixed_bern --score_mix_p 0.7"
#   "mixbern-0.8|--score_mode mixed_bern --score_mix_p 0.8"
#   "mixbern-0.9|--score_mode mixed_bern --score_mix_p 0.9"
# )

# fM2, but on an eps fraction of the rounds the score is replaced by a U(0,1) draw carrying no
# information. The partial version of score_mode=uniform: eps=0 is the clean run and eps=1 is fully
# uninformative, so the inefficiency increase should be monotone in eps while FDR tracks the clean
# row. 'clean' is kept in the list because every delta is measured against it.
SCORE_CONFIGS=(
#   "clean|--score_mode clean"
#   "mixunif-0.05|--score_mode mixed_unif --score_unif_p 0.05"
  "mixunif-0.10|--score_mode mixed_unif --score_unif_p 0.10"
  "mixunif-0.20|--score_mode mixed_unif --score_unif_p 0.20"
  "mixunif-0.30|--score_mode mixed_unif --score_unif_p 0.30"
  "mixunif-0.40|--score_mode mixed_unif --score_unif_p 0.40"
  "mixunif-0.50|--score_mode mixed_unif --score_unif_p 0.50"
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
						for CONFIG in "${SCORE_CONFIGS[@]}";
						do
							SCORE_LABEL="${CONFIG%%|*}"
							SCORE_FLAGS="${CONFIG#*|}"
							for SEED in ${SEEDLIST[*]};
							do
					RUN_NAME=${DATASETNAME}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-alpha-${ALPHA}-smode-${SCORE_LABEL}-seed-${SEED}
					OUTPUT_DIR=snapshots/ScoreRobust/${RUN_NAME}
					echo "$TAG, $DATASETNAME, $MDLNAME, $METHOD, $SCORE, $HYPOTHESIS, $T, $ALPHA, smode-${SCORE_LABEL}, seed-${SEED}"

					# SCORE_FLAGS is deliberately unquoted so it expands into separate arguments
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
						--save_round_log False \
						${SCORE_FLAGS}
							done
						done
					done
				done
			done
		done
	done
done

echo "All runs completed. Aggregating across seeds..."

# Mean +/- std over every seed, per score condition, with the inefficiency delta against the clean
# row of the same setting -- that delta is the number the claim rests on.
.venv/bin/python - <<'PY'
import glob, json, os, re
from collections import defaultdict
import statistics as st

rows = defaultdict(list)
for path in sorted(glob.glob('snapshots/ScoreRobust/*/final_results.json')):
    name = os.path.basename(os.path.dirname(path))
    m = re.match(r'(.+)-smode-(.+)-seed-(\d+)$', name)
    if not m:
        continue
    setting, smode, seed = m.groups()
    with open(path) as f:
        rows[(setting, smode)].append(json.load(f))

ALGOS = ('EW-SG', 'Exp3-SG', 'ExAUL')

summary = {}
for (setting, smode), runs in sorted(rows.items()):
    entry = {'n_seeds': len(runs), 'alpha': runs[0]['alpha'], 'T': runs[0]['T']}
    for algo in ALGOS:
        for metric in ('fdr', 'inefficiency'):
            vals = [r[algo][metric] for r in runs]
            entry[f'{algo}.{metric}.mean'] = st.mean(vals)
            entry[f'{algo}.{metric}.std'] = st.stdev(vals) if len(vals) > 1 else 0.0
    summary.setdefault(setting, {})[smode] = entry

# inefficiency increase over the clean run of the same setting
for setting, per_mode in summary.items():
    clean = per_mode.get('clean')
    for smode, e in per_mode.items():
        for algo in ALGOS:
            e[f'{algo}.inefficiency.delta_clean'] = (
                e[f'{algo}.inefficiency.mean'] - clean[f'{algo}.inefficiency.mean']
                if clean else None
            )

os.makedirs('snapshots/ScoreRobust', exist_ok=True)
with open('snapshots/ScoreRobust/score_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

for setting, per_mode in summary.items():
    alpha = next(iter(per_mode.values()))['alpha']
    print(f'\n=== {setting} (alpha={alpha}) ===')
    print(f'{"score_mode":14s} {"ExAUL FDR":>18s} {"ExAUL ineff":>18s} {"d.ineff":>9s} {"seeds":>6s}')
    # clean first, then the rest in name order, so the reference row is on top
    for smode in sorted(per_mode, key=lambda s: (s != 'clean', s)):
        e = per_mode[smode]
        d = e['ExAUL.inefficiency.delta_clean']
        print(f'{smode:14s} {e["ExAUL.fdr.mean"]:9.4f} +-{e["ExAUL.fdr.std"]:6.4f} '
              f'{e["ExAUL.inefficiency.mean"]:9.4f} +-{e["ExAUL.inefficiency.std"]:6.4f} '
              f'{(d if d is not None else float("nan")):9.4f} {e["n_seeds"]:6d}')
        if e['ExAUL.fdr.mean'] > alpha:
            print(f'{"":14s}   ^ FDR above alpha -- robustness failure, not a cost in efficiency')
print('\nWritten: snapshots/ScoreRobust/score_summary.json')
PY

echo "Done."

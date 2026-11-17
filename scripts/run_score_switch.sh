#!/bin/bash
set -e
# Does FDR control survive an abrupt change in the confidence score itself?
#
# run_score_robust.sh corrupts the score i.i.d. across rounds; here the score *function* changes
# once, at the midpoint of the stream: fM2 for t < T/2, fM1 for t >= T/2. That is a distribution
# shift on the learner's input rather than noise on it, and it is the setting the reviewer asks
# about -- the optimal threshold before the switch is not the optimal threshold after it.
#
# Conditions:
#   clean         fM2 for the whole stream. The comparison the claim is about.
#   switch-0.5    fM2 -> fM1 at t = T/2.
#   fM1-clean     fM1 for the whole stream. NOT a condition -- a reference row. fM1 is a *worse*
#                 score than fM2 (on triviaqa/gpt-5.4 the best fixed threshold attains
#                 inefficiency 0.65 under fM1 vs 0.16 under fM2), so the post-switch segment
#                 necessarily commits less often no matter how well the learner adapts. Reading
#                 the post-switch segment against the clean fM2 run would credit that gap to
#                 adaptation failure; it must be read against this row instead.
#
# What to look at, in order:
#   1. segment_results.json, post-switch FDR -- if control holds it stays at or below alpha.
#   2. post-switch inefficiency vs the fM1-clean row -- the residual is the cost of adaptation.
#   3. rounds.npz, exaul__tau -- how many rounds after T/2 before the threshold settles.
#
# T is 30000 as in run_score_robust.sh: the FDR bound is asymptotic and at T=3000 even a clean run
# overshoots alpha, which would make a post-switch overshoot unattributable.
#

TAG=NIPS25

HYPOTHESIS=1000
TS=(150000 180000)

DATASETNAMELIST=(nq triviaqa)
MDLNAMELIST=(gpt-5.4 llama3.1-8B)
METHODLIST=(All)
UNK_T=False

# Every seed is reported. Do NOT select the best seed the way run_all.sh does.
SEEDLIST=(42)

# "<label>|<confidence_score>|<flags>"
# CONFIGS=(
#   "clean|fM2|--score_mode clean"
#   "switch-0.5|fM2|--score_mode switch --score_switch_at 0.5"
#   "fM1-clean|fM1|--score_mode clean"
# )
CONFIGS=(
  "switch-0.5|fM2|--score_mode switch --score_switch_at 0.5"
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
			for T in ${TS[*]};
			do
				for ALPHA in ${ALPHAS[*]};
				do
					for CONFIG in "${CONFIGS[@]}";
					do
						LABEL="${CONFIG%%|*}"
						REST="${CONFIG#*|}"
						SCORE="${REST%%|*}"
						FLAGS="${REST#*|}"
						for SEED in ${SEEDLIST[*]};
						do
				RUN_NAME=${DATASETNAME}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-alpha-${ALPHA}-smode-${LABEL}-seed-${SEED}
				OUTPUT_DIR=snapshots/ScoreSwitch/${RUN_NAME}
				echo "$TAG, $DATASETNAME, $MDLNAME, $METHOD, $SCORE, $HYPOTHESIS, $T, $ALPHA, smode-${LABEL}, seed-${SEED}"

				# FLAGS is deliberately unquoted so it expands into separate arguments
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
					--save_round_log True \
					${FLAGS}
						done
					done
				done
			done
		done
	done
done

echo "All runs completed. Aggregating across seeds..."

.venv/bin/python - <<'PY'
import glob, json, os, re
from collections import defaultdict
import statistics as st
import numpy as np

rows = defaultdict(list)
for path in sorted(glob.glob('snapshots/ScoreSwitch/*/final_results.json')):
    d = os.path.dirname(path)
    name = os.path.basename(d)
    m = re.match(r'(.+?)-(?:fM1|fM2)-(T-.+)-smode-(.+)-seed-(\d+)$', name)
    if not m:
        continue
    prefix, tail, smode, seed = m.groups()
    with open(path) as f:
        res = json.load(f)
    seg_path = os.path.join(d, 'segment_results.json')
    res['_seg'] = json.load(open(seg_path)) if os.path.exists(seg_path) else None
    # rounds after the switch before |tau - tau_final| stays within one grid step: the recovery lag
    res['_recovery'] = None
    npz_path = os.path.join(d, 'rounds.npz')
    if res['_seg'] and os.path.exists(npz_path):
        with np.load(npz_path) as z:
            if 'exaul__tau' in z:
                tau = z['exaul__tau']
                k = int(round(res['_seg']['switch_at'] * len(tau)))
                post = tau[k:]
                tol = 1.0 / 1000  # h_size grid step
                settled = np.abs(post - post[-1]) <= tol
                # first index from which it never leaves the band again
                bad = np.flatnonzero(~settled)
                res['_recovery'] = int(bad[-1] + 1) if bad.size else 0
    rows[(f'{prefix}-{tail}', smode)].append(res)

ALGOS = ('EW-SG', 'Exp3-SG', 'ExAUL')
ms = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)

summary = {}
for (setting, smode), runs in sorted(rows.items()):
    e = {'n_seeds': len(runs), 'alpha': runs[0]['alpha'], 'T': runs[0]['T']}
    for algo in ALGOS:
        for metric in ('fdr', 'inefficiency'):
            e[f'{algo}.{metric}.mean'], e[f'{algo}.{metric}.std'] = ms([r[algo][metric] for r in runs])
        segs = [r['_seg'] for r in runs if r['_seg']]
        for part in ('pre', 'post'):
            if segs:
                for metric in ('fdr', 'inefficiency'):
                    e[f'{algo}.{part}.{metric}.mean'], e[f'{algo}.{part}.{metric}.std'] = \
                        ms([s[algo][part][metric] for s in segs])
    rec = [r['_recovery'] for r in runs if r['_recovery'] is not None]
    if rec:
        e['ExAUL.recovery_rounds.mean'], e['ExAUL.recovery_rounds.std'] = ms(rec)
    summary.setdefault(setting, {})[smode] = e

os.makedirs('snapshots/ScoreSwitch', exist_ok=True)
with open('snapshots/ScoreSwitch/switch_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

for setting, per_mode in summary.items():
    alpha = next(iter(per_mode.values()))['alpha']
    print(f'\n=== {setting} (alpha={alpha}) ===')
    print(f'{"condition":12s} {"ExAUL FDR":>18s} {"ExAUL ineff":>18s} {"post FDR":>9s} '
          f'{"post ineff":>11s} {"recov":>8s} {"seeds":>6s}')
    for smode in sorted(per_mode, key=lambda s: {'clean': 0, 'switch-0.5': 1}.get(s, 2)):
        e = per_mode[smode]
        pf = e.get('ExAUL.post.fdr.mean')
        pi = e.get('ExAUL.post.inefficiency.mean')
        rc = e.get('ExAUL.recovery_rounds.mean')
        print(f'{smode:12s} {e["ExAUL.fdr.mean"]:9.4f} +-{e["ExAUL.fdr.std"]:6.4f} '
              f'{e["ExAUL.inefficiency.mean"]:9.4f} +-{e["ExAUL.inefficiency.std"]:6.4f} '
              f'{(pf if pf is not None else float("nan")):9.4f} '
              f'{(pi if pi is not None else float("nan")):11.4f} '
              f'{(rc if rc is not None else float("nan")):8.0f} {e["n_seeds"]:6d}')
        if e['ExAUL.fdr.mean'] > alpha:
            print(f'{"":12s}   ^ overall FDR above alpha')
        if pf is not None and pf > alpha:
            print(f'{"":12s}   ^ post-switch FDR above alpha -- the changepoint broke control')
    # the number the claim rests on: post-switch inefficiency against fM1 run on its own
    sw, ref = per_mode.get('switch-0.5'), per_mode.get('fM1-clean')
    if sw and ref and 'ExAUL.post.inefficiency.mean' in sw:
        gap = sw['ExAUL.post.inefficiency.mean'] - ref['ExAUL.inefficiency.mean']
        print(f'  adaptation cost (post-switch ineff - fM1-only ineff) = {gap:+.4f}')

print('\nWritten: snapshots/ScoreSwitch/switch_summary.json')
PY

echo "Done."

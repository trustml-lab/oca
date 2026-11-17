#!/bin/bash
set -e
# Rebuttal W3 (i): steady-state imbalance. Each round is drawn i.i.d. from NQ w.p. P_NQ and from
# TriviaQA otherwise, so the mixing ratio is constant over the horizon and there is no shift to
# adapt to. Table column (p_NQ, p_TriviaQA) = (P_NQ, 1 - P_NQ).
#
# Rebuttal setting: T = 60,000, alpha = 0.10, GPT-5.4 and Llama-3.1-8B, p_NQ in 0.1 ... 0.9.
#
# --major_dist nq with --p_major P_NQ gives p_NQ = P_NQ directly; main.py does not require
# p_major >= 0.5, so one sweep covers both the NQ-major and TriviaQA-major sides.
# Results: snapshots/Mixture/*/per_dist_results.json
#   ExAUL.overall.fdr                      -> W3 table (1)
#   ExAUL.per_dist.{NQ,TriviaQA}.fdr       -> per-distribution FDR (read n_commit alongside)

TAG=NIPS25
HYPOTHESIS=1000

T=60000
UNK_T=False
SEED=42

MDLNAMELIST=(gpt-5.4 llama3.1-8B)

P_NQ_LIST=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
MAJOR_DIST=nq

METHODLIST=(Mixture)
SCORELIST=(fM2)

ALPHAS=(0.1)

for MDLNAME in ${MDLNAMELIST[*]};
do
  NQ_DATASET="data/nq/${MDLNAME}/final"
  TRIVIA_DATASET="data/triviaqa/${MDLNAME}/final"
  for METHOD in ${METHODLIST[*]};
  do
    for P_NQ in ${P_NQ_LIST[@]};
    do
      for SCORE in ${SCORELIST[@]};
      do
        for ALPHA in ${ALPHAS[@]};
        do
        EXP_NAME=pnq-${P_NQ}-${MDLNAME}-${METHOD}-T-${T}-alpha-${ALPHA}
        echo "$TAG, mixture, $MDLNAME, $METHOD, $SCORE, $HYPOTHESIS, $ALPHA, p_nq=$P_NQ, T=$T"

          .venv/bin/python main.py \
            --tag ${TAG} \
            --nq_dataset_name ${NQ_DATASET} \
            --triviaqa_dataset_name ${TRIVIA_DATASET} \
            --model ${MDLNAME} \
            --exp_name ${EXP_NAME} \
            --output_dir snapshots/${METHOD}/${EXP_NAME} \
            --method ${METHOD} \
            --alpha ${ALPHA} \
            --T ${T} \
            --p_major ${P_NQ} \
            --major_dist ${MAJOR_DIST} \
            --h_size ${HYPOTHESIS} \
            --confidence_score ${SCORE} \
            --seed ${SEED} \
            --unk_T ${UNK_T}
        done
      done
    done
  done
done

wait
echo "All tasks completed."
echo "Overall / per-distribution FDR: snapshots/Mixture/*/per_dist_results.json"

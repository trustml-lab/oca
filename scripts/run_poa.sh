#!/bin/bash
set -e

TAG=NIPS25
HYPOTHESIS=1000
TS=(30000)

DATASETNAMELIST=(triviaqa nq)
MDLNAMELIST=(gpt-5.4 llama3.1-8B)

METHODLIST=(Plot-over-alpha)

SCORELIST=(fM2)

UNK_T=False
# UNK_T=True

ALPHAS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.0)
# ALPHAS=(0.05 0.10)

for DATASETNAME in ${DATASETNAMELIST[*]}; do
  for MDLNAME in "${MDLNAMELIST[@]}"; do
    for METHOD in "${METHODLIST[@]}"; do
      for SCORE in "${SCORELIST[@]}"; do
        for T in ${TS[*]}; do
          echo $TAG, $DATASETNAME, $MDLNAME, $METHOD, $SCORE, $HYPOTHESIS, $T, alpha-sweep

          python3 main.py \
            --tag ${TAG} \
            --dataset_name data/${DATASETNAME}/${MDLNAME}/final \
            --model ${MDLNAME} \
            --exp_name ${DATASETNAME}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-Hs-${HYPOTHESIS}-unkT-${UNK_T} \
            --output_dir snapshots/${METHOD}/${DATASETNAME}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-Hs-${HYPOTHESIS}-unkT-${UNK_T} \
            --method ${METHOD} \
            --alpha ${ALPHAS[0]} \
            --ablation_alphas "${ALPHAS[@]}" \
            --h_size ${HYPOTHESIS} \
            --T ${T} \
            --confidence_score ${SCORE} \
            --unk_T ${UNK_T} &
        done
      done
    done
  done
done


wait
echo "All tasks completed."

#!/bin/bash
set -e

TAG=NIPS25
HYPOTHESIS=1000
CHUNK_SIZE=15000
NUM_CHUNKS=1

UNK_T=False   

FIRST_FLAG_LIST=(tri nq)
MDLNAMELIST=(gpt-5.4 llama3.1-8B)

METHODLIST=(Shift_Plot_over_alpha)

SCORELIST=(fM2)

ALPHAS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.0)

for MDLNAME in "${MDLNAMELIST[@]}"; do
  NQ_DATASET="data/nq/${MDLNAME}/final"
  TRIVIA_DATASET="data/triviaqa/${MDLNAME}/final"

  for METHOD in "${METHODLIST[@]}"; do
    for FIRST_FLAG in "${FIRST_FLAG_LIST[@]}"; do
      for SCORE in "${SCORELIST[@]}"; do
        echo "$TAG, shift, $MDLNAME, $METHOD, $SCORE, Hs=$HYPOTHESIS, alpha-sweep, chunk=$CHUNK_SIZE, first=$FIRST_FLAG"

        python3 main.py \
          --tag ${TAG} \
          --nq_dataset_name ${NQ_DATASET} \
          --triviaqa_dataset_name ${TRIVIA_DATASET} \
          --model ${MDLNAME} \
          --exp_name ${CHUNK_SIZE}-${MDLNAME}-${METHOD}-${SCORE}-Hs-${HYPOTHESIS}-${FIRST_FLAG}-unkT-${UNK_T} \
          --output_dir snapshots/${METHOD}/${CHUNK_SIZE}-${MDLNAME}-${METHOD}-${SCORE}-Hs-${HYPOTHESIS}-${FIRST_FLAG}-unkT-${UNK_T} \
          --method ${METHOD} \
          --alpha ${ALPHAS[0]} \
          --ablation_alphas "${ALPHAS[@]}" \
          --chunk_size ${CHUNK_SIZE} \
          --num_chunks ${NUM_CHUNKS} \
          --h_size ${HYPOTHESIS} \
          --first_flag ${FIRST_FLAG} \
          --confidence_score ${SCORE} \
          --unk_T ${UNK_T} &
      done
    done
  done
done

wait
echo "All tasks completed."

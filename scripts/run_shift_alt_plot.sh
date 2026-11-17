#!/bin/bash
set -e

TAG=NIPS25
HYPOTHESIS=1000

CHUNK_SIZE=3000
NUM_CHUNKS=5
UNK_T=False   

MDLNAMELIST=(gpt-5.4 llama3.1-8B)
FIRST_FLAG_LIST=(tri nq)

METHODLIST=(Shift_Plot)
# METHODLIST=(Shift)
SCORELIST=(fM2)

# ALPHAS=(0.1 0.15)
ALPHAS=(0.1)

for MDLNAME in ${MDLNAMELIST[*]};
do
  NQ_DATASET="data/nq/${MDLNAME}/final"
  TRIVIA_DATASET="data/triviaqa/${MDLNAME}/final"
  for METHOD in ${METHODLIST[*]};
  do 
    for FIRST_FLAG in ${FIRST_FLAG_LIST[*]};
    do
      for SCORE in ${SCORELIST[@]};
      do
        for ALPHA in ${ALPHAS[@]};
        do
        echo "$TAG, shift, $MDLNAME, $METHOD, $SCORE, $HYPOTHESIS, $ALPHA, $CHUNK_SIZE, $FIRST_FLAG"

          python3 main.py \
            --tag ${TAG} \
            --nq_dataset_name ${NQ_DATASET} \
            --triviaqa_dataset_name ${TRIVIA_DATASET} \
            --model ${MDLNAME} \
            --exp_name ${FIRST_FLAG}-${MDLNAME}-${METHOD}-C-${CHUNK_SIZE}-alpha-${ALPHA}-nc-${NUM_CHUNKS} \
            --output_dir snapshots/${METHOD}/${FIRST_FLAG}-${MDLNAME}-${METHOD}-C-${CHUNK_SIZE}-alpha-${ALPHA}-nc-${NUM_CHUNKS} \
            --method ${METHOD} \
            --alpha ${ALPHA} \
            --chunk_size ${CHUNK_SIZE} \
            --num_chunks ${NUM_CHUNKS} \
            --h_size ${HYPOTHESIS} \
            --first_flag ${FIRST_FLAG} \
            --confidence_score ${SCORE} \
            --unk_T ${UNK_T}
        done
      done
    done
  done
done

wait
echo "All tasks completed."
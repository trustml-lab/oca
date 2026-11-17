#!/bin/bash
set -e
# Rebuttal W5: domain shift between open-domain QA and MedQA-USMLE. Same setup as
# run_shift_single.sh (Shift method, one block of each, CHUNK_SIZE=15000 -> T = 30,000),
# but the second distribution is MedQA instead of TriviaQA.
#
# Rebuttal setting: T = 30,000; NQ <-> MedQA at alpha = 0.10; TriviaQA <-> MedQA at alpha = 0.05.
# Only GPT-5.4: data/medqa exists for gpt-5.4 only.
#
# The Shift method takes two dataset slots named after NQ/TriviaQA. Here the "nq" slot holds the
# open-domain dataset (NQ or TriviaQA) and the "tri" slot holds MedQA, so
#   FIRST_FLAG=nq  : open-domain -> MedQA
#   FIRST_FLAG=tri : MedQA -> open-domain
# Plot legends and per_dist_results.json keys therefore read "NQ" for the open-domain dataset and
# "TriviaQA" for MedQA; the run directory name carries the real pair.
#
# MedQA has 3,000 rows; a 15,000-row block is drawn by concatenating independent shuffles
# (main.py _sample_chunk), so each MedQA question appears ~5 times in its block.
# Results: snapshots/Shift/<dir>/per_dist_results.json (ExAUL.overall.{fdr,inefficiency})

TAG=NIPS25
HYPOTHESIS=1000

CHUNK_SIZE=15000
NUM_CHUNKS=1
UNK_T=False

MDLNAMELIST=(gpt-5.4)
FIRST_FLAG_LIST=(nq tri)

METHODLIST=(Shift)
SCORELIST=(fM2)

# "<open-domain dataset> <alpha>", paired with MedQA
SOURCE_CONFIGS=(
  "nq 0.1"
  "triviaqa 0.05"
)
TARGET=medqa

for MDLNAME in ${MDLNAMELIST[*]};
do
  TARGET_DATASET="data/${TARGET}/${MDLNAME}/final"
  for CONFIG in "${SOURCE_CONFIGS[@]}";
  do
    read -r SOURCE ALPHA <<< "${CONFIG}"
    SOURCE_DATASET="data/${SOURCE}/${MDLNAME}/final"
    for METHOD in ${METHODLIST[*]};
    do
      for FIRST_FLAG in ${FIRST_FLAG_LIST[*]};
      do
        if [ "${FIRST_FLAG}" = "nq" ]; then
          PAIR_NAME=${SOURCE}-to-${TARGET}
        else
          PAIR_NAME=${TARGET}-to-${SOURCE}
        fi
        for SCORE in ${SCORELIST[@]};
        do
        EXP_NAME=${PAIR_NAME}-${MDLNAME}-${METHOD}-C-${CHUNK_SIZE}-alpha-${ALPHA}-nc-${NUM_CHUNKS}
        echo "$TAG, medqa shift, $PAIR_NAME, $MDLNAME, $METHOD, $SCORE, $HYPOTHESIS, $ALPHA, $CHUNK_SIZE"

          .venv/bin/python main.py \
            --tag ${TAG} \
            --nq_dataset_name ${SOURCE_DATASET} \
            --triviaqa_dataset_name ${TARGET_DATASET} \
            --model ${MDLNAME} \
            --exp_name ${EXP_NAME} \
            --output_dir snapshots/${METHOD}/${EXP_NAME} \
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
echo "Overall FDR / inefficiency: snapshots/Shift/*-to-*/per_dist_results.json (ExAUL.overall)"

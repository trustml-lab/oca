#!/bin/bash
set -e
# Violin plot of the final FDR over 100 seeds for the MedQA-USMLE domain shift
# (run_shift_single_plot.sh counterpart of run_shift_medqa.sh).
#
# Shift_Plot rebuilds the shifted stream with seed+idx for idx in 0..99, trains the three
# learners on each, and draws a violin of the final FDR per learner. Same stream layout as
# run_shift_medqa.sh: one block of each distribution, CHUNK_SIZE=15000 -> T = 30,000.
#
# Rebuttal setting: NQ <-> MedQA at alpha = 0.10; TriviaQA <-> MedQA at alpha = 0.05.
# Only GPT-5.4: data/medqa exists for gpt-5.4 only.
#
# Slot convention (see run_shift_medqa.sh): the "nq" slot holds the open-domain dataset, the
# "tri" slot holds MedQA, so FIRST_FLAG=nq is open-domain -> MedQA and FIRST_FLAG=tri is
# MedQA -> open-domain.
# Results: snapshots/Shift_Plot/<pair>-*/violin.{png,pdf}, fdrs{1,2,3}.json
#          (fdrs1 = EW-CA, fdrs2 = Exp3-IX-CA, fdrs3 = ExAUL)

TAG=NIPS25
HYPOTHESIS=1000

CHUNK_SIZE=15000
NUM_CHUNKS=1
UNK_T=False

MDLNAMELIST=(gpt-5.4)
FIRST_FLAG_LIST=(nq tri)

METHODLIST=(Shift_Plot)
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
        echo "$TAG, medqa shift plot, $PAIR_NAME, $MDLNAME, $METHOD, $SCORE, $HYPOTHESIS, $ALPHA, $CHUNK_SIZE"

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
echo "Violin plots: snapshots/Shift_Plot/*-to-*/violin.png"

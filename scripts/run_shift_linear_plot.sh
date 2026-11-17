#!/bin/bash
set -e

TAG=NIPS25
HYPOTHESIS=1000
TS=(30000)

UNK_T=False   

MDLNAMELIST=(gpt-5.4 llama3.1-8B)
FIRST_FLAG_LIST=(tri nq)

# METHODLIST=(Shift_Linear)
METHODLIST=(Shift_Plot_Linear)

SCORELIST=(fM2)

ALPHAS=(0.1)

for MDLNAME in ${MDLNAMELIST[*]}; 
do
    NQ_DATASET="data/nq/${MDLNAME}/final"
    TRIVIA_DATASET="data/triviaqa/${MDLNAME}/final"
    for METHOD in ${METHODLIST[*]};
    do
        for T in ${TS[*]};
        do
            for FIRST_FLAG in ${FIRST_FLAG_LIST[*]};
            do
                for SCORE in ${SCORELIST[*]};
                do
                    for ALPHA in ${ALPHAS[*]};
                    do
                        echo "$TAG, shift, $MDLNAME, $METHOD, $SCORE, $HYPOTHESIS, $ALPHA, $FIRST_FLAG"

                        python3 main.py \
                            --tag ${TAG} \
                            --nq_dataset_name ${NQ_DATASET} \
                            --triviaqa_dataset_name ${TRIVIA_DATASET} \
                            --model ${MDLNAME} \
                            --exp_name ${FIRST_FLAG}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-alpha-${ALPHA} \
                            --output_dir snapshots/${METHOD}/${FIRST_FLAG}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-alpha-${ALPHA} \
                            --method ${METHOD} \
                            --alpha ${ALPHA} \
                            --h_size ${HYPOTHESIS} \
                            --T ${T} \
                            --first_flag ${FIRST_FLAG} \
                            --confidence_score ${SCORE} \
                            --unk_T ${UNK_T} &
                    done
                done
            done
        done
    done
done

wait
echo "All tasks completed."

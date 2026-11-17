#!/bin/bash
set -e

TAG=NIPS25

HYPOTHESIS=1000
TS=(30000)

DATASETNAMELIST=(triviaqa nq)
MDLNAMELIST=(gpt-5.4 llama3.1-8B)

METHODLIST=(Plot_over_lambda)

SCORELIST=(fM2)

UNK_T=False
overLambda=True

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
					echo $TAG, $DATASETNAME, $MDLNAME, $METHOD, $SCORE, $HYPOTHESIS, $T, $ALPHA

					python3 main.py \
						--tag ${TAG} \
						--dataset_name data/${DATASETNAME}/${MDLNAME}/final \
						--model $MDLNAME \
						--exp_name ${TAG}-${DATASETNAME}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-alpha-${ALPHA}-Hs-${HYPOTHESIS}-unkT-${UNK_T}--overLambda-${overLambda} \
						--output_dir snapshots/${METHOD}/${TAG}-${DATASETNAME}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-alpha-${ALPHA}-Hs-${HYPOTHESIS}-unkT-${UNK_T}--overLambda-${overLambda} \
						--method $METHOD \
						--alpha ${ALPHA} \
						--h_size ${HYPOTHESIS} \
						--T ${T} \
						--confidence_score ${SCORE} \
						--unk_T ${UNK_T} \
						--overLambda ${overLambda} &
						# --preprocessing_num_workers 64 \
						# --dataloader_num_workers 64 \
						# --per_device_train_batch_size 32 \
						# --per_device_eval_batch_size 32 \
						# --fp16 \
						# --n_cal $N
					done
				done
			done
		done
	done
done

wait
echo "All tasks completed."
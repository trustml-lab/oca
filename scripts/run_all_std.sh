#!/bin/bash
set -e

TAG=NIPS25

HYPOTHESIS=1000
# TS=(30000 50000 60000)
TS=(30000)

# DATASETNAMELIST=(triviaqa nq)
# DATASETNAMELIST=(triviaqa)
DATASETNAMELIST=(nq)
# DATASETNAMELIST=(medqa barexam)
MDLNAMELIST=(gpt-5.4 llama3.1-8B)
# MDLNAMELIST=(gpt-5.4)
METHODLIST=(All)
# METHODLIST=(Ablation)
# METHODLIST=(Plot)
SCORELIST=(fM1)
# SCORELIST=(fM1)
SEEDLIST=(42)
# SEEDLIST=($(shuf -i 1-1000000 -n 50))
UNK_T=False


for DATASETNAME in ${DATASETNAMELIST[*]};
do
	if [ "$DATASETNAME" = "triviaqa" ]; then
        ALPHAS=(0.05)
    elif [ "$DATASETNAME" = "nq" ]; then
        # ALPHAS=(0.05 0.10 0.15 0.20)
		ALPHAS=(0.20)
	else
		# ALPHAS=(0.25 0.2)
		ALPHAS=(0.03 0.05)
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
							for SEED in ${SEEDLIST[*]};
							do
					RUN_NAME=${DATASETNAME}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-alpha-${ALPHA}-seed-${SEED}
					OUTPUT_DIR=snapshots/${METHOD}/${RUN_NAME}
					echo $TAG, $DATASETNAME, $MDLNAME, $METHOD, $SCORE, $HYPOTHESIS, $T, $ALPHA, seed-${SEED}

					.venv/bin/python main.py \
						--tag ${TAG} \
						--dataset_name data/${DATASETNAME}/${MDLNAME}/final  \
						--model $MDLNAME \
						--exp_name "${RUN_NAME}" \
						--output_dir "${OUTPUT_DIR}" \
						--method $METHOD \
						--alpha ${ALPHA} \
						--h_size ${HYPOTHESIS} \
						--T ${T} \
						--confidence_score ${SCORE} \
						--seed ${SEED} \
						--unk_T ${UNK_T}
							done
						done
					done
				done
			done
	done
done

echo "All tasks completed."

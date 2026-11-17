#!/bin/bash
set -e

TAG=NIPS25

HYPOTHESIS=1000
# TS=(2000000 2500000 3000000)
TS=(3000000)

DATASETNAMELIST=(triviaqa nq)
MDLNAMELIST=(gpt-5.4 llama3.1-8B)
METHODLIST=(All)
SCORELIST=(fM2)
UNK_T=False


for DATASETNAME in ${DATASETNAMELIST[*]};
do
	if [ "$DATASETNAME" = "triviaqa" ]; then
        ALPHAS=(0.05)
    elif [ "$DATASETNAME" = "nq" ]; then
        # ALPHAS=(0.05 0.10 0.15 0.20)
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

					.venv/bin/python main.py \
						--tag ${TAG} \
						--dataset_name data/${DATASETNAME}/${MDLNAME}/final  \
						--model $MDLNAME \
						--exp_name ${DATASETNAME}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-alpha-${ALPHA} \
						--output_dir snapshots/${METHOD}/${DATASETNAME}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-alpha-${ALPHA} \
						--method $METHOD \
						--alpha ${ALPHA} \
						--h_size ${HYPOTHESIS} \
						--T ${T} \
						--confidence_score ${SCORE} \
						--unk_T ${UNK_T}
					done
				done
			done
		done
	done
done

echo "All tasks completed."

#!/bin/bash
set -e
# Static dialog experiment: same dialog setup as run_dialog.sh, run on fixed 500-row chunks of the
# shuffled SQuAD contexts (--static_dialog_chunk), one background job per chunk.
# The chatbot (--model) must support logprobs for fM1. Requires OPENAI_API_KEY.

TAG=NIPS25

ALPHA=0.2
HYPOTHESIS=1000
T=10000000
CHUNKS=(2000 2500 3000) # 0 500 1000 1500 2000 2500 3000

DATASETNAMELIST=(dialog/squad/contexts)
MDLNAMELIST=(gpt-3.5-turbo)

METHODLIST=(Static-Dialog-OSG-Semi-Bandit) # Static-Dialog-OSG Static-Dialog-OSG-Bandit Static-Dialog-OSG-Semi-Bandit

SCORELIST=(fM1)

for DATASETNAME in ${DATASETNAMELIST[*]};
do
	for MDLNAME in ${MDLNAMELIST[*]};
	do
		for METHOD in ${METHODLIST[*]};
		do
			for SCORE in ${SCORELIST[*]};
			do
				for CHUNK in ${CHUNKS[*]};
				do
				echo $TAG, $DATASETNAME, $MDLNAME, $METHOD, $SCORE, $T, $CHUNK

				.venv/bin/python main.py \
					--tag ${TAG} \
					--dataset_name data/${DATASETNAME} \
					--model $MDLNAME \
					--exp_name ${TAG}-${MDLNAME}-${METHOD}-CHUNK-${CHUNK}-no_con \
					--output_dir snapshots/${METHOD}/${TAG}-${MDLNAME}-${METHOD}-CHUNK-${CHUNK}-no_con \
					--method $METHOD \
					--alpha ${ALPHA} \
					--h_size ${HYPOTHESIS} \
					--T ${T} \
					--confidence_score ${SCORE} \
					--static_dialog_chunk ${CHUNK} &
				done
			done
		done
	done
done
wait

#!/bin/bash
set -e
# Dialog experiment: a simulated user (gpt-5.4-mini) asks questions about SQuAD contexts, the
# chatbot (--model) answers, and a labeler (--model) judges each answer online.
# Requires OPENAI_API_KEY (prompted on first use if unset).
#
# Data: data/dialog/squad/contexts (built from SQuAD by generation/squad_context.py)
# Resume: an interrupted run saves <output_dir>/checkpoint.pkl and resumes from it on rerun.

TAG=NIPS25

ALPHA=0.1
HYPOTHESIS=1000
T=15000

DATASETNAMELIST=(dialog/squad/contexts)
MDLNAMELIST=(gpt-5.4)

METHODLIST=(Dialog-OSG-Semi-Bandit) # Dialog-OSG Dialog-OSG-Bandit Dialog-OSG-Semi-Bandit

SCORELIST=(fM2)

for DATASETNAME in ${DATASETNAMELIST[*]};
do
	for MDLNAME in ${MDLNAMELIST[*]};
	do
		for METHOD in ${METHODLIST[*]};
		do
			for SCORE in ${SCORELIST[*]};
			do
			echo $TAG, $DATASETNAME, $MDLNAME, $METHOD, $SCORE, $T

			.venv/bin/python main.py \
				--tag ${TAG} \
				--dataset_name data/${DATASETNAME} \
				--model $MDLNAME \
				--exp_name ${TAG}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-alpha-${ALPHA}-Hs-${HYPOTHESIS} \
				--output_dir snapshots/${METHOD}/${TAG}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-alpha-${ALPHA}-Hs-${HYPOTHESIS} \
				--method $METHOD \
				--alpha ${ALPHA} \
				--h_size ${HYPOTHESIS} \
				--T ${T} \
				--confidence_score ${SCORE} \
				--unk_T False
			done
		done
	done
done

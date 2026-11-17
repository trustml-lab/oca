#!/bin/bash
set -e
# Adversarial dialog experiment: an LLM adversary (gpt-5.4-mini) sees the learner's decision
# history and picks the next question from the pooled NQ + TriviaQA GPT-3.5 generations
# (data/nq/gpt-3.5 + data/triviaqa/gpt-3.5) to make the learner fail. Answers, labels and scores
# are the pre-computed ones in those datasets; only the adversary calls the OpenAI API.
# Requires OPENAI_API_KEY. --dataset_name is ignored for Adversary-* methods (see main.py).

TAG=NIPS25

ALPHA=0.2
HYPOTHESIS=1000
T=15000
SEED=100

DATASETNAMELIST=(nq-triviaqa)
MDLNAMELIST=(gpt-5.4)

METHODLIST=(Adversary-Dialog-OSG-Bandit) # Adversary-Dialog-OSG Adversary-Dialog-OSG-Bandit Adversary-Dialog-OSG-Semi-Bandit

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
				--exp_name real${SEED}-${TAG}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-alpha-${ALPHA}-Hs-${HYPOTHESIS} \
				--output_dir snapshots/${METHOD}/real${SEED}-${TAG}-${MDLNAME}-${METHOD}-${SCORE}-T-${T}-alpha-${ALPHA}-Hs-${HYPOTHESIS} \
				--method $METHOD \
				--alpha ${ALPHA} \
				--h_size ${HYPOTHESIS} \
				--T ${T} \
				--confidence_score ${SCORE} \
				--unk_T False \
				--seed ${SEED}
			done
		done
	done
done

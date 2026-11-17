#!/bin/bash
set -e
# Rebuttal W3 (ii) and W4: discrete shift whose two blocks have different lengths. One NQ block
# and one TriviaQA block (NUM_CHUNKS=1); the pair (CHUNK_NQ, CHUNK_TRI) sums to 60,000 so
# T = 60,000. Table column (p_NQ, p_TriviaQA) = (CHUNK_NQ, CHUNK_TRI) / 60000.
#
# Rebuttal setting: T = 60,000, alpha = 0.10, GPT-5.4 and Llama-3.1-8B.
#   FIRST_FLAG=tri : TriviaQA -> NQ ordering  (W3 table (2), overall FDR)
#   FIRST_FLAG=nq  : NQ -> TriviaQA ordering  (W4 table, per-distribution FDR)
# Results: snapshots/Imbalanced_Shift/*/per_dist_results.json
#   ExAUL.overall.fdr                      -> W3 table (2)
#   ExAUL.per_dist.{NQ,TriviaQA}.fdr       -> W4 table (FDR_NQ, FDR_TriviaQA)

TAG=NIPS25
HYPOTHESIS=1000

NUM_CHUNKS=1
UNK_T=False
SEED=42

MDLNAMELIST=(gpt-5.4 llama3.1-8B)

# "<nq_block> <tri_block>", each pair sums to 60000. 30000 30000 is the balanced reference.
CHUNK_PAIRS=(
  "6000 54000"
  "12000 48000"
  "18000 42000"
  "24000 36000"
  "30000 30000"
  "36000 24000"
  "42000 18000"
  "48000 12000"
  "54000 6000"
)

# which block comes first; block sizes are set by CHUNK_PAIRS and are independent of this
FIRST_FLAG_LIST=(tri nq)

METHODLIST=(Imbalanced_Shift)
SCORELIST=(fM2)

ALPHAS=(0.1)

for MDLNAME in ${MDLNAMELIST[*]};
do
  NQ_DATASET="data/nq/${MDLNAME}/final"
  TRIVIA_DATASET="data/triviaqa/${MDLNAME}/final"
  for METHOD in ${METHODLIST[*]};
  do
    for PAIR in "${CHUNK_PAIRS[@]}";
    do
      read -r CHUNK_NQ CHUNK_TRI <<< "${PAIR}"
      for FIRST_FLAG in ${FIRST_FLAG_LIST[*]};
      do
        for SCORE in ${SCORELIST[@]};
        do
          for ALPHA in ${ALPHAS[@]};
          do
          EXP_NAME=${FIRST_FLAG}-${MDLNAME}-${METHOD}-Cnq-${CHUNK_NQ}-Ctri-${CHUNK_TRI}-alpha-${ALPHA}-nc-${NUM_CHUNKS}
          echo "$TAG, imbalanced shift, $MDLNAME, $METHOD, $SCORE, $HYPOTHESIS, $ALPHA, nq=$CHUNK_NQ, tri=$CHUNK_TRI, $FIRST_FLAG"

            .venv/bin/python main.py \
              --tag ${TAG} \
              --nq_dataset_name ${NQ_DATASET} \
              --triviaqa_dataset_name ${TRIVIA_DATASET} \
              --model ${MDLNAME} \
              --exp_name ${EXP_NAME} \
              --output_dir snapshots/${METHOD}/${EXP_NAME} \
              --method ${METHOD} \
              --alpha ${ALPHA} \
              --chunk_size_nq ${CHUNK_NQ} \
              --chunk_size_tri ${CHUNK_TRI} \
              --num_chunks ${NUM_CHUNKS} \
              --h_size ${HYPOTHESIS} \
              --first_flag ${FIRST_FLAG} \
              --confidence_score ${SCORE} \
              --seed ${SEED} \
              --unk_T ${UNK_T}
          done
        done
      done
    done
  done
done

wait
echo "All tasks completed."
echo "Overall / per-distribution FDR: snapshots/Imbalanced_Shift/*/per_dist_results.json"
echo "Plots: fdrs_plot.png, fdrs_per_dist_plot.png, inefficiencies_per_dist_plot.png"
echo "T is len(dataset) = (nq_block + tri_block) * num_chunks, printed at the start of each run."

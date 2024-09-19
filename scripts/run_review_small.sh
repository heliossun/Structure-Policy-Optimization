# CHUNKS: Globally maximum number of 4-GPUs
# IDX1,IDX2: manually set them when run this file, 
# E.g., IDX1=0 and IDX2=1 on the first node, on the second node will be 2 and 3

CHUNKS=2
IDX1=0
IDX2=1

CKPT="0.5b-sqa-labling"

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/preference_labeling/llava-ov-72b.py \
  --rule scripts/preference_labeling/rule.json \
  --video_folder ./data/video \
  --qafile ./data/sqllava-ov-0.5b/merge.json \
  --answers-file ${CHUNKS}_$IDX1.json \
  --out_dir ./data/labling/$CKPT \
  --num-chunks $CHUNKS \
  --chunk-idx $IDX1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/preference_labeling/llava-ov-72b.py \
  --rule scripts/preference_labeling/rule.json \
  --video_folder ./data/video \
  --qafile ./data/sqllava-ov-0.5b/merge.json \
  --answers-file ${CHUNKS}_$IDX2.json \
  --out_dir ./data/labling/$CKPT \
  --num-chunks $CHUNKS \
  --chunk-idx $IDX2

#wait

# output_file=./data/7b_selfQA_labeling-72b-chat.json

# # Clear out the output file if it exists.
# > "$output_file"



# cat ./data/labling/$CKPT/${CHUNKS}_$IDX1.json >> "$output_file"
# cat ./data/labling/$CKPT/${CHUNKS}_$IDX2.json >> "$output_file"
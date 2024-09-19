CHUNKS=2
CKPT="0.5b-sqa-labling"
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/preference_labeling/llava-ov-72b.py \
  --rule scripts/preference_labeling/rule.json \
  --video_folder ./data/video \
  --qafile ./data/sqllava-ov-0.5b/merge.json \
  --answers-file ${CHUNKS}_0.json \
  --out_dir ./data/labling/$CKPT \
  --num-chunks $CHUNKS \
  --chunk-idx 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/preference_labeling/llava-ov-72b.py \
  --rule scripts/preference_labeling/rule.json \
  --video_folder ./data/video \
  --qafile ./data/sqllava-ov-0.5b/merge.json \
  --answers-file ${CHUNKS}_1.json \
  --out_dir ./data/labling/$CKPT \
  --num-chunks $CHUNKS \
  --chunk-idx 1

wait

output_file=./data/7b_selfQA_labeling-72b-chat.json

# Clear out the output file if it exists.
> "$output_file"



cat ./data/labling/$CKPT/${CHUNKS}_0.json >> "$output_file"
cat ./data/labling/$CKPT/${CHUNKS}_1.json >> "$output_file"

CHUNKS=16    # change this to the number of available GPUs / for multi-gpu node, add & between each job
CKPT="0.5b-sqa-labling"


iDX=0
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.preference_labeling.llava-ov-72b \
  --rule scripts/preference_labeling/rule.json \
  --video_folder ./data/video \
  --image_folder ./data/image \
  --qafile ./data/sqllava-ov-0.5b/selfQA_0.5b.json \
  --answers-file ${CHUNKS}_${IDX}.json \
  --out_dir ./data/labling/$CKPT \
  --num-chunks $CHUNKS \
  --chunk-idx $IDX &

iDX=1
CUDA_VISIBLE_DEVICES=2,3 python -m scripts.preference_labeling.llava-ov-72b \
  --rule scripts/preference_labeling/rule.json \
  --video_folder ./data/video \
  --image_folder ./data/image \
  --qafile ./data/sqllava-ov-0.5b/selfQA_0.5b.json \
  --answers-file ${CHUNKS}_${IDX}.json \
  --out_dir ./data/labling/$CKPT \
  --num-chunks $CHUNKS \
  --chunk-idx $IDX

## Add more python job after this



#wait

# output_file=./data/7b_selfQA_labeling-72b-chat.json

# # Clear out the output file if it exists.
# > "$output_file"



# cat ./data/labling/$CKPT/${CHUNKS}_$IDX1.json >> "$output_file"
# cat ./data/labling/$CKPT/${CHUNKS}_$IDX2.json >> "$output_file"
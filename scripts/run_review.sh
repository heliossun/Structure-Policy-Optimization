python scripts/preference_labeling/llava-ov-72b.py \
  --rule scripts/preference_labeling/rule.json \
  --video_folder ./data/video \
  --output ./data/0.5b_selfQA_labeling-72b-chat.json \
  --qafile ./data/sqllava-ov-0.5b/merge.json


accelerate launch --num_processes 1 llava-ov-72b.py \
  --rule rule.json \
  --video_folder /guohao/data/video_train_zip \
  --output /guohao/data/videoinstruct/m4_instruct_self_QA_labeling-7b.json \
  --qafile data/m4_instruct_self_QA_0.5_10k.json
#python gpt4o.py \
#  --rule rule.json \
#  --video_folder /guohao/data/video_train_zip \
#  --output /guohao/data/videoinstruct/m4_instruct_self_QA_labeling_gpt4o.json

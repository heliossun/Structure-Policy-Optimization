

python -m scripts.merge_lora_base \
    --model-path ./checkpoints/sqllava-7b-qwen-lora-spo-Q-b0.2-lmd10 \
	--model-base ZachSun/sqllava-qwen-7b-interleave \
    --model_name sqllava_qwen-lora-7b \
    --output_dir ./checkpoints/sqllava-qwen-7b-lora-spo-Q-merge

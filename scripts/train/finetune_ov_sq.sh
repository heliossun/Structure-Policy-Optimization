ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=8 --nnodes=1 \
    llava/train/train_mem.py \
    --lora_enable True --lora_r 32 --lora_alpha 64 --mm_projector_lr 1e-5 \
    --deepspeed scripts/zero3.json \
    --model_name_or_path lmms-lab/llava-onevision-qwen2-7b-ov \
    --version qwen_sq\
    --data_path /mnt/disks/new-disk/data/spo/ours_interleave_iv_filtered.json \
    --image_folder /mnt/disks/new-disk/LVLM-reasoning/data/m4-instruct \
    --video_folder /mnt/disks/new-disk/LVLM-reasoning/data/sharegptvideo \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr 5e-6 \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name sqllava-qwen-lora32-7b-interleave-1e5-0.3sq-30frm \
    --output_dir "./checkpoints/sqllava-qwen-lora32-7b-interleave-1e5-0.3sq-30frm" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --ToME False \
    --merging_r 16 \
    --trend -1.0 \
    --sq_r 0.3
# You can delete the sdpa attn_implementation if you want to use flash attn

huggingface-cli upload ZachSun/sqllava-qwen-lora32-7b-interleave-1e5-0.3sq-30frm ./checkpoints/sqllava-qwen-lora32-7b-interleave-1e5-0.3sq-30frm



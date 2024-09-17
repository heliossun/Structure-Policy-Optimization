#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="sqllava-ov-7b"
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.visual_qa \
        --model-path ./checkpoints/sqllava-lora-qwen-7b-ovVideo-v2 \
        --model-base lmms-lab/llava-onevision-qwen2-7b-ov \
        --question-file ./data/m4_instruct_video.json \
        --video-folder ./data/video \
        --model-name sqllava-lora-7b-qwen \
        --answers-file ${CHUNKS}_${IDX}.json \
        --out_dir ./data/selfqa/$CKPT \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

output_file=./data/selfqa/$CKPT/merge.json

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/selfqa/$CKPT/${CHUNKS}_${IDX}.json >> "$output_file"
done





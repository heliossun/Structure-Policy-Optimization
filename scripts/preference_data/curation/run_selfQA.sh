#!/bin/bash


CHUNKS=12   # change this to the number of available GPUs / for multi-gpu node, add & between each job
CKPT="sqllava-ov-7b"

IDX=0    # chunk index, change this for each job 
CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.visual_qa \
    --model-path ./checkpoints/sqllava-lora-qwen-7b-interleave-1e5-0.3sq-30frm \
    --model-base lmms-lab/llava-onevision-qwen2-7b-ov \
    --question-file ./data/ours_interleave_sampler_selfqa.json \
    --model-name sqllava-lora-7b-qwen \
    --video-folder ./data/video \
    --image-folder ./data/image \
    --answers-file ${CHUNKS}_${IDX}.json \
    --out_dir ./data/selfqa/$CKPT \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX &

IDX=1
CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.visual_qa \
    --model-path ./checkpoints/sqllava-lora-qwen-7b-interleave-1e5-0.3sq-30frm \
    --model-base lmms-lab/llava-onevision-qwen2-7b-ov \
    --question-file ./data/ours_interleave_sampler_selfqa.json \
    --model-name sqllava-lora-7b-qwen \
    --video-folder ./data/video \
    --image-folder ./data/image \
    --answers-file ${CHUNKS}_${IDX}.json \
    --out_dir ./data/selfqa/$CKPT \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX &

IDX=2
CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.visual_qa \
    --model-path ./checkpoints/sqllava-lora-qwen-7b-interleave-1e5-0.3sq-30frm \
    --model-base lmms-lab/llava-onevision-qwen2-7b-ov \
    --question-file ./data/ours_interleave_sampler_selfqa.json \
    --model-name sqllava-lora-7b-qwen \
    --video-folder ./data/video \
    --image-folder ./data/image \
    --answers-file ${CHUNKS}_${IDX}.json \
    --out_dir ./data/selfqa/$CKPT \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX &

IDX=3
CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.visual_qa \
    --model-path ./checkpoints/sqllava-lora-qwen-7b-interleave-1e5-0.3sq-30frm \
    --model-base lmms-lab/llava-onevision-qwen2-7b-ov \
    --question-file ./data/ours_interleave_sampler_selfqa.json \
    --model-name sqllava-lora-7b-qwen \
    --video-folder ./data/video \
    --image-folder ./data/image \
    --answers-file ${CHUNKS}_${IDX}.json \
    --out_dir ./data/selfqa/$CKPT \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX &

## Add more python job after this
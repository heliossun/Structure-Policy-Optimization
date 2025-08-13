import os
import argparse
import torch
import json
from collections import defaultdict
import sys
sys.path.insert(0,'/home/gs4288/LLaVA-NeXT')
from llava.model import *

def parse_args():
    parser = argparse.ArgumentParser(description='Extract MMProjector weights')
    parser.add_argument('--model-path', type=str, help='model folder')
    parser.add_argument('--output', type=str, help='output file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    keys_to_match = ['mm_projector','vision_resampler',"vision_tower"]
    to_return={}
    
    lora_cfg_pretrained = LlavaQwenConfig.from_pretrained(args.model_path)
    #model_indices = json.load(open(os.path.join(args.model_path, 'pytorch_model.bin.index.json')))
    model_indices=LlavaQwenForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation="flash_attention_2")
    
    for k, t in model_indices.named_parameters():
        
        if "vision_tower" in k:
            #print(t)
            to_return[k]=t.detach().cpu().to(dtype=torch.bfloat16)
            print(k,to_return[k].shape)
    

    loaded_weights = {}

    # for ckpt_name, weight_keys in ckpt_to_key.items():
    #     ckpt = torch.load(os.path.join(args.model_path, ckpt_name), map_location='cpu')
    #     for k in weight_keys:
    #         loaded_weights[k] = ckpt[k]
    # print(loaded_weights.keys())
    torch.save(to_return, args.output)

# -*- coding: utf-8 -*-


import argparse
import sys
import torch
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument('--output_dir', default='./merged', type=str)
    parser.add_argument("--lora_pretrain", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--vit_name", type=str, default=None)
    args = parser.parse_args()

    disable_torch_init()
    #model_path = os.path.expanduser(args.model_path)
    #model_name = get_model_name_from_path(model_path)
    device = "cuda"
    device_map = "auto"
    lora_pt = os.path.join(args.model_path, "Vit-lora")
    tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_path, args.model_base, args.model_name, device_map=device_map,lora_pt=lora_pt)

    #print("Saving to Hugging Face format...")
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    #print("Saving VIT")
    #vision_tower.vision_tower.save_pretrained(os.path.join(args.output_dir,args.vit_name))


if __name__ == '__main__':
    main()
import argparse
import json
import os
from datetime import timedelta
import sys
import copy
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import math
from llava.utils import rank0_print
torch.backends.cuda.matmul.allow_tf32 = True

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_frames(video_file, num_frames_to_sample=20):
    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if
                   os.path.isfile(os.path.join(video_file, f))]
    frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

    # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
    total_frames = len(frame_files)
    sampled_indices = np.linspace(0, total_frames - 1, min(total_frames,num_frames_to_sample), dtype=int)

    # Read and store the sampled frames
    video = []
    for idx in sampled_indices:
        frame_path = frame_files[idx]
        try:
            with Image.open(frame_path) as img:
                frame = img.convert("RGB")
                video.append(frame)
        except IOError:
            rank0_print(f"Failed to read frame at path: {frame_path}")
    return video

def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            rank0_print('error', review)
            return [-1, -1]
    except Exception as e:
        rank0_print(e)
        rank0_print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--qafile')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--video_folder')
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()
    device_map = "auto" # this will dispatch model
    device = "cuda"

    pretrained = "lmms-lab/llava-critic-72b"
    model_name = "llava_qwen"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name,
                                                                          device_map=device_map,load_4bit=True)
    model.eval()
    try:
        data_dict = json.load(open(args.qafile, 'r',encoding='utf-8'))
    except:
        data_dict=[json.loads(line) for line in open(args.qafile, encoding='utf-8')]
    data_dict = get_chunk(data_dict, args.num_chunks, args.chunk_idx)


    if not os.path.exists(args.out_dir):
    # If it doesn't exist, create the directory
        os.makedirs(args.out_dir)
    review_file = open(os.path.join(args.out_dir,args.answers_file), 'a')

    # Labeling preference / reject answers and save the results in $review_file.
    conv_template = "qwen_1_5"
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))
    scores=[]
    if os.path.isfile(os.path.join(args.out_dir, args.answers_file)):
        cur_reviews = [json.loads(line) for line in open(os.path.join(args.out_dir, args.answers_file))]
        print(f"Continue from {len(cur_reviews)}.")
    else:
        cur_reviews = []
    conv = copy.deepcopy(conv_templates[conv_template])
    start = 0 if len(cur_reviews) == 0 else len(cur_reviews)
    video_file = None
    image_file = None
    for source in tqdm(data_dict[start:]):
        vis_tok_len=0
        if 'video' in source:
            video_file = source["video"]
            video = os.path.join(args.video_folder, video_file)
            video_frames = load_frames(video,32)
            image_tensors = process_images(video_frames, image_processor, model.config)
            image_sizes = [frame.size for frame in video_frames]
            vis_tok_len=len(video_frames)*196
        elif 'image' in source:
            image_file = source["image"]
            if type(image_file) is list:
                imgs = [Image.open(os.path.join(args.image_folder, img_f)).convert("RGB") for img_f in image_file]

                if len(image_file) > 1:
                    continue
                else:
                    image_sizes = [img.size for img in imgs]
                    image_tensors = process_images(imgs, image_processor, model.config)
            else:
                img = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
                image_sizes = [img.size]
                image_tensors = process_images([img], image_processor, model.config)
            vis_tok_len = 10 * 729
        image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
        qs=source['questions']
        answers=source['answers']
        system_sq = rule_dict['self_q_critique']['prompt']
        system_asr = rule_dict['asr_critique']['prompt']
        content_sq = f"{DEFAULT_IMAGE_TOKEN}\n{system_sq}\nQuestion 1: {qs[0]}\nQuestion 2: {qs[1]}"
        content_asr1 = f"{DEFAULT_IMAGE_TOKEN}\n{system_asr}\nQuestion: {qs[0]}\nAnswer 1: {answers[0]}\nAnswer 2: {answers[1]}"
        content_asr2 = f"{DEFAULT_IMAGE_TOKEN}\n{system_asr}\nQuestion: {qs[1]}\nAnswer 1: {answers[2]}\nAnswer 2: {answers[3]}"
        contents=[content_sq, content_asr1, content_asr2]
        score=[]
        tok_len=0
        for i,content in enumerate(contents):
            conv.clear_message()
            conv.append_message(conv.roles[0], content)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
                0).to(device)
            cont = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
                use_cache=True
            )
            if i!=0:
                tok_len+=input_ids.shape[1]+vis_tok_len
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            score.append(parse_score(text_outputs))

        if score[0][0]>score[0][1]:
            q_w=qs[0]
            q_l=qs[1]
            if score[1][0]>score[1][1]:
                a1_w=answers[0]
                a1_l=answers[1]
            else:
                a1_w=answers[1]
                a1_l=answers[0]
            if score[2][0]>score[2][1]:
                a2_w=answers[2]
                a2_l=answers[3]
            else:
                a2_w=answers[3]
                a2_l=answers[2]
        else:
            q_w=qs[1]
            q_l=qs[0]
            if score[1][0]>score[1][1]:
                a2_w=answers[0]
                a2_l=answers[1]
            else:
                a2_w=answers[1]
                a2_l=answers[0]
            if score[2][0]>score[2][1]:
                a1_w=answers[2]
                a1_l=answers[3]
            else:
                a1_w=answers[3]
                a1_l=answers[2]
        if video_file:
            visual_modality="video"
            visual_file=video_file
        else:
            visual_modality="image"
            visual_file=image_file
        cur_review={visual_modality: visual_file,"sampler":source['sampler'],"c_pref": {'q':q_w,'a_w':a1_w,'a_l':a1_l},"c_rej":{'q':q_l,'a_w':a2_w,'a_l':a2_l},"tok_len":tok_len,'score':score}
        review_file.write(json.dumps(cur_review)+'\n')
        review_file.flush()
    review_file.close()

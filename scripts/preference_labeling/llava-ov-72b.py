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
sys.path.insert(0,'/home/gs4288/stepQ-LVA')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from llava.utils import rank0_print
torch.backends.cuda.matmul.allow_tf32 = True

def load_frames(video_file, num_frames_to_sample=10):
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
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()
    device_map = "auto" # this will dispatch model
    device = "cuda"

    pretrained = "lmms-lab/llava-onevision-qwen2-72b-ov-chat"
    model_name = "llava_qwen"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name,
                                                                          device_map=device_map,
                                                                          attn_implementation="sdpa")
    model.eval()

    data_dict = json.load(open(args.qafile, 'r'))
    review_file = open(f'{args.output}', 'a')
    # Labeling preference / reject answers and save the results in $review_file.
    conv_template = "qwen_1_5"
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))
    scores=[]
    for source in tqdm(data_dict):
        conv = copy.deepcopy(conv_templates[conv_template])
        video_file = source["video"]
        video = os.path.join(args.video_folder, video_file)
        video_frames = load_frames(video,num_frames_to_sample=32)
        image_tensors = process_images(video_frames, image_processor, model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
        image_sizes = [frame.size for frame in video_frames]
        qs=source['questions']
        answers=source['answers']
        system_sq = rule_dict['self_q']['prompt']
        system_asr = rule_dict['preference']['prompt']
        content_sq = f"{DEFAULT_IMAGE_TOKEN}\nQuestion 1: {qs[0]}\nQuestion 2: {qs[1]}\n{system_sq}"
        content_asr1 = f"{DEFAULT_IMAGE_TOKEN}\nQuestion: {qs[0]}\nAnswer 1: {answers[0]}\nAnswer 2: {answers[1]}\n{system_asr}"
        content_asr2 = f"{DEFAULT_IMAGE_TOKEN}\nQuestion: {qs[1]}\nAnswer 1: {answers[2]}\nAnswer 2: {answers[3]}\n{system_asr}"
        contents=[content_sq, content_asr1, content_asr2]
        score=[]
        for content in contents:
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
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            score.append(parse_score(text_outputs))
        if min(score[0])>=5.0:
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
            cur_review={"video": video_file,"sampler":source['sampler'],"c_pref": {'q':q_w,'a_w':a1_w,'a_l':a1_l},"c_rej":{'q':q_l,'a_w':a2_w,'a_l':a2_l}}
            review_file.write(json.dumps(cur_review)+'\n')
            review_file.flush()
    review_file.close()

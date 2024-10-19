from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import rank0_print
import os
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
import argparse
import json
from tqdm import tqdm
import math
import random
random.seed(10)
warnings.filterwarnings("ignore")
# Load the OneVision model


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
# Function to extract frames from video
def load_frames(video_file, num_frames_to_sample=10):
    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if
                   os.path.isfile(os.path.join(video_file, f))]
    frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

    # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
    total_frames = len(frame_files)
    sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

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

def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    return frames

def eval_model(args):

    device = "cuda"
    device_map = "auto"
    model_path = os.path.expanduser(args.model_path)
    lora_pt=None
    if args.model_base:
        lora_pt = os.path.join(args.model_path, "Vit-lora")
    tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, args.model_base, args.model_name, device_map=device_map,lora_pt=lora_pt)

    model.eval()
    data_dict = json.load(open(args.question_file,'r'))

    data_dict = get_chunk(data_dict, args.num_chunks, args.chunk_idx)
    rank0_print("chunk size:",len(data_dict))
    if not os.path.exists(args.out_dir):
    # If it doesn't exist, create the directory
        os.makedirs(args.out_dir)
    out_file = open(os.path.join(args.out_dir,args.answers_file), 'a',encoding='utf-8')


    video_file=None
    image_file=None
    # we only use single image and video data for preference data generation
    for source in tqdm(data_dict):
        token_len=0
        if 'video' in source:
            video_file = source["video"]
            video = os.path.join(args.video_folder, video_file)
            video_frames = load_frames(video,32)
            image_tensors = process_images(video_frames, image_processor, model.config)
            image_sizes = [frame.size for frame in video_frames]
        elif 'image' in source:
            image_file = source["image"]
            if type(image_file) is list:
                imgs=[Image.open(os.path.join(args.image_folder, img_f)).convert("RGB") for img_f in image_file]

                if len(image_file) > 1:
                    continue
                else:
                    image_sizes = [img.size for img in imgs]
                    image_tensors = process_images(imgs,image_processor, model.config)
            else:
                img = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
                image_sizes = [img.size]
                image_tensors = process_images([img], image_processor, model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
        conv_template = "qwen_sq"
        fq=source['conversations'][0]['value'].replace('<image>\n','')
        fq=fq.replace('\n<image>','')
        fqs = f"{DEFAULT_IMAGE_TOKEN}\n{fq}"
        first_answer=source['conversations'][1]['value']
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.clear_message()

        # SQ
        conv.append_message(conv.roles[0], fqs)
        conv.append_message(conv.roles[1], first_answer)
        conv.append_message(conv.roles[2], None)
        prompt = conv.get_prompt()
        #rank0_print(prompt)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
            0).to(device)

        questions = []
        answers = []
        for i in range(args.n_shot):
            cont = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=True,
                temperature=1.2,
                max_new_tokens=4096,
                top_k=300,
                top_p=0.95,
                use_cache=True
            )

            sq= tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            questions.append(sq)
        conv.clear_message()

        for q in questions:
            #rank0_print("questions: ",q)
            conv.append_message(conv.roles[0], fqs)
            conv.append_message(conv.roles[1], first_answer)
            conv.append_message(conv.roles[2], q)
            conv.append_message(conv.roles[1],None)
            prompt = conv.get_prompt()
            #rank0_print(prompt)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
                0).to(device)
            for i in range(2):
                cont = model.generate(
                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=True,
                    temperature=0.75,
                    max_new_tokens=4096,
                    top_k=300,
                    top_p=0.95,
                    use_cache=True
                )
                answer = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
                answers.append(answer)
                token_len+=input_ids.shape[1]+cont.shape[1]
            conv.clear_message()
        if video_file:
            visual_modality="video"
            visual_file=video_file
            token_len+=196*len(video_frames)*2
        else:
            visual_modality="image"
            visual_file=image_file
            token_len+=7290*2
        out_p={"id": source['id'],
               visual_modality: visual_file,
               "sampler": [fqs,first_answer],
               "questions": questions,
               "answers":answers,
               "token_len":token_len,}

        try:
            out_file.write(json.dumps(out_p)+'\n')
            out_file.flush()
        except:
            pass
    out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--n_shot", type=int, default=2)
    parser.add_argument("--sampleNum", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)

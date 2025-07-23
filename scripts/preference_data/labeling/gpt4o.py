import argparse
import json
import os
import base64
from openai import OpenAI
import time
import numpy as np
from tqdm import tqdm
import math
NUM_SECONDS_TO_SLEEP = 5
client = OpenAI(api_key="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")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_task(v, content: str, max_tokens: int, id:int,idx:int):
    if type(v) is list:
        content=[content,*map(lambda x: {"type":"image_url", "image_url":{"url":f'data:image/jpg;base64,{x}',"detail":"low"}}, v)]
    else:
        content=[content,{"type": "image_url","image_url": { "url": f"data:image/jpeg;base64,{v}"}}]
    task={
        "custom_id": f"task-{id}-{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": "gpt-4o",
            "temperature": 0,
            "response_format": { 
                "type": "json_object"
            },
            "messages": [
                {
                    'role': 'user',
                    'content': content,
                }
            ],
        }
    }
    return task

            
            

    return response.choices[0].message.content
def get_eval(v, content: str, max_tokens: int):
    if type(v) is list:
        mix_content=[content,*map(lambda x: {"type":"image_url", "image_url":{"url":f'data:image/jpg;base64,{x}',"detail":"low"}}, v)]
    else:
        mix_content=[content,{"type": "image_url","image_url": { "url": f"data:image/jpeg;base64,{v}"}}]
    while True:
        try:
            response = client.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {
                        'role': 'user',
                        'content': mix_content,
                    }, ],
                temperature=0,
                max_tokens=max_tokens,
            )
            break

        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response.choices[0].message.content


def parse_score(review):
    try:
        score_pair = review.split('\n')[-1]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            return [-1, -1]
    except Exception as e:
        return [-1, -1]

def load_frames(video_file, num_frames_to_sample=30):
    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if
                   os.path.isfile(os.path.join(video_file, f))]
    frame_files.sort()  # Ensure the frames are sorted if they are named sequentially
    video = []
    total_frames = len(frame_files)
    sampled_indices = np.linspace(0, total_frames - 1, min(total_frames, num_frames_to_sample), dtype=int)

    for idx in sampled_indices:
        frame_path = frame_files[idx]
        video.append(encode_image(frame_path))
    return video


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--qafile')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('--video_folder')
    parser.add_argument('--image_folder')
    parser.add_argument('--max-tokens', type=int, default=256, help='maximum number of tokens produced in the output')
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    try:
        data_dict = json.load(open(args.qafile, 'r', encoding='utf-8'))
    except:
        data_dict = [json.loads(line) for line in open(args.qafile, encoding='utf-8')]
    data_dict = get_chunk(data_dict, args.num_chunks, args.chunk_idx)
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))
    if not os.path.exists(args.out_dir):
        # If it doesn't exist, create the directory
        os.makedirs(args.out_dir)
    scores = []
    if os.path.isfile(os.path.join(args.out_dir, args.answers_file)):
        cur_reviews = [json.loads(line) for line in open(os.path.join(args.out_dir, args.answers_file))]
        print(f"Continue from {len(cur_reviews)}.")
    else:
        cur_reviews = []

    review_file = open(os.path.join(args.out_dir, args.answers_file), 'a',encoding='utf-8')
    # Labeling preference / reject answers and save the results in $review_file.
    start =0 if len(cur_reviews) == 0 else len(cur_reviews)
    
    tasks=[]
    for id,d in enumerate(data_dict[start:]):
        video_file = None
        image_file = None
        if 'video' in d:
            video_file = d["video"]
            video = os.path.join(args.video_folder, video_file)
            v = load_frames(video,30)
        elif 'image' in d:
            image_file = d["image"]
            
            if type(image_file) is list:
                if len(image_file) > 1:
                    continue
                else:
                    v=encode_image(os.path.join(args.image_folder, image_file[0]))
            else:
                v=encode_image(os.path.join(args.image_folder, image_file))
        
        qs=d['questions']
        answers=d['answers']
        system_sq = rule_dict['self_q']['prompt']
        system_asr = rule_dict['asr_scoring']['prompt']
        content = (f'[Question 1]\n{qs[0]}\n\n[End of Answer 1]\n\n'
                    f'[Question 2]\n{qs[1]}\n\n[End of Answer 2]\n\n'
                    f'[System]\n{system_sq}\n\n')
        content1 = (f'[Question]\n{qs[0]}\n\n'
                   f'[Answer 1]\n{answers[0]}\n\n[End of Answer 1]\n\n'
                   f'[Answer 2]\n{answers[1]}\n\n[End of Answer 2]\n\n'
                   f'[System]\n{system_asr}\n\n')
        content2 = (f'[Question]\n{qs[1]}\n\n'
                    f'[Answer 1]\n{answers[2]}\n\n[End of Answer 1]\n\n'
                    f'[Answer 2]\n{answers[3]}\n\n[End of Answer 2]\n\n'
                    f'[System]\n{system_asr}\n\n')
        contents=[content,content1,content2]
        score=[]
        for idx,content in enumerate(contents):
            tsk=get_task(v,content,args.max_tokens,id,idx)
            tasks.append(tsk)
            review_file.write(json.dumps(tsk) + '\n')
            review_file.flush()
    
    #out_file = os.path.join(args.out_dir, args.answers_file)
    print("task numbers: ",len(tasks))
    #print("+++++",tasks[0])
    #print("-----",tasks[-1])
    # with open(out_file, 'w') as file:
    #     for obj in tasks:
    #         file.write(json.dumps(obj) + '\n')        

    review_file.close()










        #for content in contents:

            #review = get_eval(v, content, args.max_tokens)
            #score.append(parse_score(review))
        #print(score)
        #if min(score[0]) >= 5.0:
    #     if score[0][0] > score[0][1]:
    #         q_w = qs[0]
    #         q_l = qs[1]
    #         if score[1][0] > score[1][1]:
    #             a1_w = answers[0]
    #             a1_l = answers[1]
    #         else:
    #             a1_w = answers[1]
    #             a1_l = answers[0]
    #         if score[2][0] > score[2][1]:
    #             a2_w = answers[2]
    #             a2_l = answers[3]
    #         else:
    #             a2_w = answers[3]
    #             a2_l = answers[2]
    #     else:
    #         q_w = qs[1]
    #         q_l = qs[0]
    #         if score[1][0] > score[1][1]:
    #             a2_w = answers[0]
    #             a2_l = answers[1]
    #         else:
    #             a2_w = answers[1]
    #             a2_l = answers[0]
    #         if score[2][0] > score[2][1]:
    #             a1_w = answers[2]
    #             a1_l = answers[3]
    #         else:
    #             a1_w = answers[3]
    #             a1_l = answers[2]
    #     if video_file:
    #         visual_modality="video"
    #         visual_file=video_file
    #     else:
    #         visual_modality="image"
    #         visual_file=image_file
    #     cur_review = {visual_modality: visual_file, "sampler": d['sampler'],
    #                   "c_pref": {'q': q_w, 'a_w': a1_w, 'a_l': a1_l},
    #                   "c_rej": {'q': q_l, 'a_w': a2_w, 'a_l': a2_l}, 'scores':score}
    #     review_file.write(json.dumps(cur_review) + '\n')
    #     review_file.flush()
    # review_file.close()




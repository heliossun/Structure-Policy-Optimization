import argparse
import json
import os
import base64
from openai import OpenAI
import time
import numpy as np
NUM_SECONDS_TO_SLEEP = 0.5
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_eval(video, content: str, max_tokens: int):

    while True:
        try:
            response = client.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            content,
                            *map(lambda x: {"type":"image_url",
                                            "image_url":{"url":f'data:image/jpg;base64,{x}',"detail":"low"}}, video)

                        ],
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
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]

def load_frames(video_file, num_frames_to_sample=100):
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
    parser.add_argument('-q', '--question')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--video_folder')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    qa_file = "/guohao/data/videoinstruct/m4_instruct_self_QA.json"
    data_dict = json.load(open(qa_file, 'r'))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    scores = []
    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
        print(f"Continue from {len(cur_reviews)+1}.")
    else:
        cur_reviews = []
    review_file = open(f'{args.output}', 'a')
    # Labeling preference / reject answers and save the results in $review_file.
    start =0 if len(cur_reviews) == 0 else len(cur_reviews)+1
    for d in data_dict[start:]:
        video_file = d["video"]
        video = os.path.join(args.video_folder, video_file)
        video_frames = load_frames(video)
        qs=d['questions']
        answers=d['answers']
        system_sq = rule_dict['self_q']['prompt']
        system_asr = rule_dict['preference']['prompt']
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
        for content in contents:
            review = get_eval(video_frames, content, args.max_tokens)
            score.append(parse_score(review))
        if min(score[0])>=6.0 and min(score[1])>=6.0 and min(score[2])>=6.0:
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
            cur_review={"video": video_file, "sampler": d['sampler'], "c_pref": {'q': q_w, 'a_w': a1_w, 'a_l': a1_l},
                 "c_rej": {'q': q_l, 'a_w': a2_w, 'a_l': a2_l}}
            review_file.write(json.dumps(cur_review) + '\n')
            review_file.flush()
    review_file.close()




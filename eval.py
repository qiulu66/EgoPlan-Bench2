import os
import tqdm
import csv
import re
import glob
import json
import torch
import random
import numpy as np
import imageio
import argparse

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

QA_template = """
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.

Considering the progress shown in the video and my current observation in the last frame, what action should I take next in order to {}?

A. {}
B. {}
C. {}
D. {}
"""

def extract_characters_regex(s):
    # https://github.com/thanku-all/parse_answer/blob/main/eval_your_results.py
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ""
    return matches[0]

def cut_keyframes(video_dir, qa_id, start_frame_id, end_frame_id, frame_number, keyframes_dir):
    frame_idx = np.linspace(start_frame_id, end_frame_id, frame_number, endpoint=True, dtype=int)
    print(f"start frame id: {start_frame_id}, end frame id: {end_frame_id}, sampled frames: {frame_idx}")

    if not os.path.exists(os.path.join(keyframes_dir, qa_id)):
        os.makedirs(os.path.join(keyframes_dir, qa_id))
    
    clip = imageio.get_reader(os.path.join(video_dir, qa_id.split('_')[0]+'.mp4'))

    for idx, frame_id in enumerate(frame_idx):
        frame = clip.get_data(frame_id)
        imageio.imwrite(os.path.join(keyframes_dir, qa_id, f'frame-{idx}_frameID-{frame_id}.png'), frame)

def cut_video_clip(video_dir, qa_id, start_frame_id, end_frame_id, clip_dir):
    if not os.path.exists(clip_dir):
        os.makedirs(clip_dir)

    clip = imageio.get_reader(os.path.join(video_dir, qa_id.split('_')[0]+'.mp4'))
    fps = clip.get_meta_data()['fps']

    writer = imageio.get_writer(os.path.join(clip_dir, qa_id+'.mp4'), fps=fps)
    for i in range(start_frame_id, end_frame_id + 1):
        frame = clip.get_data(i)
        writer.append_data(frame)
    writer.close()

def build_model(model_name, weight_dir):
    if model_name == 'longva':
        from models.LongVA.inference import VLM
        model = VLM(weight_dir)
        input_type = 'video'
    elif model_name == 'internvl2':
        from models.InternVL2 import VLM
        model = VLM(weight_dir)
        input_type = 'image'

    return model, input_type

def run_inference(model, input_type, qa_anno, video_dir, output_dir, clip_dir, keyframes_dir, frame_number):
    count, correct = 0, 0
    output_f = open(os.path.join(output_dir), "a")

    for qa_item in qa_anno:
        qa_id = qa_item['sample_id']
        start_frame_id = qa_item['task_start_frame']
        end_frame_id = qa_item['current_observation_frame']

        text_input = QA_template.format(
            qa_item['task_goal'],
            qa_item['choice_a'],
            qa_item['choice_b'],
            qa_item['choice_c'],
            qa_item['choice_d']
        )
        
        if input_type == 'video':
            visual_input = os.path.join(clip_dir, qa_id+'.mp4')
            if not os.path.exists(visual_input):
                cut_video_clip(video_dir, qa_id, start_frame_id, end_frame_id, clip_dir)
        elif input_type == 'image':
            if not os.path.exists(os.path.join(keyframes_dir, qa_id)):
                cut_keyframes(video_dir, qa_id, start_frame_id, end_frame_id, frame_number, keyframes_dir)
            visual_input = [glob.glob(os.path.join(keyframes_dir, qa_id, f'frame-{x}_frameID-*.png'))[0] for x in range(frame_number)]
        print(f"\ntext input: {text_input}")
        print(f"\nvisual input: {visual_input}")

        response = model.inference(text_input, visual_input)
        extraction = extract_characters_regex(response).upper()
        print(f"\nmodel response: {response}, extracted answer: {extraction}, ground truth: {qa_item['golden_choice_idx']}")

        count += 1
        correct += extraction == qa_item['golden_choice_idx']
        print(f"\nQA nums: {count}, correct: {correct}, acc: {correct/count}\n")

        answer_record = {
            "sample_id": qa_id,
            "ground_truth": qa_item['golden_choice_idx'],
            "model_response": response,
            "extracted_answer": extraction,
            "count": count,
            "correct": correct,
            "acc": correct/count
        }
        output_f.write(json.dumps(answer_record) + "\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--model', type=str)
    parser.add_argument('--weight_dir', type=str)
    parser.add_argument('--video_dir', type=str)
    parser.add_argument('--anno_path', type=str)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--visual_input_dir', type=str, default='visual')
    parser.add_argument('--frame_number', type=int, default=8) # only for image models
    args = parser.parse_args()

    model, input_type = build_model(args.model, args.weight_dir)
    qa_anno = json.load(open(args.anno_path))
    video_dir = args.video_dir
    output_dir = os.path.join(args.output_dir, args.model+'.json')
    clip_dir = os.path.join(args.visual_input_dir, 'clip_dir')
    keyframes_dir = os.path.join(args.visual_input_dir, 'keyframes_dir')
    frame_number = args.frame_number

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f'evaluating.. {args.model}')
    run_inference(model, input_type, qa_anno, video_dir, output_dir, clip_dir, keyframes_dir, frame_number)
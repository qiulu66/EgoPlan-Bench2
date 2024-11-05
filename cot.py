import os
import re
import csv
import json
import glob
import argparse

from CoT.call_gpt import request_base64_gpt4v, encode_image

system_prompt='''
You are a task planner. You must answer the question to determine what action to take from four options of [OPTION] based on the task progress and current observation state.
'''

incontext_prompt='''
# INPUT
[CURRENT OBSERVATION]: The current observation image from the first-person perspective, usually the last frame before the action about to be taken begins.
[VISUAL PROMPT]: The current observation image with bounding boxes which outline key objects that may be relevant to the task, indicating their position and status.
[HISTORICAL PROGRESS]: Textual descriptions of first-person perspective videos, about natural human activities. Each line represents a description of the process of a completed action. At the beginning of each line, the #C indicates the image seen from your point of view.
[QUESTION]: A question about video that needs to be answered.
[OPTION]: Four candidates for the question, each option representing a possible action about to be taken.

# OUTPUT FORMAT
Please analyze the question step by step based on the following steps:
1. [TASK PROGRESS ANALYSIS]: Analyze the task progress based on [HISTORICAL PROGRESS]. If the option has already been completed, do not choose to repeat the action.
2. [CURRENT OBSERVATION DESCRIPTION]: Based on [CURRENT OBSERVATION] and [VISUAL PROMPT], describe what activity you are engaged in, what objects your hands are interacting with, whether the task-related objects mentioned in the options are visible, and what their status is if they are visible.
3. [OPTION REASONING]: Reason about the [OPTION]. You should reason whether the option is suitable as the next action based on [TASK PROGRESS ANALYSIS], and determine if the option can be completed in the current state based on [CURRENT OBSERVATION DESCRIPTION].
4. [ANSWER]: You SHOULD choose the most likely action you are about to take and respond with only the letter (A, B, C, or D) of the best option.

Now, please analyze the following case:
[CURRENT OBSERVATION]
<observation>
[VISUAL PROMPT]
<obj>
[HISTORICAL PROGRESS]
{history}
[QUESTION]
Considering the historical task progress and the current observation image, what action should you take next in order to {goal}?
[OPTION]
A. {choice_a}  B. {choice_b}  C. {choice_c}  D. {choice_d}
"OUTPUT":
'''

def extract_characters_regex(s):
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

def gpt4v_interleave(instruction, images_path, extra_images_path = None, system_prompt = None, seed = 42):

    content_images = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(x)}", "detail": "low"},
        } for x in images_path
    ]

    if extra_images_path:
        content_images_extra = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(x)}", "detail": "high"},
            } for x in extra_images_path
        ]
    
    if system_prompt:
        system_content = [{"type": "text", "text": system_prompt}]
    else:
        system_content = None

    texts = re.split('<observation>|<obj>', instruction)
    message_content = [{"type": "text", "text": texts[0]}] + content_images + [{"type": "text", "text": texts[1]}] + content_images_extra + [{"type": "text", "text": texts[2]}]
    
    print(f"SYSTEM:\n{system_content}\nUSER:\n")
    for x in message_content:
        if x["type"] == "text":
            print(x)
        else:
            print("<image>")
    
    response = request_base64_gpt4v(message_content, system_content, seed)
    return response

def run_inference(qa_anno, keyframes_dir, cap_dir, bbox_dir, output_dir, seed):
    count, correct = 0, 0
    output_f = open(os.path.join(output_dir), "a")

    for qa_item in qa_anno:
        qa_id = qa_item['sample_id']
        
        # multimodal prompts
        history = json.load(open(f"{cap_dir}/{qa_id}.json"))["Caption"]
        images_path = [glob.glob(os.path.join(keyframes_dir, qa_id, f'frame-7_frameID-*.png'))[0]]
        obj_images_path = [f"{bbox_dir}/{qa_id}.jpg"]
        print(f"\nimage paths: {images_path}")
        print(f"\nextra image paths: {obj_images_path}")

        instruction = incontext_prompt.format(
            history = history,
            goal = qa_item['task_goal'],
            choice_a = qa_item['choice_a'],
            choice_b = qa_item['choice_b'],
            choice_c = qa_item['choice_c'],
            choice_d = qa_item['choice_d']
        )

        response = gpt4v_interleave(instruction, images_path, extra_images_path = obj_images_path, system_prompt = system_prompt, seed = seed)
        extraction = extract_characters_regex(re.split('4\. \[ANSWER\]:', response)[-1]).upper()
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--anno_path', type=str)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--visual_input_dir', type=str, default='visual')
    args = parser.parse_args()

    qa_anno = json.load(open(args.anno_path))
    output_dir = os.path.join(args.output_dir, f'CoT_{args.seed}.json')
    keyframes_dir = os.path.join(args.visual_input_dir, 'keyframes_dir')
    cap_dir = os.path.join(args.visual_input_dir, 'cot', 'cap')
    bbox_dir = os.path.join(args.visual_input_dir, 'cot', 'bounding_box')

    run_inference(qa_anno, keyframes_dir, cap_dir, bbox_dir, output_dir, args.seed)
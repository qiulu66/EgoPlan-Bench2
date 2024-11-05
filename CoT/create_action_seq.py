# modify from https://github.com/Kkskkkskr/EPD/blob/main/gpt4o_extraction.py

import os
import re
import time
import ast
import openai
import pandas as pd
import collections
import numpy as np
import imageio
from multiprocessing.pool import Pool
import base64
import json, random
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from tqdm import tqdm
import argparse

from call_gpt import request_base64_gpt4v, encode_image

systerm='''
You are a visual question answering expert. Based on the [VIDEO], [OVERALL GOAL] and [COMPLETED STEPS], you can accurately determine the [SUB GOAL] of this video. 
'''
incontext_prompt='''
[VIDEO]: Four frames extracted from a two-second video clip, demonstrating the process of achieving a specific sub-goal. Due to the short duration, typically only a minor action is performed.
[OVERALL GOAL]: The overall goal that is achieved through gradual completion of many sub-goals. Reference information can be provided if the video does not clearly identify the sub-goal.
[COMPLETED STEPS]: Actions already completed before the start of the video, establishing prerequisites for the current sub-goal.
[SUB GOAL]: The specific sub-goal the person in the first-person perspective aims to achieve in the video.

Your task is to output the [SUB GOAL] as a brief phrase based on the [VIDEO], [OVERALL GOAL], and [COMPLETED STEPS].
I will give you an example as follow:

Example 1:
[VIDEO] 
<video>
[OVERALL GOAL]
Add garlic to the food and stir
[COMPLETED STEPS]
none
[SUB GOAL]
"pick up the knife"
/*At the beginning of the video, the knife is initially on the table. Through interaction with the person's hand, the knife's status changes to being held in the hand. As for the other objects, the hand reaches towards the broccoli because it is reaching for the knife, while the garlic remains on the cutting board from start to finish. Therefore, the sub-goal and primary action of this video clip is to pick up the knife.*/

Example 2:
[VIDEO] 
<video>
[OVERALL GOAL]
Store the food in the refrigerator
[COMPLETED STEPS]
cover the container; 
pick up container. 
[SUB GOAL]
"carry containers to the refrigerator"
/*In the video, the first-person perspective is moving towards the refrigerator. Although the last frame shows the person about to open the refrigerator, the sub-goal refers to actions completed within the video, not actions that are about to be completed. Therefore, the sub-goal is to carry containers to the refrigerator.*/

Note that comments within /**/ are only for explanatory purposes and should not be included in the output.
Now, you should determine the sub-goal of this video based on the [VIDEO], [OVERALL GOAL], and [COMPLETED STEPS]. You SHOULD follow the format of example.
1. Observe the changes in the state of objects throughout the video frames. Identify the action that results in an actual change in the state of an object. Do not infer actions that are about to be performed; focus only on actions that are completed within the video frames.
2. Please output the subgoal as a brief phrase and do not output anything else!


[VIDEO]
The four frames are shown.
[OVERALL GOAL]
{task_goal}
[COMPLETED STEPS]
{c_steps}
[SUB GOAL]
'''

def cut_keyframes(frame_number, start_frame_id, stop_frame_id, qa_id, action_seg_frame_dir, video_dir):
    video_id = qa_id.split('_')[0]

    frame_idx = np.linspace(start_frame_id, stop_frame_id, frame_number, endpoint=True, dtype=int)
    
    if not os.path.exists(os.path.join(action_seg_frame_dir, video_id)):
        os.makedirs(os.path.join(action_seg_frame_dir, video_id))
    
    clip = imageio.get_reader(os.path.join(video_dir, video_id+'.mp4'))
    
    images_path = []
    for idx, frame_id in enumerate(frame_idx):
        image_path = os.path.join(action_seg_frame_dir, video_id, f'{video_id}_frameID-{frame_id}.png')
        frame = clip.get_data(frame_id)
        imageio.imwrite(image_path, frame)
        images_path.append(image_path)

    return images_path

def llm_inference(query, output_dir, action_seg_frame_dir, video_dir):
    qa_id = query["sample_id"]
    try:
        c_steps = ''
        cap_list = []
        for i in range(len(query["task_progress_metadata"])):
            
            start_frame = query["task_progress_metadata"][i]["start_frame"]
            stop_frame = query["task_progress_metadata"][i]["stop_frame"]
            images_path = cut_keyframes(4, start_frame, stop_frame, qa_id, action_seg_frame_dir, video_dir)

            if i != 0:
                instruction=str(qa_id)+"\n"+systerm + "\n" + incontext_prompt.format(task_goal = query['task_goal'], c_steps = c_steps)+"\n"
            else:
                instruction=str(qa_id)+"\n"+systerm + "\n" + incontext_prompt.format(task_goal = query['task_goal'], c_steps = "none")+"\n"
            
            # call gpt4v
            message_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(x)}", "detail": "low"},
                } for x in images_path
            ]
            message_content = [{"type": "text", "text": instruction}] + message_content
            response = request_base64_gpt4v(message_content)

            cap_list.append(response)
            if i!= len(query["task_progress_metadata"]) - 1: c_steps += response + "; "
            else: c_steps += response + ". "
            print(response)
            
        response_dict = {
            "ANSWER": cap_list,
        }
        with open(f"{output_dir}/{qa_id}.json", "w") as f:
            json.dump(response_dict, f)

    except Exception:
        print(f"Error occurs when processing this query: {qa_id}", flush=True)
        traceback.print_exc()   

def cap_2_json(queries, extract_dir, cap_dir):
    for query in queries:
        qa_id = query["sample_id"]

        if not os.path.exists(f"{extract_dir}/{qa_id}.json"):
            print(f"No such file: {qa_id}.json")
        else:
            js_path = f"{extract_dir}/{qa_id}.json"
            js_file = open(js_path, "r")
            c_dict = json.load(js_file)
            str_list = c_dict["ANSWER"]
            caps = ''
            for s in str_list:
                s = s.replace('\n', '')
                if s.find("[") != -1 and s.find("]") != -1:
                    id1 = s.find("[")
                    id2 = s.rfind("]")
                    s = s[:id1] + s[id2 + 1:]
                s = s.replace('\"', '')
                assert s.find("[") == -1 and s.find("]") == -1, f"{qa_id}  []"
                assert s.rfind("\"") == -1, f"{qa_id}  answer"
                if s == str_list[-1]: caps += "#C C " + s + ".\n"
                else: caps += "#C C " + s + ".\n"
            if caps == '': caps = "none" 
            ans_dict = {
                "Video Name": qa_id,
                "Caption": caps,
            }
            if not os.path.exists(cap_dir):
                os.makedirs(cap_dir)
            ans_path = f"{cap_dir}/{qa_id}.json"
            ans_file = open(ans_path, "w")
            json.dump(ans_dict, ans_file, indent=4)

def action_seq(anno_path, video_dir, extract_dir, cap_dir, action_seg_frame_dir):
    queries = json.load(open(anno_path))
    
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    for i in tqdm(range(len(queries))):
        f_name = queries[i]['sample_id']+'.json'
        if os.path.exists(os.path.join(extract_dir, f_name)): continue
        llm_inference(
            queries[i], 
            extract_dir,
            action_seg_frame_dir,
            video_dir
        )
    
    cap_2_json(queries, extract_dir, cap_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--video_dir', type=str)
    parser.add_argument('--anno_path', type=str)
    parser.add_argument('--action_seq_dir', type=str, default='visual/cot')
    args = parser.parse_args()

    anno_path = args.anno_path
    video_dir = args.video_dir

    action_seq(
        anno_path,
        video_dir,
        os.path.join(args.action_seq_dir, 'extract'),
        os.path.join(args.action_seq_dir, 'cap'),
        os.path.join(args.action_seq_dir, 'action_seg_frame')
    )
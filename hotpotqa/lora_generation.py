import json
import os
import argparse
import concurrent.futures
import random
import time
import logging
logging.getLogger().setLevel(logging.ERROR)
from functools import partial

import torch
import numpy as np
from models.llama import LlamaInterface
from tasks.hotpotqa import HotpotQATask
from datetime import datetime
from langchain.agents.react.base import DocstoreExplorer
from langchain import Wikipedia
import re

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def prepare_prompt(question):
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:\n"

def prune_thought(prompt):
    if prompt.startswith("Thought:"):
        return prompt[len("Thought:"):].strip()
    return prompt

def run(task, idxs, gpts, do_sample=False, prev_traj=None):
    questions = [task[idx] for idx in idxs]
    rs, infos = {}, {}
    if prev_traj is not None:
        new_idxs = []
        for idx in idxs:
            if str(idx) not in prev_traj:
                continue
            if prev_traj[str(idx)]['em'] is False:
                new_idxs.append(idx)
            else:
                rs[idx] = True
            infos[idx] =  prev_traj[str(idx)]
        idxs = new_idxs
        questions = [task[idx] + '\n' + prev_traj[str(idx)]['traj'] + '\n' for idx in idxs]

    prompts = [prepare_prompt(q.rstrip()) for q in questions]


    docstore = Wikipedia()
    docstore = DocstoreExplorer(docstore)

    iteration = 0
    while iteration < 1:
        iteration += 1
        print(f"Iteration {iteration}")
        
        thought_action_pairs = gpts([prompt + f"Thought:" for prompt in prompts], stop=[f"\nObservation:", "Observation:", "###"], do_sample=do_sample)

        max_trials = 1
        for _ in range(max_trials):
            bad_ids = [i for i, pair in enumerate(thought_action_pairs) if "Action: " not in pair]
            if not bad_ids: break

            bad_prompts = [prompts[i] for i in bad_ids]
            bad_pairs = gpts([prompt + f"Thought:" for prompt in bad_prompts], stop=[f"\nObservation:", "Observation:", "###"], do_sample=do_sample)
            for i, pair in zip(bad_ids, bad_pairs):
                thought_action_pairs[i] = pair
                if _ == max_trials-1 and "Action: " not in pair:
                    thought_action_pairs[i] = "Thought: failed\nAction: finish[]"

        thoughts, actions, obs, bad_ids, done_ids = [], [], [], [], []
        for i, thought_action in enumerate(thought_action_pairs):
            try:
                if "\nAction: " in thought_action.strip():
                    thought, action = thought_action.strip().split("\nAction: ")[:2]
                elif "Action: " in thought_action.strip():
                    thought = ""
                    action = thought_action[len("Action: "):]
                else: 
                    thought = thought_action.split("\n")[0]
                    action = None
                    bad_ids.append(i)
            except:
                continue
            
            thoughts.append(thought)
            actions.append(action)
        
        threads = []
        results = {}
        for i, action in enumerate(actions):
            try:
                action_type, action_args = action.split('[')[:2]
                action_args = action_args[:-1]
            except:
                continue

            if "finish" in action_type.lower():
                r, info = task.evaluate(idxs[i], action_args)
                done_ids.append(i)
            else:
                if action_type.lower() == 'search':
                    try:
                        obs = format_step(docstore.search(action_args))
                    except Exception as e:
                        print(e)
                        obs = 'Could not find that page, please try again.'

                elif action_type.lower() == 'lookup':
                    try:
                        obs = format_step(docstore.lookup(action_args))
                    except ValueError:
                        obs = f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
                else:
                    obs = 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'            

            
            if obs == "Observation: None": 
                print(f"Warning: action {action} has observation {obs}")

            prompts[i] += f"Thought: {prune_thought(thoughts[i])}\nAction: {action}\nObservation: {obs}\n"
                
            if "finish" in action_type.lower():
                traj = prompts[i]
                rs[idxs[i]] = r
                if prev_traj is not None:
                    info.update({'traj_reflect': traj, 'traj_by_line_reflect': traj.split('\n')})
                    infos[idxs[i]].update(info)
                else:
                    info.update({'traj': traj, 'traj_by_line': traj.split('\n')})
                    infos[idxs[i]] = info
                
        
        prompts = [prompts[i] for i in range(len(prompts)) if i not in done_ids]
        idxs = [idxs[i] for i in range(len(idxs)) if i not in done_ids]
        if not prompts:
            break
        
    return rs, infos


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, default='hotpotqa')
    args.add_argument('--task_split', type=str, default='train')
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=100)

    args.add_argument('--seed', type=int, default=-1)
    args.add_argument('--add_lora', action='store_true')
    args.add_argument('--random', action='store_true')
    args.add_argument('--do_sample', action='store_true')

    args.add_argument('--modelpath', type=str, default='')
    args.add_argument('--peftpath', type=str, default='')
    args.add_argument('--prev_traj', type=str, default='')

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    task = HotpotQATask(args.task_split)
    
    if args.seed >= 0:
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # For some operations in Python which are hash-based (like dictionaries), set:
        os.environ["PYTHONHASHSEED"] = str(seed)

    pathname = args.peftpath.replace('/', '_') if args.add_lora else args.modelpath.replace('/', '_')
    modelname = f"llama_{pathname}"
    time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    outfilename = f"trajs/{args.task}_{args.task_split}_{args.task_start_index}_{args.task_end_index}_{modelname}_{args.temperature}_{time_str}.json"
    print(outfilename)
    
    idxs_all = list(range(len(task)))
    if args.random:
        random.Random(233).shuffle(idxs_all)
    idxs = idxs_all[args.task_start_index:args.task_end_index]

    prev_traj = json.load(open(args.prev_traj)) if args.prev_traj != '' else None

    llama = LlamaInterface(args.modelpath, args.peftpath, args.add_lora)
    model = llama.generate_responses_from_llama
    
    rs, infos = run(task, idxs, model, \
                    do_sample=args.do_sample,
                    prev_traj=prev_traj)

    with open(outfilename, "w") as fout:
        json.dump(infos, fout, indent=2)
    em = sum(rs.values()) / len(idxs)
    print("em", em)

import json
import torch
import os
import numpy as np
import argparse
import concurrent.futures
import random
import time
import logging
logging.getLogger().setLevel(logging.ERROR)
from functools import partial
from datetime import datetime

from tasks.hotpotqa import HotpotQATask
from models.llama import LlamaInterface
from langchain.agents.react.base import DocstoreExplorer
from langchain import Wikipedia
import re

def get_fewshot_prompt(promptpath, task=None):
    with open(f"./prompts/{promptpath}.json", "r") as fin:
        prompt = json.load(fin)
    return prompt

def prune_thought(prompt, iteration=None):
    if prompt.startswith("Thought:"):
        return prompt[len("Thought:"):].strip()
    if iteration is not None and prompt.startswith(f"Thought {iteration}:"):
        return prompt[len(f"Thought {iteration}:"):].strip()
    return prompt

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def run(task, idxs, gpts, promptpath='', prev_traj=None, do_sample=False):
    fewshot_prompt = get_fewshot_prompt(promptpath, task)

    docstore = Wikipedia()
    docstore = DocstoreExplorer(docstore)


    rs, infos = {}, {}

    if prev_traj is not None:
        new_idxs = []
        for idx in idxs:
            if str(idx) not in prev_traj:
                continue
            infos[idx] =  prev_traj[str(idx)]
            if prev_traj[str(idx)]['em'] is False:
                new_idxs.append(idx)
            else:
                rs[idx] = True

        idxs = new_idxs
        questions = [task[idx] + '\nPrevious Trial:\n' + prev_traj[str(idx)]['trajs']  for idx in idxs]
    else:
        questions = [task[idx] for idx in idxs]
    prompts = [fewshot_prompt + [{'role': 'user', 'content': 'Question: ' + question}] for question in questions]

    iteration = 0
    cur_prompts = ['' for _ in prompts]

    while iteration < 6:
        iteration += 1
        print(f"Iteration {iteration}")
        if iteration == 1 and prev_traj is not None:
            reflects = gpts(prompts, start_prompts=[x + f'Reflection: I should have' for x in cur_prompts], stop=[f'Thought {iteration}:'], do_sample=do_sample)
            cur_prompts = [x.strip()+'\n' for x in reflects]

        thought_action_pairs = gpts(prompts, start_prompts=[x + f'Thought {iteration}:' for x in cur_prompts], stop=[f'Observation {iteration}:'], do_sample=do_sample)
        thought_action_pairs = [f'Thought {iteration}:'+(re.sub(r'\n+', '\n', x).split(f'Thought {iteration}:')[-1].strip()).replace(f'Action {iteration+1}:', f'Action {iteration}:') for x in thought_action_pairs]
        max_trials = 1
        for _ in range(max_trials):
            bad_ids = [i for i, pair in enumerate(thought_action_pairs) if f"Action {iteration}: " not in pair]
            if not bad_ids: break

            bad_prompts = [prompts[i] for i in bad_ids]
            bad_pairs = gpts(bad_prompts, start_prompts=[cur_prompts[i] + f'Thought {iteration}:' for i in bad_ids], do_sample=True, stop=[f'Observation {iteration}:'])
            bad_pairs = [f'Thought {iteration}:'+re.sub(r'\n+', '\n', x).split(f'Thought {iteration}:')[-1].strip() for x in bad_pairs]
            for i, pair in zip(bad_ids, bad_pairs):
                thought_action_pairs[i] = pair
                if _ == max_trials-1 and f"Action {iteration}: " not in pair:
                    thought_action_pairs[i] = f"Thought {iteration}: failed\nAction {iteration}: finish[]"

        ## execute actions
        thoughts, actions, obs, bad_ids, done_ids = [], [], [], [], []
        for i, thought_action in enumerate(thought_action_pairs):
            try:
                if f"\nAction {iteration}: " in thought_action.strip():
                    thought, action = thought_action.strip().split(f"\nAction {iteration}: ")[:2]
                elif f"Action {iteration}: " in thought_action.strip():
                    thought = ""
                    action = thought_action[len(f"Action {iteration}: "):]
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

        cur_prompts = [x + thought_action_pairs[i] for i, x in enumerate(cur_prompts)]
        results = {}
        for i, action in enumerate(actions):
            try:
                action_type, action_args = action.split('[')[:2]
                action_args = action_args.split(']')[0]
            except:
                continue
                
            if action_type.lower() == 'finish':
                r, info = task.evaluate(idxs[i], action_args)
                info['trajs'] = cur_prompts[i]
                info['question'] = task[idxs[i]]
                rs[idxs[i]] = r
                if prev_traj is not None:
                    info['previous_trajs'] = prev_traj[str(idxs[i])]['trajs']
                infos[idxs[i]] = info                
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
                cur_prompts[i] += f'\nObservation {iteration}: {obs}\n'
                
    
        cur_prompts = [cur_prompts[i] for i in range(len(cur_prompts)) if i not in done_ids]
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

    args.add_argument('--add_lora', action='store_true')
    args.add_argument('--random', action='store_true')

    args.add_argument('--modelpath', type=str, default='')
    args.add_argument('--peftpath', type=str, default='')
    args.add_argument('--prev_traj', type=str, default='')
    args.add_argument('--promptpath', type=str, default='')
    args.add_argument('--do_sample', action='store_true')
    args.add_argument('--seed', type=int, default=-1)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    task = HotpotQATask(args.task_split)
    
    modelname = 'llama'
    pathname = args.peftpath.replace('/', '_') if args.add_lora else args.modelpath.replace('/', '_')
    modelname += f"_{pathname}"
    time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    outfilename = f"trajs/{args.task}_{args.task_split}_{args.task_start_index}_{args.task_end_index}_{modelname}_{args.temperature}_{time_str}.json"
    print(outfilename)
    
    idxs_all = list(range(len(task)))
    if args.random:
        random.Random(233).shuffle(idxs_all)

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

    idxs = idxs_all[args.task_start_index:args.task_end_index]

    prev_traj = json.load(open(args.prev_traj)) if args.prev_traj != '' else None
    
    llama = LlamaInterface(args.modelpath, args.peftpath, args.add_lora)
    model = llama.generate_responses_from_llama
    
    rs, infos = run(task, idxs, model, \
                    promptpath=args.promptpath,
                    do_sample=args.do_sample,
                    prev_traj=prev_traj)

    with open(outfilename, "w") as fout:
        json.dump(infos, fout, indent=2)
    em = sum(rs.values()) / len(idxs)
    print("em", em)

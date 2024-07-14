import json
import torch
from utils import write_jsonl
import os
import numpy as np
import argparse
import concurrent.futures
import random
import time
import logging
logging.getLogger().setLevel(logging.ERROR)
from functools import partial

from models.llama import LlamaInterface

from datetime import datetime
from executors import executor_factory
from utils import read_jsonl, read_jsonl_gz

def prepare_prompt(x):
    def extract_task(p):
        return p.split('"""')[1].strip()
    prompt = extract_task(x['prompt'])
    tests = '\n'.join(x['visible_tests'])
    prompt = f'You are an expert Python programmer, and here is your task: {prompt}\nYour code should pass these tests:\n\n{tests}\nYour code should start with a [PYTHON] tag and end with a [/PYTHON] tag.'
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def run(task, gpts, evaluate=True, outfilename=None, do_sample=False):
    prompts = [prepare_prompt(x) for x in dataset]

    rs, infos = {}, {}

    outs = gpts(prompts, do_sample=do_sample)

    if evaluate:
        acc = 0
        for i, ans in enumerate(outs):
            exe = executor_factory('py', is_leet=False)
            tests_i = dataset[i]['visible_tests']
            if '[PYTHON]' in ans:
                ans = ans.split('[PYTHON]')[1]
            if '[/PYTHON]' in ans:
                ans = ans.split('[/PYTHON]')[0]
            is_passing, feedback, _ = exe.execute(ans, tests_i, timeout=20)
            dataset[i]['is_solved'] = is_passing
            dataset[i]['implementation'] = ans
            acc += int(is_passing)

            write_jsonl(outfilename, [dataset[i]], append=True)
        print(f'acc {acc/len(outs)}')


    
    return


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, default='llama')
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, default='mbpp-clean-test')
    args.add_argument('--task_split', type=str, default='train')
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=100)

    args.add_argument('--evaluate', action='store_true')
    args.add_argument('--add_lora', action='store_true')
    args.add_argument('--random', action='store_true')

    args.add_argument('--modelpath', type=str, default='')
    args.add_argument('--peftpath', type=str, default='')
    args.add_argument('--do_sample', action='store_true')
    args.add_argument('--seed', type=int, default=-1)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    #task = get_task(args.task, args.task_split)
    
    modelname = args.backend
    pathname = args.peftpath.replace('/', '_') if args.add_lora else args.modelpath.replace('/', '_')
    modelname += f"_{pathname}"
    time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    outfilename = f"trajs/{args.task}__{modelname}_{args.temperature}_{time_str}.jsonl"
    print(outfilename)
    
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


    

    dataset = read_jsonl(f'benchmarks/{args.task}.jsonl')

    print(args.modelpath, args.peftpath, args.add_lora)
    llama = LlamaInterface(args.modelpath, args.peftpath, args.add_lora)
    model = llama.generate_responses_from_llama

    run(dataset, model, outfilename=outfilename, evaluate=args.evaluate, \
                    do_sample=args.do_sample)

import os

import fire
import pandas as pd
import torch
import tqdm
import wandb
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

wandb.init(mode="disabled")


def generate(base_model="codellama/CodeLlama-13b-Python-hf", peft_path=None, bsz: int = 8,
             prompt_file="/home/xueqing.wu/codes/viper/results/gqa/testdev-1000subset/results_106.csv",
             output_fname="out.csv"):
    output_fname = os.path.join(peft_path, output_fname)
    # assert not os.path.exists(output_fname)

    # Load model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model, load_in_8bit=True, device_map="auto",
    )
    assert peft_path is not None
    model = PeftModel.from_pretrained(
        model, peft_path, torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Get data prompt
    data = pd.read_csv(prompt_file)
    prompts = ['# ' + q.splitlines()[0].replace('Given an image: ', '') + '\n' for q in data['query']]

    generations = []
    import pdb
    pdb.set_trace()
    for i in tqdm.trange(0, len(prompts), bsz):
        batch = tokenizer(prompts[i:][:bsz], return_tensors='pt', padding=True, truncation=True,
                          max_length=1024 - 256)
        gen_ids = model.generate(
            input_ids=batch.input_ids.cuda(), attention_mask=batch.attention_mask.cuda(),
            do_sample=False, max_new_tokens=256,
        )
        gen_ids = gen_ids[:, batch.input_ids.shape[-1]:]
        generations += [tokenizer.decode(gen_id, skip_special_tokens=True) for gen_id in gen_ids]

    data['code'] = generations
    data.to_csv(output_fname)


if __name__ == "__main__":
    fire.Fire(generate)

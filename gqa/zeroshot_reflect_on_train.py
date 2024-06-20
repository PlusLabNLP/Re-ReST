import argparse
import os

import pandas as pd
import torch
import tqdm
from torchvision import transforms
from transformers import LlamaForCausalLM, CodeLlamaTokenizer

from datasets.gqa import GQADataset

B_INST, E_INST = "[INST]", "[/INST]"


@torch.no_grad()
def main(args):
    dataset = GQADataset(split='train', testing=False, max_samples=2500, balanced=True,
                         data_path=args.gqa_data_path,
                         image_transforms=transforms.Compose([transforms.ToTensor()]))
    data = pd.read_csv(args.input)

    with open(os.path.join(os.path.dirname(__file__), 'zeroshot_reflect.prompt')) as f:
        prompt = f.read().strip()

    model_id = 'codellama/CodeLlama-13b-Instruct-hf'

    tokenizer = CodeLlamaTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    def reflect_prompt(data_row, code):
        code = code.strip('\n')
        prompt_ = prompt.format(
            QUESTION=data_row['query'].replace("Given an image: ", "").splitlines()[0],
            ANSWER=repr(data_row['answer']), RESULT=repr(data_row['result']), CODE=code,
        )
        prompt_ = B_INST + prompt_.strip() + E_INST
        return prompt_

    def reflect_generate(prompts):
        batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        generated_ids = model.generate(
            input_ids.to("cuda"), attention_mask=attention_mask.to("cuda"), max_new_tokens=1024, do_sample=False
        )
        ret = []
        for i in range(len(prompts)):
            response = tokenizer.decode(generated_ids[i, input_ids.shape[1]:], skip_special_tokens=True).strip()
            try:
                code = response.split("```python\ndef execute_command(image) -> str:")[1].split("```")[0]
            except:
                code = ''
            ret.append((response, code))
        return ret

    responses_all = []
    codes_all = []
    for i in tqdm.trange(len(data)):
        code = data['code']
        acc = dataset.accuracy([data['result'][i], ], [data['answer'][i], ], [[], ], [dataset.input_type, ])
        if acc == 1:
            responses_all.append('')
            codes_all.append(code)
        else:
            response, code = reflect_generate([reflect_prompt(data.iloc[i], code), ])[0]
            responses_all.append(response)
            codes_all.append(code)

    data['code'] = codes_all
    data['responses'] = responses_all
    data.to_csv(args.input + ".zero-shot-reflect.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--gqa-data-path', required=True)
    args = parser.parse_args()

    main(args)

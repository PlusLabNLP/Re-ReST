## Setup
```
conda create -n rerest python=3.9
conda activate rerest
pip install -r requirements.txt
```

## Zero-Shot Inference
Agent:
```
python zs_generation.py \
    --random \
    --task_split dev \
    --task_end_index 500 \
    --modelpath meta-llama/Meta-Llama-3-8B-Instruct \
    --promptpath react_hotpotqa
```

Reflector:
```
python zs_generation.py \
    --random \
    --task_split dev \
    --task_end_index 500 \
    --modelpath meta-llama/Meta-Llama-3-8B-Instruct \
    --promptpath reflect_react_hotpotqa \
    --prev_traj PREV_TRAJ_JSON_FILE
```

## Training
Agent/Reflector:
```
python lora_finetune.py \
     --base_model meta-llama/Meta-Llama-3-8B-Instruct \
     --data_path TRAIN_DATA_JSON_FILE \
     --micro_batch_size 2 \
     --num_epochs 3 \
     --batch_size 128 \
     --train_on_inputs False \
     --val_set_size 0.01 \
     --output_dir OUTPUT_DIR
```

## Inference After Training
Agent:
```
python lora_generation.py \
    --random \
    --task_split dev \
    --modelpath meta-llama/Meta-Llama-3-8B-Instruct \
    --task_end_index 500 \
    --add_lora \
    --peftpath PEFT_MODEL_PATH
```

Reflector:
```
python lora_generation.py \
    --random \
    --task_split dev \
    --modelpath meta-llama/Meta-Llama-3-8B-Instruct \
    --task_end_index 500 \
    --add_lora \
    --peftpath PEFT_MODEL_PATH \
    --prev_traj PREV_TRAJ_JSON_FILE
```

## Model Zoo
| Model | Weight |
| ---------  | ------ |
| Llama-3-8B Agent | [link](https://huggingface.co/zdou0830/Re-ReST/tree/main/agent) |
| Llama-3-8B Reflector | [link](https://huggingface.co/zdou0830/Re-ReST/tree/main/reflector) |



## References
Our code is modified from [anchen1011/FireAct](https://raw.githubusercontent.com/anchen1011/FireAct) under [the MIT license](https://github.com/PlusLabNLP/Re-ReST/blob/main/hotpotqa/LICENSE_FireAct), whose generation code is based on [ysymyth/ReAct](https://github.com/ysymyth/ReAct) and LoRA training code is based on [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora). Our reflection code is based on [noahshinn/reflexion](https://github.com/noahshinn/reflexion).


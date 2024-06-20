## Setup

Our experiments on GQA is based on [ViperGPT: Visual Inference via Python Execution for Reasoning](https://viper.cs.columbia.edu/).
Please setup the environment, dataset and dependencies according to [their guidelines](https://github.com/cvlab-columbia/viper/).

## Zero-Shot Inference
Under `viper/` directory, zero-shot agent:
```
CONFIG_NAMES=your_config_name python main_batch.py
```
And the code and execution results will be stored into a csv file.

Reflector:
```
python zeroshot_reflect_on_train.py YOUR_CSV_FILE_FROM_VIPER --gqa-data-path GQA_DATA_PATH
```

## Training
Agent/Reflector:
```
python ./finetune.py \
     --base_model codellama/CodeLlama-13b-Python-hf \
     --data_path TRAIN_JSON_FILE \
     --micro_batch_size 4 \
     --num_epochs 6 \
     --output_dir PEFT_PATH \
     --val_set_size 0.01 \
     --cutoff_len 4096 \
     --train_on_inputs False \
     --learning_rate 1e-04
```

## Inference After Training
Agent:
```
python generate.py \
    --peft_path PEFT_PATH \
    --output_fname OUTPUT_FNAME \
    --prompt_file YOUR_CSV_FILE_FROM_VIPER
```

Then you can execute the code stored in `OUTPUT_FNAME` using `viper/` to obtain its accuracy.

## References
Our code is modified from [anchen1011/FireAct](https://raw.githubusercontent.com/anchen1011/FireAct) under [the MIT license](https://github.com/PlusLabNLP/Re-ReST/blob/main/hotpotqa/LICENSE_FireAct), whose generation code is based on [ysymyth/ReAct](https://github.com/ysymyth/ReAct) and LoRA training code is based on [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora).

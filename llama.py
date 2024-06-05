from datetime import datetime
import os
import sys
 
import torch
 
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from transformers import (AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq)
 
from datasets import load_dataset
 
train_dataset = load_dataset('json', data_files='train_data.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='val_data.jsonl', split='train')
 
# 读取模型
base_model = 'hfl/chinese-llama-plus-lora-7b'
 
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
 
tokenizer = AutoTokenizer.from_pretrained(base_model)
import os
import re
import xml.etree.ElementTree as ET
import torch
import pandas as pd
import wandb 
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth.chat_templates import train_on_responses_only # <-- 引入 Unsloth 的专用魔法函数

# --- 0. 初始化 W&B (Loss 追踪) ---
# 如果你不想用 wandb，可以注释掉这行，并把下面 TrainingArguments 里的 report_to="wandb" 删掉
wandb.init(project="qwen-svg-generation", name="qwen-1.5b-lora-completion-only")

# --- 1. 配置参数 ---
CONFIG = {
    "model_name": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    "max_seq_length": 2048,
    "lora_r": 16,
    "learning_rate": 2e-4,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2, 
    "gradient_accumulation_steps": 8,
    "output_dir": "./qwen2.5_svg_lora_v2", 
    "train_path": "./data/train_cleaned_old.csv",
    "test_size": 0.1, 
    "eval_steps": 50, 
    "logging_steps": 10, 
}

# --- 2. 数据加载与校验 ---
def is_valid_svg(svg_text):
    if not svg_text or not str(svg_text).strip().startswith("<svg"):
        return False
    try:
        ET.fromstring(svg_text)
        return True
    except ET.ParseError:
        return False

df = pd.read_csv(CONFIG["train_path"])
df = df[df['svg'].apply(is_valid_svg)]
full_ds = Dataset.from_pandas(df)

# --- 3. 格式化 Prompt 与 切分数据集 ---
SYSTEM_PROMPT = "You generate compact, valid SVG markup. Return only SVG code."

def format_sft_text(example):
    return {"text": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                    f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{example['svg']}<|im_end|>"}

full_ds = full_ds.map(format_sft_text)
split_ds = full_ds.train_test_split(test_size=CONFIG["test_size"], seed=42)
train_ds = split_ds["train"]
eval_ds = split_ds["test"]

print(f"训练集大小: {len(train_ds)}, 测试集大小: {len(eval_ds)}")

# --- 4. 加载 Unsloth 模型 ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CONFIG["model_name"],
    max_seq_length=CONFIG["max_seq_length"],
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=CONFIG["lora_r"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# --- 5. 初始化 Trainer ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds, 
    dataset_text_field="text",
    max_seq_length=CONFIG["max_seq_length"],
    # 注意这里已经去掉了旧的 data_collator 参数
    args=TrainingArguments(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        num_train_epochs=CONFIG["num_train_epochs"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        
        logging_strategy="steps",
        logging_steps=CONFIG["logging_steps"], 
        eval_strategy="steps",                 
        eval_steps=CONFIG["eval_steps"],       
        save_strategy="steps",                 
        save_steps=CONFIG["eval_steps"],       
        report_to="wandb",                     
        load_best_model_at_end=True,           
        
        optim="paged_adamw_8bit",
    ),
)

# --- 6. 施加 Unsloth 魔法：只对 SVG 结果计算 Loss ---
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# --- 7. 开始训练与保存 ---
trainer.train()

# 结束 wandb 记录
if wandb.run is not None:
    wandb.finish() 

# 保存微调后的 LoRA 权重
model.save_pretrained(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])

import os
import re
import xml.etree.ElementTree as ET
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer


# --- 配置参数 ---
CONFIG = {
    "model_name": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit", # 使用 Starter 推荐的模型
    "max_seq_length": 2048,
    "lora_r": 16,
    "learning_rate": 2e-4,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "output_dir": "./qwen2.5_svg_lora_with_cleaned_data",
    "train_path": "./data/train_cleaned_old.csv", # 你本地的数据路径
}

# --- 1. 数据加载与 Starter 风格的校验 ---
def is_valid_svg(svg_text):
    """来自 Starter 的逻辑：确保生成的 SVG 是合法的 XML"""
    if not svg_text or not svg_text.strip().startswith("<svg"):
        return False
    try:
        ET.fromstring(svg_text)
        return True
    except ET.ParseError:
        return False

# 假设你使用本地 CSV
import pandas as pd
df = pd.read_csv(CONFIG["train_path"])
# 过滤掉无效数据
df = df[df['svg'].apply(is_valid_svg)]
train_ds = Dataset.from_pandas(df)

# --- 2. 格式化 Prompt ---
SYSTEM_PROMPT = "You generate compact, valid SVG markup. Return only SVG code."

def format_sft_text(example):
    return {"text": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                    f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{example['svg']}<|im_end|>"}

train_ds = train_ds.map(format_sft_text)

# --- 3. 加载 Unsloth 模型 ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CONFIG["model_name"],
    max_seq_length=CONFIG["max_seq_length"],
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=CONFIG["lora_r"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", # 节省显存的关键
    random_state=42,
)

# --- 4. 训练 ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    dataset_text_field="text",
    max_seq_length=CONFIG["max_seq_length"],
    args=TrainingArguments(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        num_train_epochs=CONFIG["num_train_epochs"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="paged_adamw_8bit",
    ),
)

trainer.train()
model.save_pretrained(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])

import os
import warnings
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model

# Paths & base settings
ROOT = Path(__file__).resolve().parents[1]  
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")  
DATA_PATH = os.getenv("DATA_PATH", str(ROOT / "train_data" / "chat_train.jsonl"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(ROOT / "out-qwen25-3b-lora"))

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Make sure torch shows cuda available: True")

bf16_ok = torch.cuda.get_device_capability(0)[0] >= 8
dtype = torch.bfloat16 if bf16_ok else torch.float16

warnings.filterwarnings("ignore", message=".*pin_memory.*")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Base model + LoRA
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=dtype,         
    device_map="auto",
)

model.gradient_checkpointing_enable()
model.config.use_cache = False
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

# LoRA
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_cfg)

# Dataset -> prompt formatting
ds = load_dataset("json", data_files=DATA_PATH)["train"]

def format_row(ex):
    instr = (ex.get("instruction") or "").strip()
    inp   = (ex.get("input") or "").strip()
    out   = (ex.get("output") or "").strip()
    user = f"{instr}\n\n{inp}" if inp else instr
    text = f"<s>[INST] {user} [/INST] {out}</s>"
    return {"text": text}

ds = ds.map(format_row, remove_columns=ds.column_names)

# Tokenization
MAX_LEN = int(os.getenv("MAX_LEN", "768")) 
def tokenize_fn(ex):
    enc = tokenizer(
        ex["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors=None,
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

ds_tok = ds.map(tokenize_fn, batched=False, remove_columns=["text"])

# TrainingArguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,        
    gradient_accumulation_steps=16,       
    learning_rate=2e-4,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    bf16=bf16_ok,
    fp16=not bf16_ok,
    optim="adamw_torch",                  
    report_to="none",
    dataloader_num_workers=0,             
    dataloader_pin_memory=False,          
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok,
    data_collator=default_data_collator,
)

# Train & Save
trainer.train()

(Path(OUTPUT_DIR) / "adapter").mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(Path(OUTPUT_DIR) / "adapter"))
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved LoRA adapter to:", str(Path(OUTPUT_DIR) / "adapter"))

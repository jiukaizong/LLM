import os
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from services.data_loader import build_data_context

ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", str(ROOT / "out-qwen25-3b-lora" / "adapter"))

bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
DTYPE = torch.bfloat16 if bf16_ok else torch.float16
DEVICE_MAP = "auto"

_tokenizer = None
_model = None


def _ensure_loaded():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            use_fast=True,
            trust_remote_code=True,
        )
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

    if _model is None:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=DTYPE,
            device_map=DEVICE_MAP,
            trust_remote_code=True,
        )
        base.eval()

        if ADAPTER_DIR and Path(ADAPTER_DIR).exists():
            _model = PeftModel.from_pretrained(base, ADAPTER_DIR)
            _model.eval()
        else:
            _model = base


# ---------------------------
# 系统提示词
# ---------------------------
SYSTEM_PROMPT = (
    "You are a rigorous risk & performance analyst. "
    "Always use ONLY the facts in [DATA CONTEXT] built from the CSVs. "
    "When the user asks, infer and reason from these facts, and provide concise, actionable answers "
    "with bullet points and concrete metrics if available. "
    "If the data context is insufficient for a precise answer (e.g., missing a time dimension), "
    "respond with what additional fields or filters are needed."
)


def _build_messages(user_text: str, context: str) -> list[dict]:
    sys_content = f"{SYSTEM_PROMPT}\n\n[DATA CONTEXT]\n{context}".strip()
    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_text},
    ]


def _post_clean(text: str) -> str:
    text = re.sub(r"<s>|</s>|<<\s*/?\s*SYS\s*>>|\[/?INST\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<<.*?>>", "", text) 
    return text.strip()


@torch.inference_mode()
def chat_once(
    user_text: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_data: bool = True,
    extra_context: str | None = None,
) -> str:
    _ensure_loaded()

    context = build_data_context(user_text, extra_context=extra_context) if use_data else ""
    messages = _build_messages(user_text, context)

    inputs = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  
        return_tensors="pt",
    ).to(_model.device)

    gen_ids = _model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.05,
        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=_tokenizer.eos_token_id,
        use_cache=True,
    )

    new_tokens = gen_ids[0][inputs.shape[-1]:]
    output = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _post_clean(output)

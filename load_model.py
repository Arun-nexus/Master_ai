import logging
import os
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
model_id = "Qwen/Qwen3-4B"
base = Path(__file__).parent / "model"
base.mkdir(parents=True, exist_ok=True)
quant = base / "quant_model"
full = base / "full_model"

QUANT = os.getenv("QUANT", "4bit").lower()

SYSTEM_PROMPT = (
    "You are the assistant of the Accure Medical ERP. Use the available tools to "
    "fetch or change ERP data. Never invent data: if you need information, call a tool. "
    "Call a tool only with the arguments its schema asks for. After you get the tool "
    "result, answer briefly and clearly in the same language the user writes in."
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

device_map = {"": 0}

quant_model_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)


def load_model():
    if QUANT == "none":
        if not (full / "config.json").exists():
            snapshot_download(repo_id=model_id, local_dir=full)
        tok = AutoTokenizer.from_pretrained(full)
        mod = AutoModelForCausalLM.from_pretrained(
            full, torch_dtype=torch.float16, device_map=device_map,
            low_cpu_mem_usage=True, attn_implementation="sdpa",
        )
        return mod.eval(), tok

    if (quant / "config.json").exists():
        tok = AutoTokenizer.from_pretrained(quant)
        mod = AutoModelForCausalLM.from_pretrained(quant, device_map=device_map, attn_implementation="sdpa")
        return mod.eval(), tok

    if not (full / "config.json").exists():
        snapshot_download(repo_id=model_id, local_dir=full)

    tok = AutoTokenizer.from_pretrained(full)
    mod = AutoModelForCausalLM.from_pretrained(
        full,
        quantization_config=quant_model_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    mod.save_pretrained(quant)
    tok.save_pretrained(quant)
    return mod.eval(), tok


if __name__ == "__main__":
    model, tokenizer = load_model()
    print("Model loaded successfully.", next(model.parameters()).device)
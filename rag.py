import logging
import os
import torch

log = logging.getLogger("rag")

# LLM_BACKEND=hf   -> transformers (default, Windows/Linux, 8GB par chalta hai)
# LLM_BACKEND=vllm -> vLLM + AWQ model (Linux/WSL2/Docker), sabse tez
BACKEND = os.getenv("LLM_BACKEND", "hf").lower()

# HF backend tuning
TOKEN_BUDGET = int(os.getenv("GEN_TOKEN_BUDGET", "8000"))   # ek batch mein (input+output) tokens. 8GB: 8000, 24GB: 40000+
MAX_BS = int(os.getenv("GEN_MAX_BS", "32"))

if BACKEND == "vllm":
    from vllm import LLM, SamplingParams

    _llm = LLM(
        model=os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B-AWQ"),
        gpu_memory_utilization=float(os.getenv("VLLM_GPU_UTIL", "0.55")),  # baaki memory bge-m3 ke liye
        max_model_len=int(os.getenv("VLLM_MAX_LEN", "6144")),
        enable_prefix_caching=True,
    )
    tokenizer = _llm.get_tokenizer()
    model = None
else:
    from load_model import load_model

    model, tokenizer = load_model()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _chat(prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def generate(prompt, max_new_tokens=300):
    return generate_batch([prompt], max_new_tokens)[0]


@torch.inference_mode()
def chat_generate(messages, tools=None, max_new_tokens=512):
    """Multi-turn chat (optionally with tool schemas) on the SAME model the tender pipeline uses.
    Caller must hold extract.gpu_lock."""
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if BACKEND == "vllm":
        outs = _llm.generate([prompt], SamplingParams(temperature=0, max_tokens=max_new_tokens), use_tqdm=False)
        return outs[0].outputs[0].text.strip()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def generate_batch(prompts, max_new_tokens=300):
    if not prompts:
        return []
    texts = [_chat(p) for p in prompts]
    if BACKEND == "vllm":
        outs = _llm.generate(texts, SamplingParams(temperature=0, max_tokens=max_new_tokens), use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]
    return _hf_generate(texts, max_new_tokens)


# ---------------- HF backend ----------------

def _make_batches(lengths, max_new_tokens):
    """Lambe prompts pehle (OOM jaldi pakda jaye), phir token budget ke hisaab se batch."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batches, cur = [], []
    for i in order:
        longest = lengths[cur[0]] if cur else lengths[i]
        if cur and ((len(cur) + 1) * (longest + max_new_tokens) > TOKEN_BUDGET or len(cur) >= MAX_BS):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


@torch.inference_mode()
def _run(texts, idx, max_new_tokens):
    try:
        inputs = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        dec = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return dict(zip(idx, (d.strip() for d in dec)))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(idx) == 1:
            raise
        log.warning("OOM at batch size %d, splitting", len(idx))
        h = len(idx) // 2
        res = _run(texts, idx[:h], max_new_tokens)
        res.update(_run(texts, idx[h:], max_new_tokens))
        return res


def _hf_generate(texts, max_new_tokens):
    lengths = [len(t) for t in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    results = [""] * len(texts)
    batches = _make_batches(lengths, max_new_tokens)
    log.info("generating %d prompts in %d batches", len(texts), len(batches))
    for idx in batches:
        for i, d in _run(texts, idx, max_new_tokens).items():
            results[i] = d
    return results
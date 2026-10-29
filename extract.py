import threading
from rag import generate_batch

gpu_lock = threading.Lock()

SPEC_PROMPT = (
    "Extract only technical specifications, features, ratings, dimensions, "
    "certifications and standards from the text. Keep exact numbers and units. "
    "One per line as 'Parameter: value'. Ignore marketing, safety warnings, "
    "contact info and legal text. If there is nothing, reply NONE.\n\nText:\n{text}"
)

REQ_PROMPT = (
    "Extract every individual technical requirement or specification from this "
    "tender text. One per line starting with '- '. Keep exact numbers and units. "
    "Ignore commercial terms, eligibility, payment and legal clauses. "
    "If there is nothing, reply NONE.\n\nText:\n{text}"
)


def windows(text, size=3000):
    out, cur = [], ""
    for line in text.splitlines():
        if len(cur) + len(line) > size and cur:
            out.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        out.append(cur)
    return out


def run_batch(prompts, tokens):
    if not prompts:
        return []
    with gpu_lock:
        return generate_batch(prompts, tokens)


def is_none(text):
    return not text or text.strip().upper().startswith("NONE")


def spec_summaries(chunks):
    chunks = list(dict.fromkeys(chunks))
    outs = run_batch([SPEC_PROMPT.format(text=c) for c in chunks], 300)
    return list(dict.fromkeys(o.strip() for o in outs if not is_none(o)))


def extract_requirements(text, limit):
    outs = run_batch([REQ_PROMPT.format(text=w) for w in windows(text, 1500)], 300)
    seen, reqs = set(), []
    for out in outs:
        if is_none(out):
            continue
        for line in out.splitlines():
            line = line.strip().lstrip("-•* ").strip()
            if len(line) < 8 or line.lower() in seen:
                continue
            seen.add(line.lower())
            reqs.append(line)
            if len(reqs) >= limit:
                return reqs
    return reqs
"""
Random sample of raw completions for the write-up.

Neel asks for randomly selected qualitative examples, not cherry-picked, so the
sampling is seeded and the indices are printed. Anyone can rerun with the same
seed and get the same examples.

Your main run saved vectors but not text, so this regenerates a handful. 4 per
group is ~12 generations, a couple of minutes.

python samples.py

Writes samples.txt (and samples.json for reuse).
"""

import csv, io, json, random, re, urllib.request
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"       # the model with real refusal behaviour
ENABLE_THINKING = "Qwen3" in MODEL
PER_GROUP = 4
MAX_NEW_TOKENS = 400
SAMPLE_SEED = 12345             # different from the prompt-selection seed

HARMFUL_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
SAFE_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
JAILBREAK_URL = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_12_25.csv"

IMPERATIVE = re.compile(
    r"^(write|develop|create|give|explain|describe|design|generate|provide|"
    r"suggest|compose|outline|make|build|list|construct|produce|draft)\b", re.I)


def get(url):
    return urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")


def load_prompts():
    rng = random.Random(0)      # identical to the main run
    harmful = [r["goal"].strip() for r in csv.DictReader(io.StringIO(get(HARMFUL_URL)))]
    harmful = rng.sample(harmful, 200)
    safe = [d["instruction"].strip() for d in json.loads(get(SAFE_URL))
            if not d.get("input", "").strip()
            and IMPERATIVE.match(d["instruction"].strip())]
    safe = rng.sample(safe, 200)
    rows = list(csv.DictReader(io.StringIO(get(JAILBREAK_URL))))
    templates = [r["prompt"].strip() for r in rows if len(r.get("prompt", "")) > 200]
    jailbreak = [templates[rng.randrange(len(templates))] + "\n\n" + g for g in harmful]
    return {"benign": safe, "harmful": harmful, "jailbreak": jailbreak}


@torch.no_grad()
def generate(tok, model, prompt):
    kw = {"enable_thinking": True} if ENABLE_THINKING else {}
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False, **kw)
    enc = tok(chat, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                         temperature=0.6, top_p=0.95,
                         pad_token_id=tok.eos_token_id)[0]
    return tok.decode(out[enc.input_ids.shape[1]:], skip_special_tokens=False)


def split(text):
    end = text.find("</think>")
    if end == -1:
        return text.strip(), "(no </think> — generation hit the token cap)"
    return text[:end].strip(), text[end + len("</think>"):].strip()


prompts = load_prompts()
picker = np.random.default_rng(SAMPLE_SEED)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()

lines = []
records = []


def emit(text=""):
    """Print to screen and collect for the file."""
    print(text, flush=True)
    lines.append(text)


emit(f"Model: {MODEL}")
emit(f"Sampling seed: {SAMPLE_SEED}. Indices chosen before generation, "
     f"nothing discarded after seeing output.")
emit()

for group in ("benign", "harmful", "jailbreak"):
    idx = picker.choice(200, size=PER_GROUP, replace=False)
    emit("=" * 72)
    emit(f"{group.upper()}  (indices {sorted(idx.tolist())})")
    emit("=" * 72)
    for i in sorted(idx.tolist()):
        p = prompts[group][i]
        thinking, answer = split(generate(tok, model, p))

        # Full text goes to JSON; the readable file gets excerpts.
        records.append({"group": group, "index": i, "prompt": p,
                        "thinking": thinking, "answer": answer})

        emit(f"\n--- index {i} ---")
        # Jailbreak prompts are very long; show the tail where the request is.
        shown = p if len(p) < 600 else "[...wrapper truncated...]\n" + p[-500:]
        emit(f"PROMPT:\n{shown}\n")
        emit(f"THINKING (first 600 chars):\n{thinking[:600]}\n")
        emit(f"ANSWER (first 400 chars):\n{answer[:400]}\n")

with open("samples.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
with open("samples.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=1, ensure_ascii=False)

print("\nwrote samples.txt and samples.json")
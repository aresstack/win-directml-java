#!/usr/bin/env python3
"""External quality probe for google/gemma-3-270m-it (MODEL-GEMMA-2).

Runs a representative subset of the eval prompts in
docs/model-candidate-gemma3-270m-eval-prompts.md through Hugging Face transformers on CPU.

This is a *throwaway external probe*. It does NOT touch this project's Java/WARP runtime, the
.wdmlpack format, the workbench, or the downloader. It only helps decide GO / WAIT / NO-GO for a
future gemma/ runtime family (see docs/model-candidate-gemma3-270m.md, currently WAIT).

Gemma is gated: accept the licence on the model page and set HF_TOKEN (see
docs/model-candidate-gemma3-270m-external-probe.md). A 401/403 means licence/token are missing.

Usage:
    python scripts/probe-gemma3-270m-it.py
    python scripts/probe-gemma3-270m-it.py --compare smollm2 qwen
    python scripts/probe-gemma3-270m-it.py --max-new-tokens 128 --limit 12 --out gemma-probe.md
"""
from __future__ import annotations

import argparse
import sys
import time

GEMMA_MODEL = "google/gemma-3-270m-it"

# Optional comparison models (best effort: loaded from local HF cache or downloaded if available).
COMPARE_MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "qwen": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
}

NATURAL = (
    "DEFINE DATA LOCAL\n"
    "1 #EMPLOYEES VIEW OF EMPLOYEES\n"
    "  2 NAME\n"
    "  2 FIRST-NAME\n"
    "  2 PERSONNEL-ID\n"
    "END-DEFINE\n"
    "READ #EMPLOYEES BY NAME\n"
    "  DISPLAY NAME FIRST-NAME PERSONNEL-ID\n"
    "END-READ\n"
    "END"
)
JAVA = (
    'try (RerankerModelHandle reranker = runtime.loadReranker(rerankerConfig)) {\n'
    '    var results = reranker.rerank("What is machine learning?", List.of(\n'
    '            "Machine learning is a branch of artificial intelligence.",\n'
    '            "The weather today is sunny.",\n'
    '            "Deep learning uses neural networks."));\n'
    '    for (var r : results) {\n'
    '        System.out.printf("  index=%d, score=%.4f%n", r.originalIndex(), r.score());\n'
    '    }\n'
    '}'
)

# Representative subset of docs/model-candidate-gemma3-270m-eval-prompts.md (all 5 groups).
PROMPTS = [
    ("1 explain", "Explain in 2-3 sentences what this Natural/ADABAS program does:\n\n" + NATURAL),
    ("1 explain", "Erkläre in einem Satz auf Deutsch, was `READ #EMPLOYEES BY NAME` bewirkt:\n\n" + NATURAL),
    ("1 explain", "Explain what this Java block does, in 2-3 sentences:\n\n" + JAVA),
    ("2 pseudocode", "Translate this Natural program into ~8 lines of language-agnostic pseudocode:\n\n" + NATURAL),
    ("2 pseudocode", "Sketch a Java method `void listEmployeesByName()` equivalent in intent to this "
                     "(skeleton only, no real DB API):\n\n" + NATURAL),
    ("2 pseudocode", "Rewrite this Natural program as a short SQL SELECT ... ORDER BY with the same intent:\n\n" + NATURAL),
    ("3 json", "From this code, output JSON only: {\"view\": \"...\", \"accessType\": \"...\", "
               "\"fields\": [\"...\"]}\n\n" + NATURAL),
    ("3 json", "From this code, extract called methods as JSON only: {\"calls\": [\"...\"]}\n\n" + JAVA),
    ("3 json", "From this code, extract the query and documents as JSON only: "
               "{\"query\": \"...\", \"documents\": [\"...\"]}\n\n" + JAVA),
    ("4 summary", "Summarize in one English sentence: The Qwen2.5-Coder series is a code-specific large "
                  "language model built on Qwen2.5, pretrained on source code, with the 0.5B variant suited "
                  "to lightweight low-latency code generation on CPU."),
    ("4 summary", "Fasse in einem deutschen Satz zusammen: Die Qwen2.5-Coder-Reihe ist eine auf Code "
                  "spezialisierte Modellfamilie; die 0.5B-Variante eignet sich für ressourcenschonende "
                  "Code-Generierung auf der CPU."),
    ("4 summary", "Summarize in 2 bullet points: Gemma's 256k LM-head dominates per-token FP32 cost and sits "
                  "near Qwen-0.5B, while Gemma's body is cheaper, so a clear speed win is not guaranteed."),
    ("5 rephrase", "Rephrase in simpler English (1 sentence): The 0.5B variant is the smallest model in the "
                   "series and is suitable for lightweight, low-latency code generation tasks on CPU."),
    ("5 rephrase", "Übersetze ins Deutsche: A clear speed win over Qwen-0.5B is therefore not guaranteed."),
    ("5 rephrase", "Übersetze ins Englische: Die 0.5B-Variante eignet sich für ressourcenschonende "
                   "Code-Generierung auf der CPU."),
]


def load(model_id: str):
    import torch  # noqa: F401  (ensures a clear error if torch is missing)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    model.eval()
    return tok, model


def generate(tok, model, prompt: str, max_new_tokens: int):
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", tokenize=True
    )
    start = time.time()
    out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.time() - start
    gen = out[0][inputs.shape[-1]:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    tok_per_s = (len(gen) / elapsed) if elapsed > 0 else 0.0
    return text, len(gen), elapsed, tok_per_s


def run_model(name: str, model_id: str, prompts, max_new_tokens: int, out_lines):
    print(f"\n=== {name}  ({model_id}) ===")
    try:
        tok, model = load(model_id)
    except Exception as e:  # gated 401/403, offline, missing model, etc.
        msg = str(e)
        hint = ""
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            hint = ("  -> Gemma is gated: accept the licence on the model page and set HF_TOKEN "
                    "(see docs/model-candidate-gemma3-270m-external-probe.md).")
        print(f"  SKIPPED: could not load {model_id}: {msg}{hint}")
        out_lines.append(f"\n### {name} ({model_id}) — SKIPPED: {msg}\n")
        return
    out_lines.append(f"\n### {name} ({model_id})\n")
    for group, prompt in prompts:
        text, n, elapsed, tps = generate(tok, model, prompt, max_new_tokens)
        first_line = prompt.splitlines()[0]
        print(f"\n[{group}] {first_line}")
        print(f"  ({n} tok, {elapsed:.1f}s, {tps:.1f} tok/s)")
        print("  " + text.replace("\n", "\n  "))
        out_lines.append(f"- **[{group}]** {first_line}  _( {n} tok, {tps:.1f} tok/s )_\n\n"
                         f"  ```\n  {text.replace(chr(10), chr(10) + '  ')}\n  ```\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="External quality probe for google/gemma-3-270m-it.")
    ap.add_argument("--compare", nargs="*", default=[], choices=list(COMPARE_MODELS),
                    help="also run local comparison models (best effort; skipped if unavailable)")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=len(PROMPTS), help="number of prompts to run")
    ap.add_argument("--out", default=None, help="optional markdown output file")
    args = ap.parse_args()

    prompts = PROMPTS[: max(1, args.limit)]
    out_lines = ["# Gemma 3 270M-it external probe output\n"]

    run_model("gemma-3-270m-it", GEMMA_MODEL, prompts, args.max_new_tokens, out_lines)
    for key in args.compare:
        run_model(key, COMPARE_MODELS[key], prompts, args.max_new_tokens, out_lines)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.writelines(out_lines)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

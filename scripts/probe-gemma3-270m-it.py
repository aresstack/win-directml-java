#!/usr/bin/env python3
"""External quality probe for google/gemma-3-270m-it (MODEL-GEMMA-2/3/4).

Runs the eval prompts from docs/model-candidate-gemma3-270m-eval-prompts.md through Hugging Face
transformers on CPU. This is a *throwaway external probe*: it does NOT touch this project's Java/WARP
runtime, the .wdmlpack format, the workbench, or the downloader.

Gemma is gated: accept the licence on the model page and set HF_TOKEN (see
docs/model-candidate-gemma3-270m-external-probe.md). A 401/403 means licence/token are missing.

The PROMPTS list below mirrors all 28 prompts from the eval-prompts markdown (5 groups), with the
shared SNIP-NATURAL / SNIP-JAVA snippets inlined so the set is machine-readable and self-contained.

Usage:
    python scripts/probe-gemma3-270m-it.py
    python scripts/probe-gemma3-270m-it.py --compare smollm2 qwen
    python scripts/probe-gemma3-270m-it.py --out build/gemma-probe/gemma3-270m-it.jsonl --limit 28 --max-new-tokens 160
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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

# Full 28 prompts mirroring docs/model-candidate-gemma3-270m-eval-prompts.md. (id, group, prompt).
PROMPTS = [
    # Group 1 — Code erklären (7)
    ("P1.1", "1 explain", "Explain in 2-3 sentences what this Natural/ADABAS program does:\n\n" + NATURAL),
    ("P1.2", "1 explain", "Was bewirkt `READ #EMPLOYEES BY NAME` im folgenden Programm? Antworte in einem Satz "
                          "auf Deutsch:\n\n" + NATURAL),
    ("P1.3", "1 explain", "In this program: which database view and which three fields are read? List them.\n\n"
                          + NATURAL),
    ("P1.4", "1 explain", "Explain what this Java block does, in 2-3 sentences:\n\n" + JAVA),
    ("P1.5", "1 explain", "In this Java code, what is the purpose of the try-with-resources around `reranker`? "
                          "One sentence.\n\n" + JAVA),
    ("P1.6", "1 explain", "Erkläre kurz (Deutsch, 2 Sätze), was `reranker.rerank(query, docs)` semantisch tut:\n\n"
                          + JAVA),
    ("P1.7", "1 explain", "Explain this Java line: `float[] vector = embeddings.embed(\"Hello, world!\");`. "
                          "What is the return type and what does it represent?"),
    # Group 2 — Pseudocode / Java-Skizze (5)
    ("P2.1", "2 pseudocode", "Translate this Natural program into ~8 lines of language-agnostic pseudocode:\n\n"
                             + NATURAL),
    ("P2.2", "2 pseudocode", "Sketch a Java method `void listEmployeesByName()` equivalent in intent to this "
                             "(skeleton only, no real DB API):\n\n" + NATURAL),
    ("P2.3", "2 pseudocode", "Convert this Java code into 5-7 lines of pseudocode (no Java syntax):\n\n" + JAVA),
    ("P2.4", "2 pseudocode", "Rewrite this Natural program as a short SQL SELECT ... ORDER BY with the same "
                             "intent:\n\n" + NATURAL),
    ("P2.5", "2 pseudocode", "Sketch as pseudocode how to extend this Java code so only hits with score > 0.5 "
                             "are printed:\n\n" + JAVA),
    # Group 3 — JSON-Extraktion (6)
    ("P3.1", "3 json", "From this code, output JSON only: {\"view\": \"...\", \"accessType\": \"...\", "
                       "\"fields\": [\"...\"]}\n\n" + NATURAL),
    ("P3.2", "3 json", "From this code, list database side effects as JSON only: {\"reads\": [\"...\"], "
                       "\"writes\": []}\n\n" + NATURAL),
    ("P3.3", "3 json", "From this code, extract called methods as JSON only: {\"calls\": [\"...\"]}\n\n" + JAVA),
    ("P3.4", "3 json", "From this code, extract side effects as JSON only: {\"io\": [\"...\"], "
                       "\"mutations\": [], \"externalCalls\": [\"...\"]}\n\n" + JAVA),
    ("P3.5", "3 json", "From this code, extract the query and documents as JSON only: "
                       "{\"query\": \"...\", \"documents\": [\"...\"]}\n\n" + JAVA),
    ("P3.6", "3 json", "From this code, produce JSON only describing the loop: {\"loopKind\": \"...\", "
                       "\"orderedBy\": \"...\", \"displays\": [\"...\"]}\n\n" + NATURAL),
    # Group 4 — Technische Zusammenfassung (5)
    ("P4.1", "4 summary", "Summarize in one English sentence: The Qwen2.5-Coder series is a code-specific large "
                          "language model built upon the Qwen2.5 architecture, pretrained on a large corpus of "
                          "source code, with the 0.5B variant suited to lightweight low-latency code generation "
                          "on CPU."),
    ("P4.2", "4 summary", "Fasse in einem deutschen Satz zusammen: Die Qwen2.5-Coder-Reihe ist eine auf Code "
                          "spezialisierte Familie großer Sprachmodelle auf Basis der Qwen2.5-Architektur; die "
                          "0.5B-Variante eignet sich für ressourcenschonende Code-Generierung auf der CPU."),
    ("P4.3", "4 summary", "Summarize in 2 bullet points: Gemma's 256k LM-head is the dominant per-token cost on "
                          "FP32-WARP and sits roughly in Qwen-0.5B's range, while Gemma's transformer body is much "
                          "cheaper. A clear speed win over Qwen-0.5B is therefore not guaranteed."),
    ("P4.4", "4 summary", "In einem Satz (Deutsch): Worin liegt das Hauptrisiko bei gemma-3-270m-it? "
                          "gemma-3-270m-it ist von Google explizit als Fine-Tune-Basis / task-specific positioniert "
                          "(instruction following, strukturierte Ausgaben, Extraktion), nicht als starker "
                          "genereller Chat-Assistent."),
    ("P4.5", "4 summary", "Give a 3-item TL;DR of: embedding models map text to fixed-length float vectors; "
                          "rerankers score query/document pairs; both run on CPU or DirectML in this project."),
    # Group 5 — DE/EN-Umformulierung (5)
    ("P5.1", "5 rephrase", "Rephrase in simpler English (1 sentence): The 0.5B variant is the smallest model in "
                           "the series and is suitable for lightweight, low-latency code generation tasks on CPU."),
    ("P5.2", "5 rephrase", "Übersetze ins Deutsche: A clear speed win over Qwen-0.5B is therefore not guaranteed."),
    ("P5.3", "5 rephrase", "Übersetze ins Englische: Die 0.5B-Variante eignet sich für ressourcenschonende "
                           "Code-Generierung auf der CPU."),
    ("P5.4", "5 rephrase", "Formuliere höflicher und knapper (Deutsch): Mach das Modell kleiner, sonst ist es zu "
                           "langsam."),
    ("P5.5", "5 rephrase", "Rewrite for a release note (concise, English): we made special_tokens_map.json "
                           "optional so a 404 no longer aborts the Qwen download."),
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
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", tokenize=True
    )
    # transformers <5 returns a tensor; >=5 returns a BatchEncoding (dict with input_ids).
    if hasattr(enc, "shape"):
        input_ids = enc
        extra = {}
    else:
        input_ids = enc["input_ids"]
        extra = {k: v for k, v in enc.items() if k != "input_ids"}
    start = time.time()
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, **extra)
    elapsed = time.time() - start
    gen = out[0][input_ids.shape[-1]:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    tok_per_s = (len(gen) / elapsed) if elapsed > 0 else 0.0
    return text, len(gen), elapsed, tok_per_s


def run_model(name: str, model_id: str, prompts, max_new_tokens: int, records: list):
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
        records.append({"model": name, "model_id": model_id, "skipped": True, "reason": msg})
        return
    for pid, group, prompt in prompts:
        text, n, elapsed, tps = generate(tok, model, prompt, max_new_tokens)
        first_line = prompt.splitlines()[0]
        print(f"\n[{pid} {group}] {first_line}")
        print(f"  ({n} tok, {elapsed:.1f}s, {tps:.1f} tok/s)")
        print("  " + text.replace("\n", "\n  "))
        records.append({
            "model": name, "model_id": model_id, "id": pid, "group": group,
            "prompt": prompt, "response": text, "tokens": n,
            "elapsed_s": round(elapsed, 3), "tok_per_s": round(tps, 2),
        })


def write_output(out_path: str, records: list) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)  # robust: create the output dir if missing
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        with path.open("w", encoding="utf-8") as f:
            f.write("# Gemma 3 270M-it external probe output\n")
            for rec in records:
                if rec.get("skipped"):
                    f.write(f"\n### {rec['model']} ({rec['model_id']}) — SKIPPED: {rec['reason']}\n")
                    continue
                f.write(f"\n- **[{rec['id']} {rec['group']}]** _( {rec['tokens']} tok, "
                        f"{rec['tok_per_s']} tok/s )_\n\n  ```\n  "
                        + rec["response"].replace("\n", "\n  ") + "\n  ```\n")
    print(f"\nWrote {path} ({sum(1 for r in records if not r.get('skipped'))} prompt records)")


def main() -> int:
    ap = argparse.ArgumentParser(description="External quality probe for google/gemma-3-270m-it.")
    ap.add_argument("--compare", nargs="*", default=[], choices=list(COMPARE_MODELS),
                    help="also run local comparison models (best effort; skipped if unavailable)")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=len(PROMPTS), help="number of prompts to run (max 28)")
    ap.add_argument("--out", default=None, help="optional output file (.jsonl -> JSONL, else markdown)")
    args = ap.parse_args()

    prompts = PROMPTS[: max(1, min(args.limit, len(PROMPTS)))]
    print(f"Running {len(prompts)} of {len(PROMPTS)} prompts, max_new_tokens={args.max_new_tokens}")
    records: list = []

    run_model("gemma-3-270m-it", GEMMA_MODEL, prompts, args.max_new_tokens, records)
    for key in args.compare:
        run_model(key, COMPARE_MODELS[key], prompts, args.max_new_tokens, records)

    if args.out:
        write_output(args.out, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())

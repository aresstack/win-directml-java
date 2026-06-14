#!/usr/bin/env python3
"""Local Gemma 3 generation bridge for the Java Workbench.

This is deliberately an external probe path. It loads an already downloaded local
Hugging Face directory with Transformers on CPU and writes the generated text to
an output file. The Java side remains free of PyTorch/Transformers dependencies.
"""
from __future__ import annotations

import argparse
import sys
import time


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text with local Gemma 3 files.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    prompt = load_text(args.prompt_file)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - exercised by the Java side in real installs
        print("GEMMA3_ERROR=Missing Python dependency: install torch and transformers.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 3

    try:
        load_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            local_files_only=True,
            torch_dtype="auto",
        )
        model.eval()
        load_elapsed_ms = int((time.time() - load_start) * 1000)

        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        generate_start = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generate_elapsed_ms = int((time.time() - generate_start) * 1000)
        generated_ids = output[0][prompt_tokens:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        output_tokens = int(generated_ids.shape[-1])
        write_text(args.out, generated_text)

        print(f"GEMMA3_MODEL_LOAD_MS={load_elapsed_ms}")
        print(f"GEMMA3_GENERATE_MS={generate_elapsed_ms}")
        print(f"GEMMA3_PROMPT_TOKENS={prompt_tokens}")
        print(f"GEMMA3_OUTPUT_TOKENS={output_tokens}")
        return 0
    except Exception as exc:  # pragma: no cover - surfaced verbatim to the Workbench
        print("GEMMA3_ERROR=Generation failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

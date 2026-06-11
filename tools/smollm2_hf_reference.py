"""
Independent Hugging Face reference for the SmolLM2 reference-runtime numerical check.

Runs the EXACT same rendered prompt the workbench feeds to the Java SmolLM2 runtime,
prints the prompt token IDs and the top-10 next-token logits (step 0, raw, pre-penalty).
Compare these against the Java workbench output:
  - "Prompt tokens" / console "promptTokenIds"
  - "top-10 @ step 0 (raw, pre-penalty)"

Usage:
  python tools/smollm2_hf_reference.py <model_dir>
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\angel\AppData\Roaming\.directml\model\smollm2-135m-instruct"

# EXACT rendered prompt from the workbench (instruction-last user turn, no system turn).
prompt = ("<|im_start|>user\n"
          "Paste a longer text or prompt here. The workbench will generate output using the selected decoder model.\n\n"
          "Translate the user's text into German. Output only the translation.<|im_end|>\n"
          "<|im_start|>assistant\n")

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()

ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
print("HF prompt token count:", ids.shape[1])
print("HF prompt token ids:", ids[0].tolist())

with torch.no_grad():
    logits = model(ids).logits[0, -1].float()

top = torch.topk(logits, 10)
print("HF top-10 @ step 0 (raw):")
for v, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  #{i}={v:.4f}  {tok.decode([i])!r}")

# Greedy first token (what the model would actually pick)
print("HF argmax token:", int(torch.argmax(logits)), repr(tok.decode([int(torch.argmax(logits))])))

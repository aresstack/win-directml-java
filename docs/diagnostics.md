# Diagnostics

Diagnostic switches and tools that are **not** part of the runtime and **not** required for
deployment. They exist purely to investigate numerical or prompt issues.

## SmolLM2 debug flags (JVM system properties)

By default the workbench output panel stays lean (only OUTPUT, runtime mode/package, token
counts, finish reason, profile timings, and a one-line generation-config summary such as
`Generation config: greedyChat, repetitionPenalty=1.2, maxTokens=256`).

Set these VM options to surface diagnostic material:

| Flag | Effect |
|------|--------|
| `-Dsmollm2.debug.prompt=true` | Logs the final rendered prompt, prompt token IDs and the effective model config (console), and shows the rendered prompt + prompt task + effective model config in the output panel. |
| `-Dsmollm2.debug.topk=N` | Captures and shows the top-`N` **raw** next-token logits (pre repetition-penalty) for the first decode steps, in the panel and the console. |
| `-Dsmollm2.debug.steps=K` | Number of decode steps to capture top-K for (default 3). Only effective together with `-Dsmollm2.debug.topk`. |

Example (IntelliJ run configuration VM options):

```
-Dsmollm2.debug.prompt=true -Dsmollm2.debug.topk=10 -Dsmollm2.debug.steps=3
```

The captured "top-K raw logits @ step 0" line is what you compare against the Hugging Face
reference below.

## Hugging Face reference: `tools/smollm2_hf_reference.py`

Use this tool **only** to compare the Java SmolLM2 reference-runtime logits against Hugging
Face Transformers. It is not part of the runtime and not required for deployment.

```
pip install torch transformers safetensors
python tools/smollm2_hf_reference.py "C:\path\to\.directml\model\smollm2-135m-instruct"
```

It prints the prompt token count, the prompt token IDs, and the top-10 next-token logits
(step 0, raw) for the exact prompt the workbench renders. Compare:

- prompt token IDs  ↔  Java console `SmolLM2 prompt ... promptTokenIds=[...]`
- top-10 step 0     ↔  Java panel/console `top-10 @ step 0 (raw, pre-penalty)`

Interpretation:

- **Prompt token IDs differ** → tokenizer/ChatML issue (before inference).
- **IDs identical, top-K logits differ strongly** → inference bug (RoPE / GQA / tensor layout / RMSNorm / LM head).
- **IDs identical, top-K logits match** → the forward pass is correct; weak output is model capability, not a bug.

### Verified result (2026-06)

For SmolLM2-135M-Instruct the Java top-10 step-0 logits matched Hugging Face Transformers
**to 4 decimal places on all 10 tokens**, and HF greedy decoding (repetitionPenalty=1.2)
produced the same degenerate German output. Conclusion: the SmolLM2 reference forward pass,
tokenizer, RoPE and GQA are numerically correct. The weak translation quality is the
capability limit of the 135M/360M models, not an engine defect. Use Qwen for translation.

> Policy: no further SmolLM2 template/penalty/forward-pass changes unless a new numerical
> divergence from the Hugging Face reference is demonstrated with the tool above.

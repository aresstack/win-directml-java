# Qwen long input prefill fix

This build removes the artificial 256-token prompt/prefill limit in the Workbench Qwen path.

## Problem

The Summarizer tab forwarded the selected output max token value to Qwen, but the WARP/INT4 prefill batch kernel
rejected input batches with `M > 256`:

```text
M out of range [1, 256]: <prompt token count>
```

That made longer texts fail before generation started.

## Fix

`MatMulNBitsKernel.matmulBatch(...)` now accepts `M > 256` and internally splits the prefill batch into chunks of at
most `MAX_BATCH_M` rows. This keeps the same GPU scratch-memory cap while allowing longer prompts.

The Workbench label was also clarified from `Max tokens` to `Max output tokens`, because this value controls generated
output length, not input/context length.

## Expected log

For prompts over 256 tokens, the terminal should show lines like:

```text
matmulBatch: splitting long batch M=... into chunks of <=256 for [N,K]
```

Long prompts will be slower because each prefill projection is split into multiple bounded GPU submissions, but they
should no longer fail at the 256-token boundary.

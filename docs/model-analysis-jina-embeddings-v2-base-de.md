# Model analysis: `jinaai/jina-embeddings-v2-base-de`

Status in registry: **planned** (`useCase=embedding`, `embedFamily=null`).

## Summary decision

- Keep this model **planned**.
- Do not claim runtime support until a dedicated Jina-v2 path exists and passes real-model tests.

## Analysis checklist

### Tokenizer type
- Expected: WordPiece-like tokenizer with Jina-specific normalization/tokenization details.
- Action: inspect `tokenizer.json`/vocab assets and compare behavior against `WordPieceTokenizer` assumptions.

### Architecture and config fields
- Expected: BERT-derived encoder with Jina-specific attention/position-bias behavior.
- Action: inspect `config.json` for:
  - hidden/layer/head/intermediate sizes,
  - max sequence length,
  - position-bias fields (ALiBi/relative-bias flags),
  - pooling/normalization expectations.

### Weight naming compatibility
- Current loader expects canonical BERT prefixes (`embeddings.*`, `encoder.layer.*`).
- Action: list safetensors keys and classify:
  - direct match with current BERT loader,
  - remap-only changes,
  - hard incompatibilities requiring runtime changes.

### Positional bias requirements
- Jina v2 is expected to require ALiBi or equivalent non-absolute positional bias.
- Current pipeline is built around absolute position embedding tables.
- Action: design explicit CPU + DirectML positional-bias integration path.

## Compatibility outcome against current core

- Current **BERT/MiniLM/E5 WordPiece core is not a guaranteed drop-in** for Jina v2.
- CPU-first support may be realistic if:
  - tokenizer compatibility is confirmed,
  - positional-bias math is integrated in CPU attention.
- DirectML support requires corresponding GPU graph additions for the same positional-bias behavior.

## DirectML impact (required work)

- Add positional-bias application in DirectML attention path.
- Ensure parity with CPU path on real model data.
- Add tests that prove equivalent behavior with and without batching.

## Download script requirements

Proposed new script: `scripts/download-jina.ps1`

Required files (minimum):
- `model.safetensors`
- `tokenizer.json`
- `config.json`

Optional metadata:
- `special_tokens_map.json`
- `tokenizer_config.json`
- vocab asset if tokenizer references it separately

Script requirements:
- fail hard when required files cannot be fetched,
- skip optional files when absent,
- print final sidecar invocation hints.

## Real-model test requirements

Must add before status promotion:
- CPU real-model reference test (dimension, norm, semantic sanity).
- CPU↔DirectML parity test with threshold policy matching shipped criteria.
- `embedBatch` parity/invariant tests (order, count, normalization flags).

## Workbench behavior

- While `planned` and `embedFamily=null`, model must not be selectable in Workbench embed dropdown.
- Sidecar must keep explicit status-aware error if passed directly via `-Dembed.model=<full-id>`.

## Status transition gates

- **planned → experimental**: CPU real-model support + docs + tests.
- **experimental → shipped**: CPU+DirectML parity + benchmark + release-smoke evidence.

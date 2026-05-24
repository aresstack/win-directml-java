# Model analysis: `intfloat/multilingual-e5-large-instruct`

Status in registry: **planned** (`useCase=embedding`, `embedFamily=null`).

## Summary decision

- Keep this model **planned**.
- It is blocked on SentencePiece/XLM-R support and is not compatible with the current WordPiece-only E5 runtime path.

## Analysis checklist

### Tokenizer and architecture
- Expected tokenizer stack: SentencePiece / XLM-R style.
- Current runtime tokenizer stack for E5 is WordPiece.
- Action: inspect tokenizer assets (`tokenizer.json`, `tokenizer.model`/SentencePiece artifacts) and map required behavior.

### Config and weight-name differences
- Action: inspect `config.json` and safetensors keys for:
  - XLM-R naming and prefix differences,
  - embedding table/token-type assumptions,
  - shape requirements.
- Compare against current BERT loader expectations and classify incompatibilities.

### Pooling and normalization
- Expected embedding contract remains mean pooling + L2 normalization.
- Action: verify model-specific defaults and confirm they align with existing embedding output contract.

### Prefix / instruction format
- Model is instruction-style and should preserve documented query formatting:
  - `Instruct: ...\nQuery: ...`
- Action: define where this formatting is enforced (request helper + docs + tests).

## Compatibility outcome against current E5 path

- Current `E5Encoders` path is explicitly WordPiece-focused.
- multilingual-e5-large-instruct needs:
  1. SentencePiece tokenizer implementation,
  2. XLM-R compatible model adapter/loader handling.

## Implementation work required

- New SentencePiece tokenizer path in encoder module.
- XLM-R-compatible embedding/weight mapping (CPU first).
- Sidecar wiring for full model ID and eventual family selector once runtime exists.
- Real-model tests for semantic behavior + normalize + embedBatch invariants.
- DirectML parity path only after CPU reference is established.

## Real-model test requirements

Before status promotion:
- CPU real-model test with multilingual semantic sanity cases.
- `embedBatch` invariants and output-order checks.
- CPU↔DirectML parity test only when DirectML path exists.

## Workbench behavior

- While planned/unimplemented, model must stay out of selectable Workbench embed dropdown.
- Direct `-Dembed.model=intfloat/multilingual-e5-large-instruct` must continue returning status-aware unimplemented error.

## Documentation requirements

When implementation starts:
- Update `SUPPORTED_MODELS.md` tokenizer/architecture notes.
- Update `README.md` and `PROTOCOL.md` for instruction/prefix guidance.
- Add download-script docs once a script exists.

## Status transition gates

- **planned → experimental**: CPU real-model support with SentencePiece/XLM-R path + docs/tests.
- **experimental → shipped**: CPU+DirectML parity, benchmark evidence, and release-smoke validation.

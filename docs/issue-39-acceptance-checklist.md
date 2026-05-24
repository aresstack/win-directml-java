# Issue #39 acceptance checklist

Issue link: `#39`  
Partial PR wording: **Part of #39** / **Refs #39**.

> Keep #39 open until every checkbox below is complete with linked evidence.

## Registry and gating

- [x] Shared Java-8 model registry in `directml-config`
- [x] Sidecar embed gate resolves full IDs through registry
- [x] Decoder/summarizer IDs rejected with explicit error text
- [x] Planned/unimplemented embedding IDs rejected with status-aware error text

## Workbench behavior

- [x] `embed.model` dropdown excludes decoder/summarizer entries
- [x] `embed.model` dropdown excludes planned embeddings without runtime implementation (`embedFamily == null`)
- [ ] Optional UI annotation strategy decided for planned models (if reintroduced as visible-but-disabled)

## E5/MiniLM validation package

- [x] E5 `download-e5.ps1` variant flow documented and validated
- [x] E5 selection validated for alias + direct model ID paths
- [x] E5 `embed` and `embedBatch` parity tests present
- [ ] E5 CPU↔DirectML parity evidence captured on release Windows hardware
- [x] MiniLM shipped behavior validated (alias/direct ID/docs/tests)
- [ ] MiniLM release Windows parity evidence refreshed

## Planned models (analysis gates)

- [x] Jina analysis document added with explicit compatibility delta and status gates
- [x] multilingual-E5 analysis document added with SentencePiece/XLM-R blockers and status gates
- [ ] Jina real-model CPU evidence (for planned→experimental)
- [ ] multilingual-E5 real-model CPU evidence (for planned→experimental)

## Benchmark package

- [x] Benchmark methodology structured for supported embedding models only
- [x] Matrix includes `embed` and `embedBatch`, `N=10/50/100`, CPU+DirectML
- [ ] Full Windows benchmark result table populated for shipped/experimental embedding variants
- [ ] Recommendation section finalized with measured data

## Docs coherence

- [x] `SUPPORTED_MODELS.md` aligned with registry and status-transition policy
- [x] `README.md` aligned with full-ID gating and Workbench selection behavior
- [x] `directml-sidecar/PROTOCOL.md` aligned with full-ID gating semantics
- [x] `WORKBENCH.md` aligned with dropdown policy
- [x] Analysis + acceptance matrix docs added under `docs/`

## Out of scope / separate follow-up

- [ ] Existing sidecar async race flake (`SidecarLifecycleTest.summarizeUsesInjectedSummarizer`) tracked separately if it blocks release confidence

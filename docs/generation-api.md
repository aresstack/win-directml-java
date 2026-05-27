# Generation API Architecture

This document describes the text-generation API introduced to support
decoder-only (causal LM) and future sequence-to-sequence models alongside
the existing embedding and reranking pipelines.

## Model Taxonomy

| Category                      | Registry                   | Interface                  | Description                                                  |
|-------------------------------|----------------------------|----------------------------|--------------------------------------------------------------|
| **Embeddings**                | `EmbeddingModelRegistry`   | `EmbeddingModel`           | Encoder models mapping text → fixed-size vectors (MiniLM, E5, Jina). |
| **Rerankers**                 | `EmbeddingModelRegistry`   | (cross-encoder)            | Cross-encoder models scoring query–document relevance.        |
| **Decoder-only (Causal LM)** | `GenerationModelRegistry`  | `CausalLanguageModel`      | Autoregressive models generating text token-by-token (Phi-3, Qwen). |
| **Seq2Seq generation**        | `GenerationModelRegistry`  | `TextGenerationModel`      | Encoder-decoder models (T5, BART). Future; issue #95.         |
| **Summarizer**                | *(use-case adapter)*       | `Summarizer`               | Application-layer wrapper around a generation model with a fixed prompt. |

## Key Design Decisions

1. **Separate registries**: `EmbeddingModelRegistry` classifies embedding and
   reranker checkpoints. `GenerationModelRegistry` classifies text-generation
   checkpoints. Neither registry pretends the other's models belong to it.

2. **Shared generation interface**: `TextGenerationModel` is the public contract
   for any model that generates text. `CausalLanguageModel` is a marker sub-interface
   for decoder-only architectures, enabling type-safe distinctions where needed.

3. **Model-agnostic request/result**: `GenerationRequest` and `GenerationResult`
   carry no Phi-3-specific or Qwen-specific types. Model-specific prompt formatting
   (chat templates, stop tokens) is handled by implementations.

4. **Summarizer is an adapter, not a model type**: The `Summarizer` interface wraps
   a `TextGenerationModel` with a summarization system prompt. It does not appear in
   the generation registry as a separate entry.

5. **Java-8 compatible**: All types in `com.aresstack.windirectml.config.generation`
   are Java-8 compatible so the Swing workbench can consume them.

## API Types

### Core Interfaces

```
TextGenerationModel          – generate(GenerationRequest) → GenerationResult
  └── CausalLanguageModel    – marker for decoder-only models
```

### Request/Result

```
GenerationRequest            – userPrompt, systemPrompt, maxTokens, sampler, stopPolicy
GenerationResult             – text, finishReason, promptTokens, completionTokens, elapsedMs
```

### Configuration Types

```
GenerationModelId            – type-safe model identifier
SamplerConfig                – greedy / temperature+topK
ChatTemplate                 – PHI3, CHATML, RAW, UNKNOWN
StopTokenPolicy              – EOS-only or custom stop strings
```

### Registry

```
GenerationModelRegistry      – static registry of known generation checkpoints
  .entries()                 – all entries
  .runnableEntries()         – only shipped/experimental
  .entriesByArchitecture()   – filter by CAUSAL_LM or SEQ2SEQ
  .entriesByStatus()         – filter by status
  .findByModelId()           – case-insensitive lookup
```

## Current Model States

| Model                                    | Architecture | Status       | Notes                              |
|------------------------------------------|--------------|--------------|------------------------------------|
| microsoft/Phi-3-mini-4k-instruct-onnx    | CAUSAL_LM    | experimental | Active summarizer/generation backend |
| microsoft/Phi-3.5-mini-instruct-onnx     | CAUSAL_LM    | planned      | Successor to Phi-3 Mini             |
| Qwen/Qwen2.5-Coder-0.5B-Instruct        | CAUSAL_LM    | planned      | Workbench-visible CPU test path; not shipped |
| Qwen/Qwen2.5-Coder-1.5B-Instruct        | CAUSAL_LM    | planned      | Scale-up candidate                  |
| Qwen/Qwen2.5-Coder-3B-Instruct          | CAUSAL_LM    | planned      | Largest planned local deployment    |

## Workbench Integration

Most workbench generation selectors query active generation models via:

```java
GenerationModelRegistry.runnableEntries()
```

This returns only models with `status = SHIPPED` or `status = EXPERIMENTAL` and
is the default filter for normal runtime-backed generation models.

The DirectML Workbench is also the manual test surface for runtime bring-up. For
that reason, the Workbench Generation/Summarizer panel may explicitly add a
planned model when there is a local test path for it. Qwen2.5-Coder 0.5B is the
current exception: it remains `status = PLANNED` in the registry, but is visible
in the Workbench so developers can download the candidate layout and exercise
`QwenInferenceEngine` locally before the model is promoted to experimental or
shipped.

## Relationship to Existing Code

- `Phi3Summarizer` implements `Summarizer` and wraps `Phi3InferenceEngine`.
  It can be adapted to additionally implement `CausalLanguageModel` in a
  follow-up ticket.
- `InferenceEngine` / `InferenceRequest` / `InferenceResult` remain as the
  internal text-generation bridge used by Phi-3 and the Qwen Workbench test path.
  The shared `GenerationRequest` / `GenerationResult` provide the public API that
  future Qwen and Seq2Seq implementations will target.
- The `EmbeddingModelRegistry` continues to hold Phi-3 and Phi-3.5 under
  `UseCase.SUMMARIZER` for backward compatibility with existing gating logic.
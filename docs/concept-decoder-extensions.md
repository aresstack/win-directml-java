# Konzept: Decoder-LLM-Erweiterungen

Tracking Issue 25.

## Ziel

Nach Phi-3 sollen weitere Decoder-LLMs in `directml-inference` möglich
sein – ohne einen universellen "ONNX-Runner" zu bauen.

## Kandidaten

- Qwen2 / Qwen2.5
- Llama (3.1 / 3.2)
- weitere Phi-Varianten (Phi-3.5, Phi-4)

## Architektur-Idee

```text
                ┌──────────────────────────────┐
                │   Summarizer / ChatRunner    │   (Use-Case-API, stabil)
                └──────────────┬───────────────┘
                               │ delegiert an
                               ▼
                ┌──────────────────────────────┐
                │   DecoderArchitecture        │   (Strategy)
                │   - Phi3Architecture         │
                │   - Qwen2Architecture        │
                │   - LlamaArchitecture        │
                └──────────────┬───────────────┘
                               │ verwendet
                               ▼
                ┌──────────────────────────────┐
                │   gemeinsamer Decoder-Core   │
                │   (KV-Cache, Sampler, Loop)  │
                └──────────────┬───────────────┘
                               │ ruft Kernels
                               ▼
                ┌──────────────────────────────┐
                │   KernelRegistry             │
                │   (LinearKernel, RmsNorm,    │
                │    SwiGlu, Attention, …)     │
                └──────────────────────────────┘
```

`DecoderArchitecture` deklariert die familienspezifischen Entscheidungen,
der Decoder-Core verarbeitet alles, was wirklich gemeinsam ist.

## Familienspezifische Unterschiede

| Aspekt          | Phi-3              | Llama 3     | Qwen2             |
|-----------------|--------------------|-------------|-------------------|
| Positions-Embed | RoPE (NeoX)        | RoPE (NeoX) | RoPE (NeoX)       |
| Attention       | GQA                | GQA         | GQA               |
| QKV-Layout      | gefused (3·hidden) | separat     | gefused (q, k, v) |
| MLP             | SwiGLU             | SwiGLU      | SwiGLU            |
| Norm            | RMSNorm            | RMSNorm     | RMSNorm           |
| Tokenizer       | Llama-BPE          | Llama-BPE   | Qwen-BPE          |
| Chat-Template   | `<                 | user        | >...<             |end|>` | `<|begin_of_text|>...` | ChatML |
| Vocab           | 32064              | 128256      | 152064            |

## Vorgeschlagene Abstraktionen

```java
public interface DecoderArchitecture {
    String name();
    int hiddenSize();
    int numLayers();
    int numAttentionHeads();
    int numKvHeads();      // GQA
    int vocabSize();
    int maxPositionEmbeddings();
    float ropeTheta();
    String activation();   // "swiglu", "gelu"
    String normalization();// "rmsnorm", "layernorm"
}

public interface ChatTemplate {
    String format(List<ChatMessage> messages, boolean addGenerationPrompt);
    List<Integer> stopTokens();
}
```

## Schrittweise Migration

1. Phi-3-Code so refaktorisieren, dass alle Modell-Konstanten aus einer
   `Phi3Architecture` kommen (keine `if "phi"` Verzweigungen).
2. Decoder-Core extrahieren: KV-Cache-Management, Sampling-Loop,
   Generierungs-Schleife.
3. Erst dann zweite Familie (z. B. Llama 3) anbinden.

## Klare Nicht-Ziele

- Kein universeller GGUF-/ONNX-Reader.
- Keine generische "transformers-Runtime in Java".
- Encoder-Pfad wird durch diese Arbeit **nicht** blockiert.

## Planungsstatus

Phase 2. Encoder-Schiene hat aktuell Priorität, damit der RAG-Use-Case
funktioniert.


# Konzept: Reranker-Runtime

Tracking Issue 24.

## Ziel

Reranker werden später als **eigene Encoder-Variante** umgesetzt, getrennt
von `EmbeddingModel`. Sie liefern keinen Vektor, sondern einen Score je
`(Query, Document)`-Paar.

## Architektur

```text
Query + Document
        ↓
   Encoder (Cross-Encoder)
        ↓
   Classification Head (1 Output-Logit)
        ↓
        Score (float)
```

Im Gegensatz zum Bi-Encoder (Sentence-Transformer für Embeddings) wird hier
ein **Cross-Encoder** verwendet: Query und Document werden zusammen
tokenisiert (mit `[SEP]` getrennt), und das Modell schaut sich beide
gleichzeitig an.

## Java-API (Vorschlag)

```java
package com.aresstack.windirectml.reranker;

public interface Reranker {

    boolean isReady();

    List<RankedDocument> rerank(RerankRequest request) throws RerankException;
}

public record RerankRequest(String query, List<String> documents, int topN) { }

public record RankedDocument(int index, String text, double score) { }
```

## Top-N-Strategie

1. Java-8-Host führt erst ein **Embedding-Recall** mit `EmbeddingModel`
   durch (z. B. Top-100 aus einem Vektorindex).
2. Diese Top-100 gehen als Documents in `Reranker.rerank(...)`.
3. Reranker liefert sortierte Top-N (typisch 5–20).

So bleibt der teure Cross-Encoder nur auf einer kleinen Kandidatenmenge.

## Batch-Verarbeitung

Reranker brauchen typischerweise einen Pass je `(query, document)`-Paar.
Effizient ist Batch-Inference mit gemeinsamem Query und N Documents:

```text
batch_input = [
   "[CLS] query [SEP] doc_1 [SEP]",
   "[CLS] query [SEP] doc_2 [SEP]",
   ...
]
```

Der `EncoderRuntime` muss dafür Batch-Größe > 1 unterstützen.

## Zielmodelle (später)

- `BAAI/bge-reranker-v2-m3`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Abgrenzung

- Reranker mischt sich **nicht** mit `EmbeddingModel`.
- Vermeidet einen universellen `Inferencer` – jedes Use-Case-Interface
  bleibt typisiert (`Summarizer`, `EmbeddingModel`, `Reranker`).

## Planungsstatus

**Status: implementiert.** Der Reranker läuft als `Reranker`-Interface
in `directml-encoder` (Paket
`com.aresstack.windirectml.encoder.reranker`) auf dem generischen
`DirectMlBertEncoderStack` – kein separater Compute-Graph. Bereitgestellt
werden `CpuReranker` und `DirectMlReranker`; der Classification-Head
({@code [1, H]}) läuft bewusst auf der CPU, weil das ein vernachlässigbar
billiger GEMM ist und der GPU-Pfad damit byte-identisch mit der
Embedding-Pipeline bleibt.

Wire-up:

- JSON-RPC-Methode `rerank` im Sidecar (siehe
  `directml-sidecar/PROTOCOL.md`).
- Java-8-Client: `SidecarClient.rerank(query, documents, topN)`.
- Workbench: Tab **Rerank**.
- Default-Modell: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  (BERT-WordPiece, 6 Layer, hidden 384). Andere BERT-basierte
  Cross-Encoder (z. B. bge-reranker-base) laden über
  `-Drerank.modelDir=<path>` ohne Codeänderungen.
- Parität: `DirectMlRerankerParityTest` vergleicht CPU- und
  DirectML-Score auf identischen synthetischen Gewichten.



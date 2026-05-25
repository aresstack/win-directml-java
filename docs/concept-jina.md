# Konzept: JinaBERT / Jina-Embeddings

Tracking Issue 34 (Milestone 8 — Weitere Modellfamilien nach dem Release).

## Ziel

Nach den stabilen Pfaden für `sentence-transformers/all-MiniLM-L6-v2`,
der E5-Familie und der Reranker-Runtime soll evaluiert werden, ob und in
welcher Form die Jina-Modelle (insbesondere
`jinaai/jina-embeddings-v2-base-de`) im Java-Host unterstützt werden.

Diese Datei ist **Konzeptarbeit**, keine Implementierungs-Pflicht. Sie
hält die Entscheidung für das Maven-Central-Release fest.

## Zielmodelle (initial)

- `jinaai/jina-embeddings-v2-base-de` – deutsch-englisch, 768-d.
- (perspektivisch) `jinaai/jina-embeddings-v2-base-en`, `…-base-zh`,
  `…-base-code`.

Reranker (`jinaai/jina-reranker-v2-base-multilingual`) ist über das
generische Reranker-Interface erreichbar, sobald die Architektur unten
geklärt ist, und wird hier nicht separat betrachtet (siehe
[concept-reranker.md](concept-reranker.md)).

## Unterschiede zu MiniLM / E5

| Aspekt        | MiniLM-L6                 | E5-base               | Jina v2 base-de                                    |
|---------------|---------------------------|-----------------------|----------------------------------------------------|
| Architektur   | BERT, 6 Layer, 384-d      | BERT/XLM-R, 12 Layer  | **JinaBERT v2** (BERT-like, 12 Layer, 768-d)       |
| Positional    | Learned absolute (max 512)| Learned absolute      | **ALiBi** (Attention Linear Biases, kein Pos-Emb)  |
| Tokenizer     | WordPiece (uncased)       | SentencePiece-BPE     | WordPiece (Jina-custom Vocab, cased)               |
| Pooling       | Mean + L2                 | Mean + L2             | Mean + L2 (identisch)                              |
| Output-Dim    | 384                       | 768                   | 768                                                |
| Max Seq-Len   | 512                       | 512                   | **8 192** (durch ALiBi möglich)                    |
| Input-Format  | Roh-Text                  | `"query: "`-Prefix    | Roh-Text (kein Prefix erforderlich)                |

Die beiden wirklich relevanten Abweichungen sind **ALiBi** und die
**lange Sequenz-Länge**. Tokenizer und Pooling sind kompatibel zur
bestehenden Bi-Encoder-Pipeline.

## Tokenizer-Prüfung

Jina v2 nutzt WordPiece mit eigenem Vokabular (`bert-base-german-cased`
sehr ähnlich, aber nicht identisch). Das aktuelle
`WordPieceTokenizer` lädt das Vokabular aus `tokenizer.json` /
`vocab.txt` und kann das Modell ohne Code-Änderung verarbeiten – nur
die Datei muss korrekt geladen werden. Casing-Flag muss von der
`tokenizer_config.json` übernommen werden (Jina v2 = cased).

→ **Tokenizer-Pfad: vorhanden, geringe Anpassung (Casing-Default).**

## Architektur-Prüfung

JinaBERT v2 ist im Kern eine BERT-Layer (Self-Attention + FFN + zwei
LayerNorms), aber:

1. **Keine Positional-Embeddings.** Stattdessen wird in der Attention
   ein zusätzliches Bias-Tensor `m_h · (i - j)` auf die Attention-Scores
   addiert (ALiBi). `m_h` ist eine kopfspezifische geometrische Reihe.
2. **GLU-FFN.** Statt `Linear → GeLU → Linear` benutzen einige Jina-v2-
   Konfigurationen `GeGLU` (`Linear → split → GeLU(a) · b → Linear`).
   Das verdoppelt die Zwischen-Dimension der ersten FFN-Schicht und
   ändert das Gewichtslayout (`intermediate_size * 2` für die erste
   Schicht).
3. Sonst BERT-Standard: gleiches QKV-Layout, gleiche LayerNorm-Positionen,
   gleiches Embedding-Layout (ohne pos-emb).

Heißt: `DirectMlBertEncoderLayerBlock` ist **nicht drop-in**
kompatibel. Es braucht:

- einen ALiBi-Bias-Pfad in der Attention (statt pos-emb-add im Embedding),
- optional einen GeGLU-FFN-Pfad,
- einen JinaBert-spezifischen Architektur-Deskriptor
  (`JinaBertV2Architecture`).

## Sequence-Length-/Pooling-Regeln

- Pooling identisch zu MiniLM/E5 (`MeanPoolingKernel` + `L2NormalizeKernel`),
  keine Änderung nötig.
- `maxSequenceLength`: Jina-v2 verspricht 8 192. Praktisch reicht für
  RAG/Retrieval meist 512–1024. Empfehlung für eine erste Integration:
  Default 512 lassen, Konfig-Knopf für bis zu 2048 freigeben. 8 192 ist
  eher Marketing als notwendig und vervielfacht den DML-Speicherbedarf
  quadratisch in der Attention (selbst mit ALiBi).

## Gewichtslayout

HF-Repository liefert `pytorch_model.bin` + Python-Custom-Code (`trust_remote_code=True`).
Für den Java-Host bedeutet das:

- `safetensors`/`pytorch_model.bin` enthält `bert.encoder.layer.*`-Keys
  weitgehend BERT-kompatibel.
- **Aber:** keine `position_embeddings`-Tensoren; ALiBi-Slopes werden
  zur Laufzeit aus der Konfiguration (`num_attention_heads`) berechnet,
  nicht aus Gewichten gelesen.
- FFN-Layer-Keys können `gated_layers.weight` (GeGLU) statt
  `intermediate.dense.weight` heißen.
- ONNX-Export ist von Jina **nicht offiziell** bereitgestellt; Community-
  Exporte existieren, sind aber nicht garantiert byte-identisch.

→ Für eine produktive Integration ist ein eigener ONNX-Export-Schritt
nötig (oder ein dediziertes Weight-Loading aus `pytorch_model.bin`),
beides außerhalb des aktuellen Phi-3/MiniLM-Aufwand.

## CPU-/DirectML-Aufwand (Schätzung)

| Aufgabe                                                     | Aufwand        | Risiko |
|-------------------------------------------------------------|----------------|--------|
| `JinaWordPieceTokenizer`-Variante (Casing-Flag)             | ≤ 0.5 PT       | gering |
| `JinaBertV2Architecture`-Deskriptor + Weight-Mapping        | 1 PT           | mittel |
| ALiBi-Bias in `DirectMlBertEncoderLayerBlock` (CPU + DML)   | 2–3 PT         | mittel |
| GeGLU-FFN-Variante (CPU + DML, Parity-Test)                 | 2 PT           | mittel |
| ONNX-Export-Pipeline + `download-jina.ps1`                  | 1 PT           | gering |
| Smoke- und Parity-Tests (CPU vs. DML, MiniLM-Stil)          | 1 PT           | gering |
| **Summe**                                                   | **≈ 7–9 PT**   |        |

Das ist deutlich mehr als der reine E5-Pfad (geschätzt 2–3 PT, weil
dort nur der Tokenizer fehlt und die Encoder-Layer identisch zu MiniLM
ist).

## Entscheidung

**Status: not planned für das Maven-Central-Release.**

Begründung:

1. Jina v2 verlangt zwei Architektur-Neuerungen (ALiBi, optional GeGLU),
   die der bestehende generische BERT-Encoder nicht abdeckt.
2. Es existiert kein offizieller ONNX-Export; die Pipeline müsste selbst
   aufgebaut werden.
3. MiniLM und (sobald freigeschaltet) E5 decken die deutsch- und
   mehrsprachigen Embeddings für den Release ausreichend ab.
4. Reranking-Use-Cases sind über die generische Reranker-Runtime mit
   `cross-encoder/ms-marco-MiniLM-L-6-v2` bzw. `bge-reranker` bereits
   abgedeckt – Jina-Reranker bringt keinen funktionalen Mehrwert für
   das Release.

→ `EmbeddingModelRegistry` lässt `jinaai/jina-embeddings-v2-base-de`
weiterhin als `PLANNED` mit der bestehenden „needs custom Jina v2
attention path“-Begründung stehen. Die Einreichung über den Sidecar
liefert weiterhin den klar lesbaren `not implemented`-Fehler.

**Re-Evaluierung** nach dem Release, wenn:

- ALiBi/GeGLU ohnehin für eine andere Modellfamilie (z. B. Llama-FFN-
  Varianten, BGE-M3) gebraucht werden, oder
- Nutzer-Feedback explizit Jina-Embeddings für deutsche Long-Document-
  Retrieval-Cases anfordert.

## Akzeptanzkriterien (Issue 34)

- [x] Tokenizer geprüft (WordPiece, Casing-Flag, vorhandener Pfad reicht).
- [x] Architektur geprüft (JinaBERT v2 = BERT + ALiBi + optional GeGLU).
- [x] Sequence-Length- / Pooling-Regeln geprüft (Pooling unverändert;
      Seq-Len-Default 512, Knopf bis 2048 sinnvoll).
- [x] Gewichtslayout geprüft (HF-PyTorch, kein offizieller ONNX-Export,
      ALiBi-Slopes laufzeitberechnet, GeGLU-Keys abweichend).
- [x] CPU-/DirectML-Aufwand geschätzt (≈ 7–9 PT, ALiBi + GeGLU dominant).
- [x] Entscheidung dokumentiert: **not planned für Maven-Central-Release**
      (Status im Registry bleibt `PLANNED`, Re-Evaluierung Post-Release).
- [x] Kein Blocker für das Maven-Central-Release – die Klassifikation
      ist bereits korrekt verdrahtet (`EmbeddingModelRegistry` →
      `Status.PLANNED`, Sidecar liefert `not implemented`-Antwort).

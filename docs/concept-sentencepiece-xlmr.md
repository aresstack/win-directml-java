# Konzept: SentencePiece-/XLM-R-Unterstützung

Tracking Issue 33 — *Milestone 8: Weitere Modellfamilien nach dem Release*.

## Ziel

Viele moderne mehrsprachige Encoder-Modelle (Embedding **und** Reranker)
verwenden eine **SentencePiece-BPE**-Tokenisierung und einen
**XLM-RoBERTa**-Encoder-Stack. Der aktuelle Encoder-Pfad in
`directml-encoder` deckt nur **WordPiece** (BERT) ab und kann diese
Familie deshalb nicht laden. Dieses Dokument plant die Erweiterung;
eine Umsetzung im **ersten Maven-Central-Release ist nicht** vorgesehen.

## Kandidaten

| Modell                                    | Use-Case  | Architektur            | Tokenizer             | Output-Dim |
|-------------------------------------------|-----------|------------------------|-----------------------|------------|
| `intfloat/multilingual-e5-base`           | Embedding | XLM-RoBERTa-base       | SentencePiece (XLM-R) | 768        |
| `intfloat/multilingual-e5-large`          | Embedding | XLM-RoBERTa-large      | SentencePiece (XLM-R) | 1024       |
| `intfloat/multilingual-e5-large-instruct` | Embedding | XLM-RoBERTa-large      | SentencePiece (XLM-R) | 1024       |
| `danielheinz/e5-base-sts-en-de`           | Embedding | XLM-RoBERTa-base       | SentencePiece (XLM-R) | 768        |
| `BAAI/bge-m3`                             | Embedding | XLM-RoBERTa-large      | SentencePiece (XLM-R) | 1024       |
| `BAAI/bge-reranker-v2-m3`                 | Reranker  | XLM-RoBERTa-large      | SentencePiece (XLM-R) | 1 (Score)  |
| sonstige XLM-R-basierte Reranker          | Reranker  | XLM-RoBERTa-base/large | SentencePiece (XLM-R) | 1 (Score)  |

`multilingual-e5-large-instruct` ist bereits als `Status.PLANNED` in
[
`EmbeddingModelRegistry`](../directml-config/src/main/java/com/aresstack/windirectml/config/models/EmbeddingModelRegistry.java)
hinterlegt und gilt als Leitmodell für diese Linie.

`danielheinz/e5-base-sts-en-de` wurde mit PR #56 von `SHIPPED` (WordPiece)
auf `Status.PLANNED` (SentencePiece / XLM-R) umklassifiziert: die
HuggingFace-Distribution hostet einen `XLMRobertaModel`-Checkpoint
(`vocab=250002`, `type_vocab_size=1`), nicht das WordPiece-BERT-base-Profil,
das die heutige E5-Runtime erwartet. Damit gehört das Modell – trotz
des "E5-base"-Namens – architektonisch in diesen XLM-R-Track und nicht
in den bestehenden WordPiece-E5-Pfad. Insbesondere darf der Name
**nicht** als Indiz dafür gewertet werden, dass eine WordPiece-Variante
existiert; sämtliche E5-multilingual-Checkpoints in dieser Tabelle nutzen
SentencePiece-Unigram (siehe
[SUPPORTED_MODELS.md](../SUPPORTED_MODELS.md)).

## Unterschiede zu BERT/WordPiece (heutiger Encoder-Pfad)

| Aspekt             | BERT (heute)                         | XLM-RoBERTa (neu)                                        |
|--------------------|--------------------------------------|----------------------------------------------------------|
| Tokenizer          | WordPiece (`##`-Suffix, oft uncased) | SentencePiece-Unigram (`▁`-Wortgrenze, cased, NFKC)      |
| Vocab              | ~30 k                                | 250 002 (XLM-R `sentencepiece.bpe.model`)                |
| Spezial-Tokens     | `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`   | `<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>`                |
| Token-IDs Spezial  | 101 / 102 / 0 / 100                  | 0 / 2 / 1 / 3 / 250001                                   |
| Segment-Embeddings | 2 (`token_type_ids` ∈ {0,1})         | 1 (kein NSP, `token_type_ids` werden ignoriert)          |
| Positions-Embed    | absolut, `position_ids` ab 0         | absolut, **`position_ids` ab `padding_idx+1 = 2`**       |
| Pair-Encoding      | `[CLS] A [SEP] B [SEP]`              | `<s> A </s></s> B </s>` (doppeltes `</s>` als Separator) |
| Pooling            | Mean + L2 (E5) / `[CLS]` (BGE BERT)  | Mean + L2 (E5/BGE-m3) / `<s>`-Pooling (BGE-reranker)     |
| Gewichtsnamen      | `bert.encoder.layer.{i}.…`           | `roberta.encoder.layer.{i}.…` bzw. `model.…` (HF)        |

Die GPU-Kernels (LayerNorm, GEMM, Self-Attention, Mean-Pool, L2)
funktionieren **unverändert**: XLM-R ist architektonisch ein RoBERTa,
und RoBERTa nutzt dieselben Bausteine wie BERT. Es geht ausschließlich
um Tokenizer + Embedding-Regeln + Gewichtsmapping.

## Tokenizer-Anforderungen

XLM-R verteilt das Tokenizer-Modell als
`sentencepiece.bpe.model` (Protobuf, Unigram-LM). HuggingFace verpackt
denselben Inhalt zusätzlich in `tokenizer.json`. Eine reine
JSON-Lösung reicht für die ältere E5-Generation (`tokenizer.json` mit
expandierten Vokabular-Tabellen vorhanden), nicht aber für bge-m3
in seiner Roh-Form.

Anforderungen:

- **SentencePiece-Unigram-Decoder** (kein BPE-Merges-Reader nötig):
    - lese Stück-Liste `(piece, score)` aus
      `tokenizer.json` → `model.vocab` (Format `[[piece, score], …]`).
    - lade Spezial-Tokens aus `added_tokens` / `special_tokens_map`.
- **Normalisierung**: NFKC + Leerzeichen → `▁`, Wortanfang erhält `▁`.
- **Segmentierung**: Viterbi-Best-Path über das Unigram-Modell
  (klassisches SentencePiece-Verfahren). Greedy-Longest-Match wie bei
  WordPiece reicht **nicht** und führt zu Token-Drift gegenüber HF.
- **Output**:
    - `inputIds = [<s>, … pieces …, </s>]`
    - `attentionMask = 1` für reale Tokens, `0` für `<pad>`.
    - `tokenTypeIds = null` (XLM-R verwendet keine).
- **Pair-Encoding** (für Reranker):
  `<s> A </s></s> B </s>` mit `tokenTypeIds = null`.

Das passt zum bestehenden
[`EncoderTokenizer`](../directml-encoder/src/main/java/com/aresstack/windirectml/encoder/EncoderTokenizer.java)
-Vertrag: `tokenTypeIds` darf `null` sein, `encodePair` ist optional.

## XLM-R-Embedding-Regeln

Zwei Stolperfallen, die bei BERT nicht existieren:

1. **`position_ids` starten bei `padding_idx + 1 = 2`**, nicht bei 0.
   Der Encoder-Code muss das pro Modell-Familie konfigurierbar machen
   (heute hart auf 0 in
   `directml-encoder/.../bert/...PositionEmbedding`).
2. **Keine `token_type_embeddings`** – das Embedding-Modul lädt nur
   `word_embeddings` + `position_embeddings`. Es gibt zwar ein
   `token_type_embeddings`-Tensor in den Checkpoints, das ist aber
   1×H groß und wird in PyTorch lediglich aufaddiert. Wir können es
   ignorieren (Wert ≈ 0) oder als konstanten Bias laden.

## Gewichtsnamen-Abbildung

```text
HF-Checkpoint (RoBERTa/XLM-R)              →  interner Tensor-Key
roberta.embeddings.word_embeddings.weight  →  embed.word
roberta.embeddings.position_embeddings.w.  →  embed.position
roberta.embeddings.LayerNorm.weight/bias   →  embed.ln.{w,b}
roberta.encoder.layer.{i}.attention.self.{q,k,v}.{weight,bias}
                                           →  layer.{i}.attn.{q,k,v}.{w,b}
roberta.encoder.layer.{i}.attention.output.{dense,LayerNorm}.*
                                           →  layer.{i}.attn.out.{w,b}, layer.{i}.ln1.{w,b}
roberta.encoder.layer.{i}.intermediate.dense.*  →  layer.{i}.mlp.fc1.{w,b}
roberta.encoder.layer.{i}.output.{dense,LayerNorm}.*
                                           →  layer.{i}.mlp.fc2.{w,b}, layer.{i}.ln2.{w,b}
```

Manche neueren bge-Checkpoints verwenden `model.encoder…` statt
`roberta.encoder…` als Präfix – der Loader muss beide Präfixe
akzeptieren.

## Architektur-Erweiterung

Vorgeschlagene Punkte im bestehenden Encoder-Modul (analog zum
`DecoderArchitecture`-Vorschlag aus
[concept-decoder-extensions.md](concept-decoder-extensions.md)):

```java
public interface EncoderArchitecture {
    String name();                  // "bert", "xlm-roberta"
    int positionIdsOffset();        // 0 für BERT, 2 für XLM-R
    boolean usesTokenTypeIds();     // true für BERT, false für XLM-R
    int padTokenId();
    int clsTokenId();               // <s> bei XLM-R
    int sepTokenId();               // </s> bei XLM-R
    String pairSeparatorScheme();   // "single-sep" | "double-sep"
}
```

Der `DirectMlBertEncoderStack` bleibt unverändert; nur Pre-/Post-
Embedding-Logik wird parametrisiert.

## Eigener Tokenizer vs. externer Java-Tokenizer

| Option                                                 | Pro                                                                | Kontra                                                            |
|--------------------------------------------------------|--------------------------------------------------------------------|-------------------------------------------------------------------|
| Eigene Java-Implementierung (Unigram + NFKC)           | keine neue Abhängigkeit, Java-8-kompatibel, byte-identisch testbar | ~600 LoC + Tests, NFKC-Tabellen erforderlich                      |
| `ai.djl.huggingface:tokenizers` (Rust JNI über DJL)    | unterstützt jedes HF-Format inkl. bge-m3 out-of-the-box            | Native Binaries für Windows-x64 zusätzlich shippen, Lizenz prüfen |
| `com.robrua.nlp.deeplearning4j:deeplearning4j-nlp-…`   | reines Java                                                        | uralt, kein Unigram-Decoder, nicht XLM-R-tauglich                 |
| HF `tokenizer.json` per Mini-Reader (kein SP-Protobuf) | reicht für E5-multilingual + BGE-Varianten mit `tokenizer.json`    | greift nicht für reine `sentencepiece.bpe.model`-Distributionen   |

**Empfehlung:** *Eigener Java-Tokenizer*, der `tokenizer.json` (HF-Format)
parst. Begründung: Sidecar und Java-8-Client bleiben dependency-arm,
keine native Bindings, und alle Zielmodelle der Tabelle oben liefern
heute ein `tokenizer.json` mit. Ein optionaler `sentencepiece.bpe.model`-
Pfad kann nachgereicht werden, ist aber nicht release-relevant.

## Aufwandsschätzung

| Arbeitspaket                                           | Aufwand     |
|--------------------------------------------------------|-------------|
| `EncoderArchitecture`-Abstraktion + BERT-Refactor      | 1 PT        |
| `SentencePieceUnigramTokenizer` (inkl. NFKC, Viterbi)  | 2–3 PT      |
| `tokenizer.json`-Reader (Unigram-Variante)             | 1 PT        |
| XLM-R-Embedding-Schicht (`positionIdsOffset = 2`)      | 0.5 PT      |
| Gewichts-Loader (`roberta.*` / `model.*` Präfixe)      | 0.5 PT      |
| Reranker-Pair-Encoding `</s></s>`                      | 0.5 PT      |
| Parity-Tests gegen HF (CPU-Referenz, je Modell)        | 1 PT        |
| Integration in Sidecar + Registry + Workbench-Dropdown | 1 PT        |
| Dokumentation + Smoke-Run                              | 0.5 PT      |
| **Summe**                                              | **~8–9 PT** |

Skaliert bge-m3 (gleicher Stack) und bge-reranker-v2-m3 mit
vernachlässigbarem Zusatzaufwand (nur Pooling-/Head-Wahl).

## Abgrenzung

- Kein universeller HuggingFace-Tokenizer-Klon. Nur Unigram (XLM-R-Linie).
- Kein BPE-Merges-Reader (wird erst für Llama/Qwen-Decoder relevant –
  siehe [concept-decoder-extensions.md](concept-decoder-extensions.md)).
- Keine Quantisierung von XLM-R-large – orthogonal, siehe Roadmap in
  [SUPPORTED_MODELS.md](../SUPPORTED_MODELS.md).

## Planungsstatus

- Konzept: dieses Dokument.
- Implementierung: **nicht** im ersten Maven-Central-Release. Wird nach
  dem Release als Milestone-8-Arbeitspaket aufgesetzt, sobald der
  WordPiece-/BERT-Encoder-Pfad stabil im Feld ist.
- Tracking: Issue 33.

# Konzept: E5-Familienadapter

Tracking Issue 23.

## Ziel

Nach `sentence-transformers/all-MiniLM-L6-v2` soll die E5-Familie unterstützt
werden.

## Zielmodelle (initial)

- `danielheinz/e5-base-sts-en-de`
- `intfloat/multilingual-e5-large-instruct`

## Unterschiede zu MiniLM

| Aspekt       | MiniLM-L6                 | E5-base / e5-large                                   |
|--------------|---------------------------|------------------------------------------------------|
| Architektur  | BERT-tiny, 6 Layer, 384-d | BERT/XLM-R-style, 12 Layer (base) / 24 Layer (large) |
| Output-Dim   | 384                       | 768 (base), 1024 (large)                             |
| Tokenizer    | WordPiece (uncased)       | XLM-R-SentencePiece (BPE)                            |
| Pooling      | Mean + L2                 | Mean + L2 (identisch zu MiniLM)                      |
| Input-Format | Roh-Text                  | mit Prefix: `"query: ..."`, `"passage: ..."`         |
| Sprache      | EN only                   | mehrsprachig (E5-multilingual)                       |

## Tokenizer-Anforderungen

E5 benutzt **SentencePiece-BPE**, nicht WordPiece. Das aktuelle
`WordPieceTokenizer` reicht nicht. Erforderlich:

- `SentencePieceTokenizer`-Implementierung
    - lese SentencePiece-Modell (`tokenizer.json` enthält bei HF die Tabellen)
    - Greedy Longest-Match wie WordPiece, aber mit ▁ als Wortgrenze
- Spezial-Tokens: `<s>`, `</s>`, `<pad>`, `<unk>`
- Optional: Lower-Case nicht erzwingen (E5 ist cased)

## Pooling-Strategie

Identisch zu MiniLM:

```text
mean(token_embeddings * attention_mask) / sum(attention_mask)
↓
L2-Normalisieren
```

Kann denselben `MeanPoolingKernel`/`L2NormalizeKernel` verwenden.

## Input-Prefixe

E5-Modelle erwarten:

```text
embed_query:    "query: " + text
embed_passage:  "passage: " + text
```

Der Prefix wird **vor** der Tokenisierung angefügt. Der
`EmbeddingRequest.prefix` ist dafür bereits vorgesehen.

## Runtime-Erweiterungen

- Neuer Tokenizer-Pfad (SentencePiece) – wiederverwendbar für Llama-Familie.
- Optional: GQA (multilingual-e5-large nutzt es nicht).
- Längeres `maxSequenceLength` (typisch 512).

## Planungsstatus

- Architektur-Deskriptor `E5Architecture` noch nicht angelegt.
- Migration wird erst geplant, wenn MiniLM end-to-end läuft.
- Kein Implementierungs-Druck in der aktuellen Phase.


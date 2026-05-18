# `directml-encoder`

Eigenständiges Modul für **Sentence/Text-Embeddings** auf Windows DirectML.

## Ziel

Encoder-Modelle (MiniLM, E5, JinaBERT, …) bekommen eine **eigene Runtime**,
getrennt vom Phi-3-Decoder. Encoder haben keinen KV-Cache, kein Chat-Template
und kein Streaming – nur `text → fester Vektor`.

## Aktueller Stand

Nur das API-Grundgerüst (Issue 14):

```
EmbeddingModel
EmbeddingRequest / EmbeddingVector / EmbeddingException
EncoderArchitecture
EncoderWeights
EncoderTokenizer
EncoderRuntime
PoolingStrategy
minilm/MiniLmArchitecture (Deskriptor, ohne Implementierung)
```

## Roadmap

| Issue | Status        | Inhalt                                                          |
|-------|---------------|-----------------------------------------------------------------|
| 14    | ✅ done        | API-Grundgerüst, dieses Modul                                   |
| 15    | ⏳ in progress | `MiniLmArchitecture`-Deskriptor angelegt, Implementierung offen |
| 11    | ⏳ todo        | DirectML-Runtime-Core aus Phi-3 extrahieren                     |
| 12    | ⏳ todo        | Tensor- und Buffer-Abstraktion                                  |
| 13    | ⏳ todo        | Wiederverwendbare Kernel (MatMul, LayerNorm, GELU, …)           |
| 16    | ⏳ todo        | `SafetensorsReader`                                             |
| 17    | ⏳ todo        | WordPiece-Tokenizer für BERT-artige Encoder                     |
| 18    | ⏳ todo        | Mean Pooling + L2-Normalisierung                                |
| 19    | ⏳ todo        | `embed`-Methode im Sidecar aktivieren                           |

## Zielmodell

`sentence-transformers/all-MiniLM-L6-v2` (384-d, 6 Layer, BERT-tiny).

Solange der Encoder nicht steht, antwortet der Sidecar auf `embed` mit
`-32005 Not implemented` (siehe `directml-sidecar/PROTOCOL.md`).


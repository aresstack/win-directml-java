# Supported Models Matrix

Diese Matrix ist als Arbeitsliste für die Implementierung gedacht.  
Status bitte direkt per Checkbox pflegen.

## 1. LLM / Driver

| Bereich | Modell | Zweck | Format | Quelle | Default | V1/V2 | Status |
|---|---|---|---|---|---|---|---|
| LLM Driver | Phi-3-mini-4k-instruct | Lokaler Test des reinen Java-/DirectML-Drivers | ONNX (Gewichtscontainer, keine ORT-Runtime) | https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx | Ja | V1 | [ ] |

### Phi-3-Teilaufgaben

- [ ] Modellgewichte und Metadaten aus dem ONNX-Container lesbar
- [ ] Prefill-Pfad implementiert
- [ ] Decode-Loop implementiert
- [ ] Greedy Decoding implementiert
- [ ] KV-Cache implementiert
- [ ] Mindestens 1 Token wird deterministisch generiert
- [ ] Läuft ohne ORT / JNA / JNI / Third-Party-DLLs
- [ ] Läuft nur mit Java 21 + FFM + Windows-DLLs

---

## 2. Lokale Embedding-Modelle

| Bereich | Modellname in Settings | Technische Quelle | Format | Lokaler Pfad im Repo | Besonderheiten | Default | Status |
|---|---|---|---|---|---|---|---|
| Embedding | all-minilm | sentence-transformers/all-MiniLM-L6-v2 | ONNX | model/embeddings/all-minilm/ | Tokenizer + Pooling beachten | Ja | [ ] |
| Embedding | nomic-embed-text | nomic-ai/nomic-embed-text-v1.5 | ONNX | model/embeddings/nomic-embed-text/ | Query-/Dokument-Präfixe unterstützen | Nein | [ ] |

### Embedding-Quellen

- all-minilm
  - Root: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
  - ONNX: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main/onnx
  - Alternative kompakte ONNX-Port-Quelle: https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx

- nomic-embed-text
  - Root: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
  - ONNX: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/tree/main/onnx

### Embedding-Implementierungscheckliste

#### all-minilm
- [ ] Modell lokal abgelegt
- [ ] tokenizer.json eingebunden
- [ ] input_ids + attention_mask korrekt erzeugt
- [ ] ONNX-Modell lokal ausführbar
- [ ] Embedding-Vektor korrekt extrahiert
- [ ] Dimensions-Check vorhanden
- [ ] Determinismus-Test vorhanden
- [ ] Regressionstest grün

#### nomic-embed-text
- [ ] Modell lokal abgelegt
- [ ] tokenizer.json eingebunden
- [ ] Präfix `search_query:` unterstützt
- [ ] Präfix `search_document:` unterstützt
- [ ] Query-Embedding testbar
- [ ] Dokument-Embedding testbar
- [ ] Determinismus-Test vorhanden
- [ ] Regressionstest grün

---

## 3. Lokale Rerank-Modelle

| Bereich | Modellname in Settings | Technische Quelle | Format | Lokaler Pfad im Repo | Besonderheiten | Priorität | Status |
|---|---|---|---|---|---|---|---|
| Rerank | BAAI/bge-reranker-base | BAAI/bge-reranker-base | ONNX | model/rerank/bge-reranker-base/ | Standard-Cross-Encoder | Hoch | [ ] |
| Rerank | BAAI/bge-reranker-v2-m3 | onnx-community/bge-reranker-v2-m3-ONNX | ONNX | model/rerank/bge-reranker-v2-m3/ | Community-ONNX-Port kennzeichnen | Hoch | [ ] |
| Rerank | jinaai/jina-reranker-v2-base-multilingual | jinaai/jina-reranker-v2-base-multilingual | ONNX | model/rerank/jina-reranker-v2-base-multilingual/ | multilingual | Mittel | [ ] |
| Rerank | BAAI/bge-reranker-large | BAAI/bge-reranker-large | ONNX | model/rerank/bge-reranker-large/ | größeres Modell | Niedrig | [ ] |
| Rerank | BAAI/bge-reranker-v2-gemma | BAAI/bge-reranker-v2-gemma | safetensors | nicht in V1 | Nicht V1 | Ausgeschlossen | [ ] Nicht V1 |

### Rerank-Quellen

- BAAI/bge-reranker-base
  - Root: https://huggingface.co/BAAI/bge-reranker-base
  - ONNX: https://huggingface.co/BAAI/bge-reranker-base/tree/main/onnx

- BAAI/bge-reranker-large
  - Root: https://huggingface.co/BAAI/bge-reranker-large
  - ONNX: https://huggingface.co/BAAI/bge-reranker-large/tree/main/onnx

- BAAI/bge-reranker-v2-m3
  - Root: https://huggingface.co/BAAI/bge-reranker-v2-m3
  - ONNX-Port: https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX
  - ONNX-Dateien: https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX/tree/main/onnx

- jinaai/jina-reranker-v2-base-multilingual
  - Root: https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual
  - ONNX: https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/tree/main/onnx

- BAAI/bge-reranker-v2-gemma
  - Root: https://huggingface.co/BAAI/bge-reranker-v2-gemma
  - Hinweis: Nicht Teil von V1

### Rerank-Implementierungscheckliste

#### BAAI/bge-reranker-base
- [ ] Modell lokal abgelegt
- [ ] Query/Dokument-Paar-Tokenisierung implementiert
- [ ] ONNX-Modell lokal ausführbar
- [ ] Score/Logit korrekt lesbar
- [ ] Mehrere Kandidaten rankbar
- [ ] Determinismus-Test vorhanden
- [ ] Regressionstest grün

#### BAAI/bge-reranker-v2-m3
- [ ] Modell lokal abgelegt
- [ ] Community-ONNX-Port dokumentiert
- [ ] Query/Dokument-Paar-Tokenisierung implementiert
- [ ] ONNX-Modell lokal ausführbar
- [ ] Score/Logit korrekt lesbar
- [ ] Mehrere Kandidaten rankbar
- [ ] Regressionstest grün

#### jinaai/jina-reranker-v2-base-multilingual
- [ ] Modell lokal abgelegt
- [ ] multilingual Tokenisierung validiert
- [ ] ONNX-Modell lokal ausführbar
- [ ] Score/Logit korrekt lesbar
- [ ] Mehrsprachiger Testfall vorhanden
- [ ] Regressionstest grün

#### BAAI/bge-reranker-large
- [ ] Modell lokal abgelegt
- [ ] Query/Dokument-Paar-Tokenisierung implementiert
- [ ] ONNX-Modell lokal ausführbar
- [ ] Score/Logit korrekt lesbar
- [ ] Ressourcenverbrauch dokumentiert
- [ ] Regressionstest grün

---

## 4. Gemeinsame technische Anforderungen

- [ ] Nur lokale Modelle
- [ ] Keine Cloud-Fallbacks
- [ ] ONNX-first für Embeddings und Reranking
- [ ] Keine generische Runtime für beliebige Modelle
- [ ] Modell-Registry vorhanden
- [ ] Embedding- und Rerank-API fachlich getrennt
- [ ] Alle Modellpfade liegen reproduzierbar im Repo oder in dokumentierten Download-Skripten
- [ ] Clone -> Build -> Test funktioniert reproduzierbar
- [ ] DirectML-/Windows-Pfad bleibt gekapselt
- [ ] Keine nativen Details außerhalb des Driver-Moduls

---

## 5. Empfohlene Umsetzungsreihenfolge

1. [ ] Phi-3-mini-4k-instruct Driver fertig
2. [ ] all-minilm
3. [ ] nomic-embed-text
4. [ ] BAAI/bge-reranker-base
5. [ ] BAAI/bge-reranker-v2-m3
6. [ ] jinaai/jina-reranker-v2-base-multilingual
7. [ ] BAAI/bge-reranker-large


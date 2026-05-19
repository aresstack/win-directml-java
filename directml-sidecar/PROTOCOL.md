# JSON-RPC-Protokoll des `directml-sidecar`

Dieser Sidecar spricht **JSON-RPC 2.0** über `stdin` und `stdout`.
Jede Zeile auf stdin ist genau eine Request, jede Zeile auf stdout ist
genau eine Response oder Notification. Logs und Stacktraces gehen
ausschließlich nach `stderr`.

## Transportregeln

```text
stdin  = JSON-RPC Requests   (eine pro Zeile, UTF-8)
stdout = JSON-RPC Responses und Notifications (eine pro Zeile, UTF-8)
stderr = Logs, Debug-Ausgaben, Stacktraces
```

* Eine JSON-Zeile = genau eine Nachricht.
* Auf stdout ist **nichts anderes** als JSON-RPC erlaubt.
* Leere Zeilen werden ignoriert.
* Nicht parsbare Zeilen erzeugen eine Error-Response mit
  `code = -32700` (Parse error), `id = null`.

## Nachrichten-Typen

### Request

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "summarize",
  "params": {
    ...
  }
}
```

* `id`: String oder Zahl. Fehlt `id` oder ist `id = null` → Notification
  (keine Response wird vom Sidecar geschrieben).

### Response (Erfolg)

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "result": {
    ...
  }
}
```

### Response (Fehler)

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "error": {
    "code": -32601,
    "message": "Method not found: foo"
  }
}
```

### Notification (Sidecar → Client)

```json
{
  "jsonrpc": "2.0",
  "method": "sidecar.modelLoaded",
  "params": {
    "loadTimeMs": 2840,
    "backend": "directml"
  }
}
```

## Fehlercodes

| Code     | Bedeutung           | Quelle                            |
|----------|---------------------|-----------------------------------|
| `-32700` | Parse error         | JSON konnte nicht gelesen werden  |
| `-32600` | Invalid Request     | kein gültiges JSON-RPC-2.0-Objekt |
| `-32601` | Method not found    | unbekannte Methode                |
| `-32602` | Invalid params      | Parameterstruktur ungültig        |
| `-32603` | Internal error      | unerwartete Exception             |
| `-32001` | Model not ready     | Modell noch nicht geladen         |
| `-32002` | Generation failed   | Inferenz-Fehler                   |
| `-32003` | Shutting down       | Sidecar fährt herunter            |
| `-32004` | Cancelled           | Auftrag wurde abgebrochen         |
| `-32005` | Not implemented     | Methode ist Platzhalter           |
| `-32006` | Unsupported backend | Backend wird nicht unterstützt    |

## Methoden

### `health`

Liefert den aktuellen Sidecar-Status. **Vor** dem Modell-Load aufrufbar.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": "h1",
  "method": "health",
  "params": {}
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": "h1",
  "result": {
    "status": "ok",
    "ready": true,
    "busy": false,
    "modelLoaded": true,
    "shuttingDown": false,
    "mode": "phi-3 (auto)"
  }
}
```

`status` ∈ `starting | ok | shutting_down`.

### `summarize`

Erzeugt eine Zusammenfassung mit dem Phi-3-Modell.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": "s1",
  "method": "summarize",
  "params": {
    "text": "Langer Eingabetext...",
    "maxTokens": 256,
    "systemPrompt": "optional, überschreibt Default-Prompt"
  }
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": "s1",
  "result": {
    "text": "Zusammenfassung...",
    "finishReason": "end_turn",
    "promptTokens": 312,
    "outputTokens": 84,
    "elapsedMs": 4123
  }
}
```

`finishReason` ∈ `end_turn | max_tokens | error`.

### `embed`

Der Sidecar unterstützt zwei BERT-style Encoder-Familien hinter demselben
JSON-RPC-Endpunkt. Welche Familie geladen wird, steuert das Systemproperty
`-Dembed.model`:

| Property                        | Werte                                                          | Default          | Wirkung                                                                                                                                                                                       |
|---------------------------------|----------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-Dembed.model`                 | `minilm`, `e5`                                                 | `minilm`         | Wählt die Encoder-Familie. Unbekannte Werte ⇒ Exit-Code `2`.                                                                                                                                  |
| `-Dminilm.modelDir`             | Pfad                                                           | auto-discovery   | Pfad zum MiniLM-Modellordner. Auto: `model/all-MiniLM-L6-v2/`.                                                                                                                                |
| `-De5.model`                    | `small-v2`, `base-v2`, `large-v2`, `base-sts-en-de`            | `base-sts-en-de` | E5-Variante. Pinnt `BertEncoderConfig` (hiddenSize/numLayers/…). Unbekannt ⇒ Exit-Code `2`.                                                                                                   |
| `-De5.modelDir`                 | Pfad                                                           | auto-discovery   | Pfad zum E5-Modellordner. Auto: variant-spezifische Hints (z. B. `model/e5-base-sts-en-de/`).                                                                                                 |
| `-Dembed.backend`               | `auto`, `directml`, `cpu`                                      | `auto`           | Wählt das Backend innerhalb der Familie. `directml`/`cpu` ⇒ Exit-Code `3` bei Fehlern; `auto` fällt sauber auf CPU zurück und schreibt die Warnung in `health.lastError`. Unbekannt ⇒ Exit `2`. |

Für E5 ist `config.json` im Modellordner **Pflicht** – die Datei wird gegen
die gewählte Variante geprüft (`hidden_size`, `num_hidden_layers`,
`num_attention_heads`, `intermediate_size`, `vocab_size`, `type_vocab_size`).
Eine Diskrepanz zwischen `-De5.model` und der `config.json` ist ein harter
Fehler – kein stilles Re-Shape.

Innerhalb einer Familie steuert `-Dembed.backend`, welches konkrete Backend
verwendet wird:

| Modus                            | Verhalten                                                                                                                                                                                                  |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-Dembed.backend=cpu`            | Reine Java-CPU-Variante erzwingen (`CpuMiniLmEncoder` bzw. `CpuBertEncoder` für E5). Fehler beim Laden ⇒ Sidecar beendet sich mit Exit-Code `3` (sichtbarer Fehler).                                       |
| `-Dembed.backend=directml`       | DirectML-Variante erzwingen (`DirectMlMiniLmEncoder` bzw. `DirectMlBertEncoder` für E5). Wenn DirectML nicht verfügbar ist (kein Windows/D3D12, keine kompatible Karte), Exit-Code `3`. Kein stiller Fallback. |
| `-Dembed.backend=auto` (Default) | Erst DirectML versuchen, bei Fehler sauber auf CPU zurückfallen. Die Fallback-Warnung steht auf `stderr`/Log und ist als `lastError` im `health`-Result sichtbar.                                          |

Fehlt das Modellverzeichnis komplett (weder die Override-Property
`-Dminilm.modelDir` / `-De5.modelDir` noch ein Auto-Discovery-Pfad),
gilt dieselbe Regel: `cpu`/`directml` ⇒ Exit-Code `3`, `auto` ⇒ Sidecar
startet weiter, aber `embed` antwortet `-32005 Not implemented`.

Der aktiv geladene Backend-Name wird in `health` ausgegeben:

```json
{
  "embeddingBackend": "directml",   // oder "cpu" / "none" / "error"
  "embeddingReady": true
}
```

Request:

```json
{
  "jsonrpc": "2.0",
  "id": "e1",
  "method": "embed",
  "params": { "text": "...", "normalize": true, "prefix": null }
}
```

`prefix` ist optional. Für E5 erwartet die Familie konventionsgemäß
`"query: "` für Suchanfragen und `"passage: "` für indizierte Dokumente
(siehe `E5Prefixes.QUERY` / `E5Prefixes.PASSAGE`). Beispiel:

```json
{ "jsonrpc": "2.0", "id": "e2", "method": "embed",
  "params": { "text": "Welche Hauptstadt hat Frankreich?",
              "normalize": true, "prefix": "query: " } }
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": "e1",
  "result": {
    "vector": [0.012, -0.083, ...],
    "dimension": 384,
    "model": "all-MiniLM-L6-v2",
    "normalized": true
  }
}
```

Solange kein Embedding-Modell gefunden wird, antwortet `embed` mit
`-32005 Not implemented`.

### `embedBatch`

Bettet **mehrere Texte in einem Aufruf** ein. Trägt das geladene
Backend Bucket-Batching (z. B. `DirectMlBertEncoder`), werden alle
Texte einer Pad-Länge in einem einzigen GPU-Dispatch verarbeitet – das
ist der primäre Hebel für RAG-Ingestion und Multi-Chunk-Encoding.
Backends ohne Batch-Override (z. B. CPU-Encoder) fallen transparent auf
N sequentielle `embed`-Aufrufe zurück; die Antwort-Struktur ist
identisch.

`normalize` und `prefix` gelten **für alle Texte gemeinsam** – pro-Text
Varianten sind aktuell nicht vorgesehen (Use-Case: gleichartige Chunks).

Request:

```json
{
  "jsonrpc": "2.0",
  "id": "eb1",
  "method": "embedBatch",
  "params": {
    "texts":     ["DirectML is a low-level Windows API for ML.",
                  "Cross-encoders score (query, document) pairs.",
                  "Python is a programming language."],
    "normalize": true,
    "prefix":    "passage: "
  }
}
```

- `texts` (string[], Pflicht) – nicht-leeres Array, keine leeren/blank Einträge.
- `normalize` (boolean, optional, Default `true`) – L2-Normalisierung pro Vektor.
- `prefix` (string, optional) – wird vor jedem Text appended (E5-Konvention).

Response:

```json
{
  "jsonrpc": "2.0",
  "id": "eb1",
  "result": {
    "vectors":   [[0.012, -0.083, ...],
                  [0.041,  0.117, ...],
                  [-0.005, 0.092, ...]],
    "dimension": 384,
    "model":     "all-MiniLM-L6-v2",
    "normalized": true,
    "count":     3
  }
}
```

Die Reihenfolge der Vektoren entspricht der Reihenfolge in `texts`.
Solange kein Embedding-Modell geladen ist, antwortet `embedBatch`
mit `-32005 Not implemented`.

### `rerank`

Cross-Encoder-Reranking: bewertet `(query, document)`-Paare gemeinsam
durch einen BERT-Cross-Encoder mit Classification-Head und liefert eine
nach Score absteigend sortierte Liste zurück. Default-Familie:
`cross-encoder/ms-marco-MiniLM-L-6-v2`. Tokenisierung erfolgt
paarweise (`[CLS] query [SEP] document [SEP]` mit
`token_type_ids = [0…0, 1…1]`).

Konfiguration:

```text
-Drerank.modelDir=<path>     # Override für das Modellverzeichnis
-Drerank.backend=auto|directml|cpu   # Default auto: DirectML mit CPU-Fallback
```

Validierung der `documents`-Liste: leere Arrays und leere/blank Einträge
werden mit `INVALID_PARAMS` abgelehnt. Bei `rerank.backend=auto` und einem
DirectML-Initialisierungsfehler fällt der Sidecar sichtbar auf CPU zurück
(`health.rerankerBackend = "cpu"`,
`health.lastError = "rerank.backend=auto fell back to cpu: …"`).

Request:

```json
{
  "jsonrpc": "2.0",
  "id": "x",
  "method": "rerank",
  "params": {
    "query": "What is DirectML?",
    "documents": [
      "DirectML is a low-level Windows API for ML.",
      "Python is a programming language.",
      "Cross-encoders score (query, document) pairs."
    ],
    "topN": 2
  }
}
```

- `query` (string, Pflicht) – Suchanfrage, darf nicht leer sein.
- `documents` (string[], Pflicht) – Kandidatenliste, mindestens ein Eintrag.
- `topN` (int, optional) – Maximalzahl zurückgegebener Ergebnisse;
  `0` oder Werte ≥ `documents.length` bedeuten "alle".

Response:

```json
{
  "jsonrpc": "2.0",
  "id": "x",
  "result": {
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "results": [
      { "index": 0, "score": 8.123 },
      { "index": 2, "score": 1.456 }
    ]
  }
}
```

`score` ist der rohe Classifier-Logit (höher = relevanter). Cross-Encoder-
Logits sind nicht modellübergreifend kalibriert; Vergleiche sind nur
innerhalb desselben Modells sinnvoll. `index` bezieht sich auf die
ursprüngliche Reihenfolge in `documents` – der Host re-resolved den Text.

Solange kein Reranker-Modell gefunden wird, antwortet `rerank` mit
`-32005 Not implemented`.

### `shutdown`

Setzt den Sidecar in den Shutdown-Modus. Nach der Response wird die
Hauptschleife verlassen und der Prozess beendet sich mit Exit-Code 0.

Response:

```json
{
  "jsonrpc": "2.0",
  "id": "x",
  "result": {
    "accepted": true,
    "status": "shutting_down"
  }
}
```

### `cancel`

Platzhalter. Solange Phi-3 synchron läuft, gibt es nichts zu canceln –
die Methode bestätigt nur Empfang.

Response:

```json
{
  "jsonrpc": "2.0",
  "id": "c",
  "result": {
    "accepted": false,
    "reason": "no_active_request"
  }
}
```

## Notifications

| Methode                   | Bedeutung                               |
|---------------------------|-----------------------------------------|
| `sidecar.started`         | Sidecar ist gestartet, Protokoll bereit |
| `sidecar.modelLoaded`     | Modell geladen, `ready = true`          |
| `sidecar.modelLoadFailed` | Modell-Load fehlgeschlagen              |

## Lebenszyklus

```text
Process Start
   ↓
sidecar.started notification
   ↓
(parallel) Modell-Load     ──→ sidecar.modelLoaded / sidecar.modelLoadFailed
   ↓
Dispatch-Loop (Requests bedient, health bereits beantwortet)
   ↓
shutdown-Request          ──→ Response + Loop verlässt
   ↓
Cleanup (Phi-3-Engine, GPU-Ressourcen)
   ↓
exit 0
```


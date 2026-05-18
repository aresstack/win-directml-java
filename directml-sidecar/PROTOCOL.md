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

Wenn ein MiniLM-Modell unter `model/all-MiniLM-L6-v2/` (oder via
`-Dminilm.modelDir=<path>`) gefunden wird, lädt der Sidecar einen
Embedding-Encoder. Welche Variante geladen wird, steuert das
Systemproperty `-Dembed.backend`:

| Modus                            | Verhalten                                                                                                                                                                                   |
|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-Dembed.backend=cpu`            | `CpuMiniLmEncoder` erzwingen. Fehler beim Laden ⇒ Sidecar beendet sich mit Exit-Code `3` (sichtbarer Fehler).                                                                               |
| `-Dembed.backend=directml`       | `DirectMlMiniLmEncoder` erzwingen. Wenn DirectML nicht verfügbar ist (kein Windows/D3D12, keine kompatible Karte), beendet sich der Sidecar mit Exit-Code `3`. **Keinen** stillen Fallback. |
| `-Dembed.backend=auto` (Default) | Erst DirectML versuchen, bei Fehler sauber auf CPU zurückfallen. Die Fallback-Warnung wird auf `stderr`/Log geschrieben und ist als `lastError` im `health`-Result sichtbar.                |

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

Solange kein MiniLM-Modell gefunden wird, antwortet `embed` mit
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


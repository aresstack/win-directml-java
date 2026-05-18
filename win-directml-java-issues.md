# Issue-Backlog für `win-directml-java`

Dieses Dokument beschreibt die Issues, die wir für das neue Repository `win-directml-java` anlegen wollen.

Ziel des Projekts:

```text
Java-8-Hauptanwendung
  ↓ JSON-RPC über stdin/stdout
Java-21-DirectML-Sidecar
  ↓ Java FFM
Windows 11 DirectML / D3D12 / DXGI
```

Nicht-Ziele:

```text
Kein ACP
Kein MCP
Keine ONNX Runtime
Kein JNA
Kein JNI
Kein universeller ONNX-/GGUF-/Transformers-Runner
Keine Java-CPU-Transformer-Runtime
```

---

## Vorgeschlagene Labels

```text
type:architecture
type:build
type:protocol
type:runtime
type:directml
type:inference
type:embedding
type:summarizer
type:testing
type:documentation
priority:high
priority:medium
priority:low
```

---

# Milestone 1: Repository bereinigen und lauffähig machen

## Issue 1: Repository-Struktur für `win-directml-java` finalisieren

**Labels:** `type:architecture`, `type:build`, `priority:high`

### Ziel

Das neue Repository soll nur noch den DirectML-/Inference-relevanten Code enthalten.

### Aufgaben

- Entferne alle ACP-spezifischen Module und Klassen.
- Entferne alle MCP-spezifischen Module und Klassen.
- Entferne Agent-/Graph-spezifische Altlasten.
- Behalte nur DirectML-, Windows-Binding-, Runtime-, Inference- und Sidecar-Code.
- Prüfe die Modulstruktur.

### Zielstruktur

```text
directml-config
directml-windows-bindings
directml-inference
directml-sidecar
```

### Akzeptanzkriterien

- Das Projekt enthält keine ACP-Packages mehr.
- Das Projekt enthält keine MCP-Packages mehr.
- Die Modulnamen spiegeln den DirectML-Zweck wider.
- Das Repository ist verständlich als eigenständiges DirectML-Java-Projekt erkennbar.

---

## Issue 2: Package-Namen auf `com.aresstack.windirectml` vereinheitlichen

**Labels:** `type:architecture`, `type:build`, `priority:high`

### Ziel

Alle Java-Packages sollen den neuen Projektnamen widerspiegeln.

### Aufgaben

- Vereinheitliche alle Package-Namen auf `com.aresstack.windirectml`.
- Entferne alte Package-Bezüge auf ACP oder Agenten.
- Passe Imports, Modulnamen und Testklassen an.

### Akzeptanzkriterien

- Kein Java-Code verwendet mehr alte ACP-Package-Namen.
- Alle Module kompilieren mit den neuen Package-Namen.
- Die Paketstruktur ist fachlich nachvollziehbar.

---

## Issue 3: Gradle-Build mit Java 21 stabilisieren

**Labels:** `type:build`, `priority:high`

### Ziel

Das Projekt soll lokal mit Java 21 gebaut werden können.

### Aufgaben

- Prüfe `settings.gradle`.
- Prüfe alle `build.gradle`-Dateien.
- Konfiguriere Java 21 als Toolchain.
- Aktiviere notwendige Preview-/Native-Access-Optionen für FFM.
- Entferne nicht mehr benötigte Abhängigkeiten.
- Dokumentiere den Build-Befehl.

### Erwarteter Build-Befehl

```bash
./gradlew clean build
```

### Akzeptanzkriterien

- `./gradlew clean build` läuft lokal durch.
- Der Build verwendet Java 21.
- Nicht benötigte ACP-/MCP-Abhängigkeiten sind entfernt.
- Die Preview-/Native-Access-Optionen sind dokumentiert.

---

## Issue 4: README für Projektziel und Nicht-Ziele erstellen

**Labels:** `type:documentation`, `priority:high`

### Ziel

Das Repository soll direkt erklären, was es ist und was es nicht ist.

### README-Inhalte

- Projektziel
- Architekturübersicht
- Java-8-App plus Java-21-Sidecar
- Windows-11-DirectML-Ziel
- Keine ONNX Runtime
- Kein JNA/JNI
- Kein ACP/MCP
- Aktueller Stand: Phi-3-Summarizer
- Nächster Fokus: Encoder-Embeddings
- Build-Anleitung
- Start-Anleitung für den Sidecar

### Akzeptanzkriterien

- README beschreibt die Zielarchitektur.
- README enthält klare Nicht-Ziele.
- README enthält Beispielbefehle für Build und Start.
- Neue Entwickler verstehen den Projektzweck ohne alten Kontext.

---

# Milestone 2: Sidecar-Protokoll definieren

## Issue 5: JSON-RPC-2.0-Protokoll über stdin/stdout spezifizieren

**Labels:** `type:protocol`, `type:architecture`, `priority:high`

### Ziel

Die Kommunikation zwischen Java-8-App und Java-21-Sidecar soll klar standardisiert werden.

### Transportregeln

```text
stdin  = JSON-RPC Requests
stdout = JSON-RPC Responses und Notifications
stderr = Logs, Debug-Ausgaben, Stacktraces
```

### Aufgaben

- Definiere JSON-RPC Request-/Response-Modelle.
- Definiere Error-Response-Modell.
- Definiere Streaming-Notifications.
- Definiere Methodenliste.
- Dokumentiere, dass stdout ausschließlich Protokollnachrichten enthalten darf.

### Initiale Methoden

```text
health
summarize
embed
shutdown
cancel
```

### Akzeptanzkriterien

- Ein Markdown-Dokument beschreibt das Protokoll.
- Request, Response, Error und Streaming sind mit Beispielen dokumentiert.
- Logs auf stdout sind ausdrücklich verboten.
- Die Java-8-App kann das Protokoll eindeutig implementieren.

---

## Issue 6: JSON-RPC-Basismodelle implementieren

**Labels:** `type:protocol`, `priority:high`

### Ziel

Der Sidecar bekommt typisierte Java-Modelle für JSON-RPC.

### Klassen

```text
JsonRpcRequest
JsonRpcResponse
JsonRpcError
JsonRpcNotification
JsonRpcMessageReader
JsonRpcMessageWriter
```

### Aufgaben

- Implementiere immutable Request-/Response-Modelle.
- Implementiere zeilenbasiertes Lesen von stdin.
- Implementiere zeilenbasiertes Schreiben nach stdout.
- Schreibe Logs ausschließlich nach stderr.
- Füge einfache Unit-Tests für Serialization/Deserialization hinzu.

### Akzeptanzkriterien

- Eine JSON-Zeile wird exakt als eine Nachricht behandelt.
- Fehlerhafte JSON-Nachrichten erzeugen einen JSON-RPC-Fehler.
- Writer schreibt genau eine JSON-Nachricht pro Zeile.
- Keine Log-Ausgaben landen auf stdout.

---

## Issue 7: Sidecar-Command-Dispatcher implementieren

**Labels:** `type:protocol`, `type:runtime`, `priority:high`

### Ziel

Der Sidecar soll eingehende JSON-RPC-Methoden sauber an Use Cases weiterleiten.

### Aufgaben

- Implementiere `SidecarCommandDispatcher`.
- Registriere Handler für `health`, `summarize`, `embed`, `shutdown`, `cancel`.
- Trenne Protokoll-Layer von Inference-Layer.
- Führe klare Fehlercodes ein.

### Akzeptanzkriterien

- Unbekannte Methoden liefern JSON-RPC `method not found`.
- Fachliche Fehler liefern definierte Runtime-Fehler.
- Der Dispatcher kennt keine DirectML-Details.
- Der Dispatcher ist isoliert testbar.

---

# Milestone 3: Phi-3-Summarizer als bestehende Funktion stabilisieren

## Issue 8: Vorhandenen Phi-3-Code als `Summarizer` kapseln

**Labels:** `type:summarizer`, `type:inference`, `priority:high`

### Ziel

Der bestehende Phi-3-DirectML-Code bleibt zunächst als Summarizer erhalten.

### Fachliche API

```java
public interface Summarizer {

    Summary summarize(SummaryRequest request) throws InferenceException;
}
```

### Aufgaben

- Erzeuge `Summarizer`-Interface.
- Erzeuge `SummaryRequest`.
- Erzeuge `Summary`.
- Kapsle bestehenden Phi-3-Code hinter `Phi3Summarizer`.
- Vermeide direkte Protokollabhängigkeiten im Summarizer.

### Akzeptanzkriterien

- Der Summarizer kann ohne JSON-RPC-Layer getestet werden.
- Der Sidecar ruft den Summarizer nur über das Interface auf.
- Phi-3-spezifische Klassen bleiben im Phi-3-Package.
- Keine neuen ACP-/MCP-Abhängigkeiten entstehen.

---

## Issue 9: `summarize`-Methode im Sidecar anbinden

**Labels:** `type:protocol`, `type:summarizer`, `priority:high`

### Ziel

Die Java-8-App soll über JSON-RPC eine Zusammenfassung anfordern können.

### Beispielrequest

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "summarize",
  "params": {
    "text": "Langer Eingabetext...",
    "maxTokens": 256
  }
}
```

### Beispielresponse

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "result": {
    "text": "Zusammenfassung..."
  }
}
```

### Aufgaben

- Implementiere Request-Mapping.
- Implementiere Response-Mapping.
- Implementiere Fehlerbehandlung.
- Optional: Streaming-Variante vorbereiten.

### Akzeptanzkriterien

- `summarize` funktioniert über stdin/stdout.
- Fachliche Fehler werden als JSON-RPC-Errors ausgegeben.
- Logs landen auf stderr.
- Die Protokollantwort ist stabil dokumentiert.

---

## Issue 10: Healthcheck für Sidecar und Modellstatus implementieren

**Labels:** `type:protocol`, `type:runtime`, `priority:high`

### Ziel

Die Java-8-App muss erkennen können, ob der Sidecar bereit ist.

### Beispielrequest

```json
{
  "jsonrpc": "2.0",
  "id": "health-1",
  "method": "health",
  "params": {}
}
```

### Beispielresponse

```json
{
  "jsonrpc": "2.0",
  "id": "health-1",
  "result": {
    "status": "ok",
    "ready": true,
    "busy": false
  }
}
```

### Aufgaben

- Implementiere Sidecar-Statusmodell.
- Unterscheide Prozess bereit, Modell geladen, Inferenz läuft.
- Füge Healthcheck-Handler hinzu.

### Akzeptanzkriterien

- Healthcheck funktioniert ohne Modellinitialisierung.
- Modellstatus ist eindeutig erkennbar.
- Busy-Zustand wird korrekt gemeldet.
- Fehlerhafte Runtime-Zustände werden sichtbar.

---

# Milestone 4: DirectML-Runtime-Core vorbereiten

## Issue 11: DirectML-Runtime-Core aus Phi-3-Code extrahieren

**Labels:** `type:directml`, `type:runtime`, `type:architecture`, `priority:high`

### Ziel

Gemeinsame DirectML-Infrastruktur soll von Phi-3 getrennt werden.

### Zu extrahierende Bereiche

```text
DirectML Device Lifecycle
D3D12 Device Lifecycle
Command Queue
Command Allocator
Command List
Descriptor Heap
Tensor Buffer
GPU Buffer Upload/Download
Kernel Dispatch
Resource Cleanup
```

### Aufgaben

- Identifiziere Phi-3-unabhängigen DirectML-Code.
- Verschiebe ihn in ein Runtime-Core-Package.
- Definiere klare Interfaces für Tensor-/Kernel-Verwaltung.
- Vermeide Phi-3-Abhängigkeiten im Runtime-Core.

### Akzeptanzkriterien

- Runtime-Core enthält keine Phi-3-Klassen.
- Phi-3 verwendet Runtime-Core über klare Interfaces.
- Ressourcen werden deterministisch freigegeben.
- Fehlerbehandlung für HRESULT/COM bleibt zentralisiert.

---

## Issue 12: Tensor- und Buffer-Abstraktion einführen

**Labels:** `type:directml`, `type:runtime`, `priority:high`

### Ziel

Encoder und Decoder sollen dieselbe Tensor-/Buffer-Schicht verwenden.

### Klassen

```text
TensorShape
TensorDataType
DirectMlTensor
GpuBuffer
CpuTensor
TensorLayout
```

### Aufgaben

- Definiere Shape- und Datentypmodelle.
- Kapsle GPU-Buffer-Erzeugung.
- Kapsle Upload/Download.
- Trenne fachliche Tensoren von DirectML-Ressourcen.

### Akzeptanzkriterien

- Tensoren können ohne Modellfamilienbezug erzeugt werden.
- Buffer-Lifetime ist eindeutig geregelt.
- Encoder- und Decoder-Code können dieselbe Schicht verwenden.
- Tests für Shape-/Layout-Berechnung existieren.

---

## Issue 13: Gemeinsame Kernel-Schnittstellen definieren

**Labels:** `type:directml`, `type:runtime`, `priority:medium`

### Ziel

Wiederverwendbare Kernel sollen über klare Interfaces verfügbar sein.

### Kernels

```text
Linear / MatMul
QuantizedLinear
LayerNorm
RMSNorm
GELU
SwiGLU
Attention
MeanPooling
L2Normalize
```

### Aufgaben

- Definiere Kernel-Interfaces.
- Implementiere zunächst nur die für MiniLM benötigten Kernels.
- Dokumentiere fehlende Kernel als TODO.
- Halte Kernel modellfamilienneutral.

### Akzeptanzkriterien

- MiniLM-Encoder kann benötigte Kernel verwenden.
- Kernel-Code enthält keine Modellnamen.
- Fehlende Kernel sind klar markiert.
- Phi-3-Code kann später schrittweise auf dieselben Kernel migriert werden.

---

# Milestone 5: Encoder-Runtime für Embeddings

## Issue 14: Encoder-Runtime-Grundstruktur einführen

**Labels:** `type:embedding`, `type:runtime`, `type:architecture`, `priority:high`

### Ziel

Embeddings sollen über eine separate Encoder-Runtime berechnet werden.

### Struktur

```text
EmbeddingModel
EncoderRuntime
EncoderArchitecture
EncoderWeights
EncoderTokenizer
PoolingStrategy
```

### Fachliche API

```java
public interface EmbeddingModel {

    EmbeddingVector embed(EmbeddingRequest request) throws InferenceException;
}
```

### Aufgaben

- Implementiere `EmbeddingModel`.
- Implementiere `EmbeddingRequest`.
- Implementiere `EmbeddingVector`.
- Implementiere `EncoderRuntime`.
- Implementiere `EncoderArchitecture`.

### Akzeptanzkriterien

- Encoder-Runtime ist getrennt von Phi-3.
- Embedding-API ist unabhängig von Sidecar-Protokoll.
- Encoder kann später MiniLM, E5 und JinaBERT aufnehmen.
- Keine Decoder-/KV-Cache-Abhängigkeiten im Encoder.

---

## Issue 15: MiniLM-Familienadapter für `all-MiniLM-L6-v2` vorbereiten

**Labels:** `type:embedding`, `type:runtime`, `priority:high`

### Ziel

`sentence-transformers/all-MiniLM-L6-v2` wird das erste Zielmodell für Embeddings.

### Aufgaben

- Analysiere Modellstruktur.
- Definiere `MiniLmArchitecture`.
- Definiere erwartete Tensor-Namen.
- Definiere benötigte Config-Felder.
- Definiere Pooling-Regel.
- Definiere Output-Dimension 384.

### Akzeptanzkriterien

- MiniLM-Architektur ist dokumentiert.
- Benötigte Tensoren sind aufgelistet.
- Benötigte Kernel sind aufgelistet.
- Es ist klar, welche Modellartefakte importiert werden müssen.

---

## Issue 16: Safetensors-Reader für Encoder-Gewichte implementieren

**Labels:** `type:embedding`, `type:runtime`, `priority:high`

### Ziel

Encoder-Gewichte sollen aus `model.safetensors` geladen werden können.

### Aufgaben

- Implementiere `SafetensorsReader`.
- Parse Header.
- Lese Tensor-Metadaten.
- Unterstütze relevante Datentypen.
- Stelle Tensoren über `TensorRegistry` bereit.

### Akzeptanzkriterien

- Tensor-Metadaten können gelesen werden.
- Tensor-Rohdaten können anhand Name/Offset geladen werden.
- Reader ist unabhängig von MiniLM/E5/Jina.
- Fehlerhafte Dateien erzeugen verständliche Fehler.

---

## Issue 17: Tokenizer-Import für Encoder-Modelle vorbereiten

**Labels:** `type:embedding`, `type:runtime`, `priority:high`

### Ziel

Encoder brauchen einen eigenen Tokenizer-Pfad, getrennt vom Phi-3-Tokenizer.

### Aufgaben

- Definiere `EncoderTokenizer`.
- Unterstütze zunächst WordPiece/BERT-artige Tokenizer.
- Lade `tokenizer.json` oder alternativ Vocabulary-Dateien.
- Erzeuge `input_ids`, `attention_mask` und optional `token_type_ids`.

### Akzeptanzkriterien

- Tokenizer ist nicht Phi-3-abhängig.
- Ein kurzer Text kann in Encoder-Inputs umgewandelt werden.
- Attention-Mask wird korrekt erzeugt.
- Sondertokens werden korrekt behandelt.

---

## Issue 18: Mean Pooling und L2-Normalisierung implementieren

**Labels:** `type:embedding`, `type:directml`, `priority:medium`

### Ziel

Sentence-Transformer-Embeddings benötigen Pooling und Normalisierung.

### Aufgaben

- Implementiere Mean Pooling über Token-Embeddings mit Attention Mask.
- Implementiere L2-Normalisierung des Ergebnisvektors.
- Entscheide, welche Operationen DirectML und welche CPU-Control-Plane bleiben.
- Halte die API modellneutral.

### Akzeptanzkriterien

- Pooling ignoriert Padding-Tokens.
- Ausgabevektor ist normalisiert.
- Ergebnisdimension entspricht dem Modell.
- Implementierung ist später für E5 wiederverwendbar.

---

## Issue 19: `embed`-Methode im Sidecar anbinden

**Labels:** `type:protocol`, `type:embedding`, `priority:high`

### Ziel

Die Java-8-App soll Embeddings über JSON-RPC anfordern können.

### Beispielrequest

```json
{
  "jsonrpc": "2.0",
  "id": "embed-1",
  "method": "embed",
  "params": {
    "text": "Dies ist ein Testtext."
  }
}
```

### Beispielresponse

```json
{
  "jsonrpc": "2.0",
  "id": "embed-1",
  "result": {
    "vector": [0.0123, -0.0456, 0.0789],
    "dimension": 384
  }
}
```

### Aufgaben

- Implementiere Request-Mapping.
- Implementiere Response-Mapping.
- Unterstütze später Batch-Embeddings.
- Dokumentiere numerische Ausgabe.

### Akzeptanzkriterien

- `embed` funktioniert über stdin/stdout.
- Einzeltext-Embedding wird unterstützt.
- Fehler werden als JSON-RPC-Errors ausgegeben.
- Das Protokoll ist erweiterbar für Batch-Embeddings.

---

# Milestone 6: Tests, Benchmarks und Qualität

## Issue 20: Referenztests gegen bekannte Embedding-Ausgaben vorbereiten

**Labels:** `type:testing`, `type:embedding`, `priority:medium`

### Ziel

Die eigene DirectML-Encoder-Runtime soll gegen Referenzwerte validiert werden.

### Aufgaben

- Definiere kleine Referenztexte.
- Erzeuge Referenz-Embeddings mit bekannter Runtime.
- Vergleiche Cosine Similarity statt exakter Float-Gleichheit.
- Definiere Toleranzen.

### Akzeptanzkriterien

- Testdaten sind dokumentiert.
- Vergleich nutzt Cosine Similarity.
- Toleranzen sind begründet.
- Regressionsfehler werden sichtbar.

---

## Issue 21: Sidecar-Prozess-Lifecycle testen

**Labels:** `type:testing`, `type:protocol`, `priority:medium`

### Ziel

Die Java-8-App muss den Sidecar robust starten, nutzen und beenden können.

### Aufgaben

- Teste Prozessstart.
- Teste Healthcheck.
- Teste Shutdown.
- Teste Fehlerfall bei ungültigem JSON.
- Teste Log-Ausgaben auf stderr.

### Akzeptanzkriterien

- Sidecar beendet sich sauber auf `shutdown`.
- Ungültige Requests crashen den Prozess nicht.
- stdout bleibt maschinenlesbares Protokoll.
- stderr enthält Diagnoseinformationen.

---

## Issue 22: Erste Performance-Benchmarks definieren

**Labels:** `type:testing`, `type:directml`, `priority:medium`

### Ziel

Performance soll früh messbar werden.

### Metriken

```text
Modellladezeit
Tokens pro Sekunde für Phi-3
Embedding-Latenz pro Text
Embedding-Durchsatz bei Batch-Verarbeitung
GPU-Speichernutzung
CPU-Auslastung
```

### Aufgaben

- Baue Benchmark-Harness.
- Dokumentiere Testhardware.
- Dokumentiere Modellversionen.
- Messe Warmup und stabile Laufzeit getrennt.

### Akzeptanzkriterien

- Benchmarks sind reproduzierbar.
- Ergebnisse enthalten Hardware- und Modellangaben.
- Warmup wird getrennt gemessen.
- Performance-Regressionen können später erkannt werden.

---

# Milestone 7: Vorbereitung für spätere Erweiterungen

## Issue 23: E5-Familienadapter konzeptionell vorbereiten

**Labels:** `type:embedding`, `type:architecture`, `priority:low`

### Ziel

Nach MiniLM soll die E5-Familie unterstützt werden.

### Zielmodelle

```text
danielheinz/e5-base-sts-en-de
intfloat/multilingual-e5-large-instruct
```

### Aufgaben

- Dokumentiere Unterschiede zu MiniLM.
- Dokumentiere Tokenizer-Anforderungen.
- Dokumentiere Pooling-Strategie.
- Dokumentiere Input-Prefixe wie `query:` und `passage:`, falls benötigt.

### Akzeptanzkriterien

- E5-Migration ist planbar.
- Unterschiede zu MiniLM sind sichtbar.
- Benötigte Runtime-Erweiterungen sind dokumentiert.

---

## Issue 24: Reranker-Runtime konzeptionell vorbereiten

**Labels:** `type:architecture`, `type:embedding`, `priority:low`

### Ziel

Reranker werden später als eigene Encoder-Variante umgesetzt.

### Architektur

```text
Query + Dokument
  ↓
Encoder
  ↓
Classification Head
  ↓
Score
```

### Aufgaben

- Definiere `Reranker`-Interface.
- Definiere `RerankRequest`.
- Definiere `RankedDocument`.
- Dokumentiere Top-N-Reranking-Strategie.
- Plane Batch-Verarbeitung.

### Akzeptanzkriterien

- Reranker-API ist skizziert.
- Reranker wird nicht mit EmbeddingModel vermischt.
- Top-N-Strategie ist dokumentiert.
- Keine Implementierungspflicht in dieser Phase.

---

## Issue 25: Decoder-LLM-Erweiterungen konzeptionell vorbereiten

**Labels:** `type:architecture`, `type:inference`, `priority:low`

### Ziel

Später sollen weitere Decoder-LLMs neben Phi-3 möglich sein.

### Kandidaten

```text
Qwen2 / Qwen2.5
Llama
weitere Phi-Varianten
```

### Aufgaben

- Definiere `DecoderArchitecture`.
- Trenne Phi-3-spezifische Logik vom gemeinsamen Decoder-Core.
- Dokumentiere Unterschiede: RoPE, GQA, QKV-Layout, Tokenizer, Chat-Template.
- Keine Implementierung in der aktuellen Encoder-Phase.

### Akzeptanzkriterien

- Phi-3 bleibt stabil.
- Decoder-Erweiterung ist architektonisch vorbereitet.
- Encoder-Arbeit wird dadurch nicht blockiert.
- Es entsteht kein Universal-Runner.

---

# Empfohlene erste Issues für GitHub

Für den Start sollten zuerst diese Issues angelegt werden:

```text
1. Repository-Struktur für win-directml-java finalisieren
2. Package-Namen auf com.aresstack.windirectml vereinheitlichen
3. Gradle-Build mit Java 21 stabilisieren
4. README für Projektziel und Nicht-Ziele erstellen
5. JSON-RPC-2.0-Protokoll über stdin/stdout spezifizieren
6. JSON-RPC-Basismodelle implementieren
7. Sidecar-Command-Dispatcher implementieren
8. Vorhandenen Phi-3-Code als Summarizer kapseln
9. summarize-Methode im Sidecar anbinden
10. Healthcheck für Sidecar und Modellstatus implementieren
```

Danach beginnt die Encoder-Schiene:

```text
11. DirectML-Runtime-Core aus Phi-3-Code extrahieren
12. Tensor- und Buffer-Abstraktion einführen
13. Encoder-Runtime-Grundstruktur einführen
14. MiniLM-Familienadapter für all-MiniLM-L6-v2 vorbereiten
15. Safetensors-Reader für Encoder-Gewichte implementieren
16. Tokenizer-Import für Encoder-Modelle vorbereiten
17. Mean Pooling und L2-Normalisierung implementieren
18. embed-Methode im Sidecar anbinden
```

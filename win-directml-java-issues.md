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
     "vector": [
        0.0123,
        -0.0456,
        0.0789
     ],
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

---

## Gelöst (Sprint `fix(runtime)/layernorm-debug-layer`): DirectMlLayerNormKernel CreateOperator gibt E_INVALIDARG

**Status:** ERLEDIGT. `DirectMlLayerNormKernelTest.layerNormMatchesCpuReference`
ist grün, `@Disabled` entfernt.
**Wurzel-Ursache:** Die Konstante `DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION`
war in `DirectMlBindings.java` als `39` definiert. Korrekt ist `73` –
ID `39` ist im DirectML-Enum tatsächlich `DML_OPERATOR_ACTIVATION_LEAKY_RELU`.
Damit wurde dem DML-Runtime das MVN-Operator-Desc-Layout (56 Bytes mit
`CrossChannel/NormalizeVariance/Epsilon`) als LeakyReLU-Desc (anderer Aufbau)
übergeben → konsequent `HRESULT 0x80070057 (E_INVALIDARG)`. Vergleichbar
falsch waren `BATCH_NORMALIZATION = 29` (richtig: `72`) und
`MEAN_VARIANCE_NORMALIZATION1 = 50` (richtig liegt deutlich höher, MVN1
wird aktuell nicht genutzt).
**Fix:**

1. Operator-IDs in `DirectMlBindings` korrigiert (MVN0 = 73, BN = 72).
2. LayerNorm-Kernel verwendet 4D-Layout `[M, 1, 1, H]` für Input/Output,
   `[1, 1, 1, H]` für `γ`/`β` mit impliziter Broadcast über die N-Achse,
   `CrossChannel = FALSE`, `NormalizeVariance = TRUE`.
3. `DirectML`-Debug-Pfad zusätzlich verdrahtet (PO-Forderung):
   System-Property `-Dwindirectml.debug=true` aktiviert in
   `WindowsBindings` sowohl den D3D12-Validation-Layer
   (`D3D12GetDebugInterface → ID3D12Debug::EnableDebugLayer`) als auch
   `DML_CREATE_DEVICE_FLAG_DEBUG`. Beim Initialisieren wird
   `ID3D12InfoQueue` über `QueryInterface` aufgegriffen;
   `WindowsBindings.drainDebugMessages()` zieht D3D12-Validierungsmeldungen
   ins Java-Land. DML-Meldungen gehen weiter an `OutputDebugString`
   (sichtbar über DbgView oder einen angeschlossenen Debugger).
4. Gradle-Build forward'ed alle `-Dwindirectml.*`-System-Properties in
   den Test-JVM.
   **Verifikation:** `gradlew :directml-windows-bindings:test` ist grün,
   sowohl mit als auch ohne `-Dwindirectml.debug=true`.

---

## Gelöst (Sprint `feat(runtime)/gelu-kernel`): DirectMlGeluKernel via nativer

`DML_OPERATOR_ACTIVATION_GELU` + DLL-Pfad-Override

**Status:** ERLEDIGT. `DirectMlGeluKernelTest.geluMatchesCpuReference`
ist grün, wenn der Prozess gegen eine `DirectML.dll` ≥ 1.10 (FL 5.1,
Windows 11 22H2+) läuft.

**Befund:** Der native `DML_OPERATOR_ACTIVATION_GELU` (Enum-ID 157,
gegen `%WindowsSdkDir%/Include/10.0.26100.0/um/DirectML.h` verifiziert)
wurde mit DirectML 1.10 eingeführt. Dev-Maschinen mit in-box
`C:\Windows\System32\DirectML.dll` 1.8.0 (Build vom Mai 2022) kennen
den Op nicht – `IDMLDevice::CreateOperator` antwortet mit
`HRESULT 0x80070057 (E_INVALIDARG)`. Moderne Windows-11-Builds liefern
1.15.5+ aus, dort funktioniert die native Op direkt.

**Revision der Policy:** Die ursprüngliche Vorgabe „nur die System32-DLL“
hätte uns dauerhaft auf einen 550-Zeilen-Composite-Kernel
(`ERF + IDENTITY + MULTIPLY`) festgenagelt und alle künftigen FL-5.1+
Operatoren (`MULTIHEAD_ATTENTION`, `MEAN_VARIANCE_NORMALIZATION1`,
`QUANTIZED_LINEAR`, …) ebenfalls als Composite-Hacks erzwungen. Daher:

- Anwendungen, die das Projekt einbetten, **dürfen** eine
  Microsoft.AI.DirectML-Redistributable neben ihre Binaries packen.
- Der Pfad zur DLL wird über die System-Property
  `-Dwindirectml.directml.dll=<absoluter Pfad>` an `DirectMlBindings`
  übergeben (`SymbolLookup.libraryLookup(Path)`).
- Ohne Property bleibt der Default das in-box `DirectML.dll` aus
  System32 (Windows-DLL-Suchreihenfolge).
- Es wird **nicht** automatisch nach DLLs in Fremdverzeichnissen wie
  `C:\Program Files\WSL\...` gesucht; das wäre nicht supportet.

**Fix (mehrere Commits):**

1. `DirectMlBindings`: System-Property
   `windirectml.directml.dll` ausgewertet vor dem `SymbolLookup`. Default-Pfad
   und Override-Pfad teilen sich denselben Lookup-Singleton.
   Hilfsmethode `directMlSource()` für Diagnose-Logs.
2. `build.gradle` (Root): `JavaExec`/Test-JVM erhält alle
   `-Dwindirectml.*`-Properties vom Gradle-Aufrufer durchgereicht.
3. `DirectMlGeluKernel` neu implementiert als **einzelner** nativer Op
   (Desc 16 Byte: `InputTensor*`, `OutputTensor*`). Pipeline identisch
   zu `DirectMlLinearKernel`/`DirectMlLayerNormKernel`:
   `CreateOperator → CompileOperator → CreateOperatorInitializer →
   BindingTable → ID3D12CommandList → ExecuteAndWait`.
4. `KernelRegistry`-Javadoc und Op-ID-Javadoc auf den neuen Stand
   aktualisiert (Verweis auf `-Dwindirectml.directml.dll`).

**Verifikation:**

```text
gradlew :directml-windows-bindings:test
  --tests *DirectMlGeluKernelTest*
  -Dwindirectml.directml.dll=<pfad zu DirectML.dll ≥ 1.10>
```

ist grün. Ohne Property auf einer Maschine mit in-box 1.8.0 schlägt
der Test mit `E_INVALIDARG` bei `CreateOperator` fehl – das ist das
erwartete, gut diagnostizierbare Signal, das nun in der Doku steht.

**Folgesprint:** dieselbe Strategie für die übrigen FL-5.1+-Operatoren
(`MULTIHEAD_ATTENTION` Op 164, `MEAN_VARIANCE_NORMALIZATION1` Op 115,
quantisierte Decoder-Pfade).

---

## Gelöst (Sprint `feat(runtime)/composite-gelu-fallback`):

`DirectMlCompositeGeluKernel` – FL-2.0-Composite-GELU für In-Box-DLL

**Status:** ERLEDIGT. Der vollständige `DirectMlMiniLmEncoder` läuft nun
auch gegen die Windows-11-In-Box `DirectML.dll` 1.8.0 (FL 5.0), ohne
zusätzliche Microsoft.AI.DirectML-Redistributable.

**Befund:** Der native fused GELU (Op 157) verlangt
`DML_FEATURE_LEVEL_5_1`. Die Akzeptanz für das Embedding-Sprint-Ziel war
ausdrücklich:

```text
Windows-11-In-Box DirectML ohne zusätzliche DirectML.dll
DirectMlMiniLmEncoder läuft auch mit in-box DirectML 1.8.0 / FL 5.0.
CPU-vs-DirectML Cosine bleibt > 0.99.
```

**Fix:**

1. Neuer Kernel `DirectMlCompositeGeluKernel` implementiert
   `GeluKernel` über drei FL-1.0/2.0-Primitives, die in jeder
   ausgelieferten `DirectML.dll` seit Win10 RS3 vorhanden sind:
   ```text
   tmpA = ERF(x,        ScaleBias = 1/√2, 0)   // erf(x/√2)
   tmpB = IDENTITY(tmpA, ScaleBias = 0.5,  0.5) // 0.5*(1 + erf(x/√2))
   y    = MULTIPLY(x, tmpB)                     // 0.5*x*(1 + erf(x/√2))
   ```
   Jeder Sub-Op hat eigenen Compiled-Op + DescriptorHeap + temp/persistent
   Buffers; die zwei Zwischenresultate liegen in zwei N-Float-Default-Buffern,
   die der Kernel selbst besitzt. Jeder `dispatch(x, y)` submittet drei
   Command-Lists mit `executeAndWait` dazwischen.
2. `GeluKernel` zur Interface-Factory erweitert
   (`GeluKernel.create(ctx, n)`): wählt `DirectMlGeluKernel` bei
   `FL ≥ 5.1`, sonst `DirectMlCompositeGeluKernel`.
3. `DirectMlMiniLmLayerBlock` zieht ab sofort über die Factory; das
   harte FL-5.1-Gate in `DirectMlMiniLmEncoder.load(...)` wurde entfernt
   (nur noch Info-Log beim Composite-Fallback).
4. Bestehende `DirectMlMiniLmLayerBlockTest` /
   `DirectMlMiniLmEncoderStackTest` /
   `DirectMlMiniLmEmbeddingReferenceTest` laufen jetzt sowohl auf FL 5.0
   (in-box) als auch auf FL 5.1+ (Redist); das FL-Gate dort wurde
   entfernt.
5. Neuer Regressionstest `DirectMlCompositeGeluKernelTest` zwingt den
   Composite-Pfad direkt (unabhängig vom Feature-Level), damit der
   Fallback auf modernen Hosts nicht still ungetestet bleibt.

**Verifikation:**

* `gradlew :directml-windows-bindings:test --tests *DirectMlCompositeGeluKernelTest*` – grün.
* In-box DLL (FL 5.0), kein `-Dwindirectml.directml.dll`:
  `gradlew :directml-encoder:test --tests *DirectMlMiniLmEmbeddingReferenceTest*` – grün.
  Log zeigt `FL=5.0 – using composite GELU fallback` und
  `DirectMlCompositeGeluKernel ready: N=… (ERF+IDENTITY+MUL composite, FL-2.0 path)` ×6 Layer.
  `cos(CPU, DML)` auf allen Referenzsätzen = **1,000000**,
  `min cos across corpus = 1,000000`.
* Redist DLL (FL 6.4) via `-Dwindirectml.directml.dll=<…>`:
  derselbe Test grün, Log meldet `FL=6.4` (kein Composite-Hinweis),
  weiterhin `cos = 1,000000`.
* Voller Test-Lauf auf in-box DLL: 69 Tests, 2 skipped (nativer
  `DirectMlGeluKernelTest` springt auf FL 5.0 weiterhin per
  `assumeTrue` raus; `DirectMlRedistDownloadIT` skipped wie bisher),
  0 failures, 0 errors.

**Optimierungsbacklog (nicht blockierend):**

* Drei `executeAndWait` pro `dispatch(x, y)` durch ein einziges
  Command-List-Recording mit UAV-Barriers zwischen den Sub-Ops und
  separaten Descriptor-Ranges in einem gemeinsamen Heap ersetzen.
* Optional: Native-vs-Composite-Wahl via `-Dwindirectml.gelu.strategy=
  {auto|native|composite}` exposen, falls Diagnose es verlangt.


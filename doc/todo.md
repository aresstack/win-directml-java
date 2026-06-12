# win-directml-java — Engine/Architecture Cleanup Plan

Stand: nach `win-directml-java-main(72)` / H2c.
Fokus: saubere Engine-Homogenisierung, Runtime-Lifecycle, gemeinsame WARP-/wdmlpack-Bausteine.
Nicht-Fokus: neue Modellfamilien, Workbench-Optik, Prompting, README-Politur, Logging-Kosmetik.

## Leitplanken

1. **Produktpfad bleibt WARP/AUTO.** WARP ist der stabile CPU-only-Pfad für gehärtete Windows-Workstations. AUTO darf später echte GPU wählen.
2. **Kein ONNX Runtime/PyTorch zur Laufzeit.** Import/Compile darf ONNX/SafeTensors lesen, Runtime lädt `.wdmlpack` bzw. runtimefähige lokale Assets.
3. **Qwen-Legacy bleibt vorerst erhalten.** Default bleibt `decoder-only session`; Legacy bleibt über `-Dqwen.runtime=legacy` erreichbar.
4. **Keine stillen Familien-Sonderwege.** Modellfamilien dürfen Adapter sein, aber keine zweite Engine implementieren, wenn dieselbe Semantik bereits in `decoderonly`, `t5`, `warp`, `model` oder `generation` gehört.
5. **Heap-light heißt nicht: float[] blind löschen.** Reference-/Diagnostic-/CPU-Pfade dürfen weiter `float[]` brauchen. WARP-Upload-Pfade sollen aber keine unnötigen vollständigen Host-Kopien materialisieren.
6. **Tests vor Doku.** Dokumentation wird am Ende geradegezogen. Während der Umsetzung zählen verifizierbare Runtime-/Architektur-Akzeptanzkriterien.
7. **Keine großen fremden Nebenbaustellen.** Kein Gemma, keine UI-Neugestaltung, kein Logging-Refactoring, keine Prompt-Änderungen, keine kosmetische Commit-Message-Korrektur mit Force-Push.

---

## 0. Repository-Stand sauberziehen

### Ziel

Der Arbeitsstand muss eindeutig auf `origin/main` liegen, bevor Architekturänderungen weiterlaufen.

### Aufgaben

* Lokale ungepushte Commits pushen, sofern noch nicht geschehen:

    * `ccaf029` — Gemma-Analyse-Doku, nur falls bereits im lokalen Verlauf enthalten.
    * `5aee13c` — H2c SmolLM2 FP32 ByteBuffer-Pilot.
* Kein Rebase, kein Amend, kein Force-Push.
* Danach bestätigen:

    * `origin/main` zeigt den erwarteten HEAD.
    * `git status` sauber.
    * `origin/main..HEAD` leer.
    * `HEAD..origin/main` leer.

### Akzeptanz

* Main ist Fast-Forward aktualisiert.
* Keine lokalen Architekturänderungen liegen unversioniert herum.

---

## 1. Runtime-Lifecycle und Ressourcenbesitz korrigieren

### Ziel

Alle Engine-Objekte, die native Ressourcen oder GPU-Pipelines besitzen, müssen eindeutig geschlossen werden. Das ist Voraussetzung für weitere Homogenisierung.

### Konkrete Baustellen

#### 1.1 `DecoderOnlyWarpForwardPass.close()` vollständig machen

Aktueller Befund:

* `DecoderOnlyWarpForwardPass.close()` schließt `layers` und `lmHead`.
* Es schließt nach aktuellem Stand nicht sichtbar vollständig:

    * `mlpPipeline`
    * `swiGluKernel`
    * `mlpBlocks`, falls diese selbst Ressourcen halten oder künftig halten werden.

Betroffene Dateien:

* `directml-inference/src/main/java/com/aresstack/windirectml/inference/decoderonly/DecoderOnlyWarpForwardPass.java`
* `DecoderOnlyWarpMlpBlock.java`
* `DecoderOnlyWarpSwiGluKernel.java`
* `DecoderOnlyWarpLayer.java`

Aufgaben:

* Eindeutiges Ownership-Modell festlegen:

    * `DecoderOnlyWarpForwardPass` besitzt nach erfolgreichem Konstruktoraufruf Layer, LM-Head, MLP-Pipeline, SwiGLU-Kernel und MLP-Blocks.
    * Bei Konstruktorfehlern werden nur bereits gebaute eigene Ressourcen geschlossen; übergebene Ressourcen bleiben bis zum Ownership-Handoff beim Aufrufer.
* `close()` idempotent machen und alle eigenen Ressourcen schließen.
* Fehler beim Schließen sammeln und suppressed weiterreichen, statt Ressourcen nach dem ersten Fehler liegen zu lassen.
* Tests ergänzen:

    * `close()` ist idempotent.
    * LM-Head wird geschlossen.
    * Layer werden geschlossen.
    * MLP-Pipeline/SwiGLU/Blocks werden geschlossen oder nachweislich nicht schließpflichtig gehalten.
    * Konstruktorfehler schließt nur eigene Teilressourcen.

### Akzeptanz

* Keine native/GPU-Ressource bleibt nach `close()` absichtlich offen.
* SmolLM2NativeWarpExecutorTest bleibt grün.
* Qwen decoder-only session bleibt grün.
* Keine Änderung an Generationsergebnissen.

---

## 2. Gemeinsamen Weight-Source-Vertrag einführen

### Ziel

T5 und SmolLM2 haben inzwischen ähnliche FP32-ByteBuffer-Pfade. Diese Logik darf nicht pro Familie wachsen. Es braucht einen gemeinsamen internen Vertrag für row-major FP32-Gewichte.

### Problem

Aktuell gibt es familiennahe Methoden wie:

* `T5TensorData.fp32LittleEndianSource()`
* `SmolLM2DenseTensor.fp32LittleEndianSource()`
* `copyValues()` / `values()` Fallbacks
* `DecoderOnlyWarpDenseProjection.fromRowMajorWeights(..., ByteBuffer)`
* `T5WarpLinearProjection.from(... ByteBuffer ...)`

Diese sind funktional richtig, aber Architektur dupliziert sich.

### Aufgaben

* Einen gemeinsamen internen Typ einführen, z. B. sinngemäß:

    * `RowMajorFp32WeightSource`
    * oder `WarpWeightSource`
    * oder `RuntimeWeightView`
* Der Typ muss ausdrücken können:

    * Name
    * Output rows
    * Input columns
    * optionaler `ByteBuffer` FP32 little-endian Source-Slice
    * optionaler/lazy `float[]` Fallback
    * Datentyp/Grund, warum ByteBuffer nicht verfügbar ist
* Der Typ darf kein Runtime-Format ändern.
* Der Typ darf keine `float[]`-Materialisierung erzwingen, wenn ByteBuffer verfügbar ist.
* T5 und SmolLM2 sollen diesen Typ verwenden, statt eigene Upload-Entscheidungen zu duplizieren.
* `WarpDenseProjection` und `DecoderOnlyWarpDenseProjection` sollen diesen Typ direkt akzeptieren können.

### Betroffene Dateien

* `directml-inference/src/main/java/com/aresstack/windirectml/inference/warp/WarpDenseProjection.java`
* `directml-inference/src/main/java/com/aresstack/windirectml/inference/decoderonly/DecoderOnlyWarpDenseProjection.java`
* `directml-inference/src/main/java/com/aresstack/windirectml/inference/t5/T5TensorData.java`
* `directml-inference/src/main/java/com/aresstack/windirectml/inference/t5/T5WarpLinearProjection.java`
* `directml-inference/src/main/java/com/aresstack/windirectml/inference/smollm2/SmolLM2DenseTensor.java`
* `directml-inference/src/main/java/com/aresstack/windirectml/inference/smollm2/SmolLM2WarpForwardPass.java`

### Akzeptanz

* T5 und SmolLM2 benutzen denselben Weight-Source-Vertrag für FP32-WARP-Projektionen.
* Bestehende `float[]`-Fallbacks bleiben erhalten.
* ByteBuffer-Pfad bleibt numerisch identisch zum bisherigen `float[]`-Pfad.
* Keine `.wdmlpack`-Formatänderung.

---

## 3. Fused Projection Builder family-neutral und heap-light machen

### Ziel

Fused QKV/Gate-Up-Projektionen dürfen nicht länger pro Familie eigene volle Host-`float[]`-Konkatenationen erzwingen, wenn FP32-ByteBuffer-Quellen vorhanden sind.

### Aktuelle konkrete Probleme

#### 3.1 Decoder-only fused projections

Datei:

* `DecoderOnlyWarpFusedDenseProjection.java`

Aktueller Zustand:

* `fromRowMajorParts(...)` nimmt `Part` mit `float[] weights`.
* Es baut immer:

    * `float[] fusedWeights = new float[totalOutputSize * inputSize]`
    * `System.arraycopy(...)`
    * Upload als kompletter Host-FP32-Fused-Array.

Betroffen:

* SmolLM2 `qkv`
* SmolLM2 `gate_up`
* später mögliche andere decoder-only FP32-Familien.

#### 3.2 T5 fused projections

Dateien:

* `T5FusedSelfAttentionProjection.java`
* `T5FusedCrossAttentionMemoryProjection.java`
* `T5WarpLinearProjectionFactory.java`

Aktueller Zustand:

* Q/K/V bzw. K/V werden über `float[] fused` gebaut.
* Danach wird daraus ein `T5TensorData.reference(...)` erzeugt.
* Dadurch geht der ByteBuffer-/mmap-Vorteil für fused Gewichte verloren.

### Aufgaben

* Einen gemeinsamen Fused-Projection-Builder in `warp` oder `decoderonly`/`t5`-neutraler Schicht bauen.
* Der Builder muss Teile aus dem gemeinsamen Weight-Source-Vertrag akzeptieren:

    * bevorzugt FP32-ByteBuffer-Slices
    * Fallback `float[]`
* Für FP32-ByteBuffer-Teile soll keine große Heap-`float[]`-Konkatenation entstehen.
* Wenn eine Konkatenation nötig bleibt, dann bevorzugt:

    * direct/native ByteBuffer
    * Upload-Buffer-Streaming
    * oder mehrere Copy-Regionen in einen Ziel-Upload, aber ohne Java-Heap-FP32-Fused-Array.
* Der Builder muss dieselbe mathematische Semantik behalten:

    * vertikale row-major Stack-Reihenfolge
    * identische Slice-Offets
    * identisches `copySlice` / `copySliceSequence` Verhalten.
* T5 Self-Attn und Cross-Attn sollen denselben Builder nutzen oder zumindest dieselbe interne Fused-Weight-Quelle.

### Akzeptanz

* SmolLM2 `qkv` und `gate_up` materialisieren keine vollständige Java-Heap-`float[]`-Fused-Matrix mehr, wenn alle Teile FP32-ByteBuffer-fähig sind.
* T5 Q/K/V und Cross-Attn K/V materialisieren keine vollständige Java-Heap-`float[]`-Fused-Matrix mehr, wenn alle Teile FP32-ByteBuffer-fähig sind.
* Fused und unfused Ergebnisse bleiben numerisch identisch innerhalb bestehender Toleranzen.
* Fallback für FP16/BF16/reference bleibt erhalten.

---

## 4. Decoder-only Engine-Grenzen bereinigen

### Ziel

Qwen und SmolLM2 sollen nur noch Adapter für Konfiguration, Tokenizer und Gewichtsmapping sein. Die Decode-Engine selbst muss family-neutral bleiben.

### Aufgaben

* Prüfen, dass folgende Semantik nur einmal in `decoderonly` liegt:

    * RMSNorm
    * RoPE
    * GQA/MQA Attention Layout
    * KV-Cache Layout
    * Stop-Token-Vertrag
    * Generation Loop
    * WARP dense projection dispatch
    * WARP MLP block dispatch
    * LM-head projection
* Qwen-spezifische und SmolLM2-spezifische Klassen dürfen nur noch:

    * Tensor-Namen auf Engine-Rollen mappen
    * Config auf `DecoderOnlyConfig` mappen
    * Tokenizer/Chat-Template bereitstellen
    * Runtime-Package laden
    * Engine aufrufen
* Keine neue Modellfamilie darf eigene Kopien der Decode-Loop oder des Stop-Vertrags bekommen.
* `DecoderOnlyGenerationResult.toSummary()` soll der interne Standardpfad werden.

### Betroffene Bereiche

* `decoderonly/*`
* `qwen/*`
* `smollm2/*`
* `generation/*`

### Akzeptanz

* Qwen und SmolLM2 benutzen denselben `DecoderOnlyGenerationLoop` und denselben Stop-Token-Vertrag.
* Qwen Default bleibt decoder-only session.
* Qwen Legacy bleibt als expliziter Fallback erhalten, aber wird nicht erweitert.
* Kein Verhalten ändert sich in den bestehenden Qwen-/SmolLM2-Smokes.

---

## 5. Qwen INT4/Fusion separat stabilisieren

### Ziel

Qwen bleibt ein eigener Sonderfall wegen INT4, MatMulNBits, Fused-QKV/Gate-Up und gepackten Quantisierungsdaten. Diese Logik soll sauber und messbar werden, aber nicht mit den FP32-Heap-Light-Pfaden vermischt werden.

### Aktueller Befund

Datei:

* `QwenGpuKernels.java`

Konkrete Stellen:

* `createFusedQKV(...)`
* `createFusedGateUp(...)`
* `tryFuseQuantizedRowwise(...)`
* `dequantOrExtract(...)`

Problem:

* Fused QKV/Gate-Up bevorzugt INT4-rowwise concat.
* Wenn Fusing nicht möglich ist, fällt es auf FP32 dequant + concat zurück.
* Nicht byte-aligned `zeroPoints` führen zum Fallback, weil Bit-shift-Pfad nicht implementiert ist.
* Dadurch kann unbemerkt viel FP32-Heap entstehen und Performance/Heap stark schwanken.

### Aufgaben

* Qwen INT4-Fusion messbar machen:

    * pro Layer loggen/profilieren, ob INT4-fused oder FP32-fallback genutzt wird.
    * Zähler in Profil/Diagnose aufnehmen.
* Fallback-Politik festlegen:

    * Produktpfad darf nicht still unerwartet auf große FP32-Konkatenation fallen, ohne sichtbare Diagnose.
* Bit-shift-Pfad für packed-nibble `zeroPoints` implementieren oder bewusst als harte Unsupported-Konstellation behandeln.
* Prüfen, ob `MatMulNBitsKernel` zusätzliche ByteBuffer-/packed-source Overloads braucht, um Qwen-Fusion ohne Java-Heap-Kopie zu bauen.
* Qwen LM-Head und Einzelprojektionen getrennt analysieren:

    * INT4 packed path bevorzugen.
    * dense FP32 nur als expliziter Fallback.

### Akzeptanz

* Für ein bekanntes Qwen-0.5B-Modell ist pro Layer sichtbar, ob INT4-Fusion aktiv ist.
* Kein unerwarteter FP32-Fallback bleibt unbemerkt.
* Wenn der Fallback weiter existiert, ist er getestet und diagnostisch klar.
* Qwen Ausgabe bleibt token-identisch oder innerhalb der bisherigen Qwen-Paritätsgrenzen.
* Legacy bleibt unverändert verfügbar.

---

## 6. T5 Manifest und Runtime-Status korrigieren

### Ziel

T5-Manifeste und Runtime-Status dürfen nicht mehr historisch „not implemented“ behaupten, wenn der aktuelle Stand bereits runtimefähige Teilpfade hat. Gleichzeitig darf T5 nicht fälschlich als vollständig produktionsreif markiert werden, falls noch Engine-Lücken bestehen.

### Aktueller Befund

Dateien:

* `T5WdmlPackManifestWriter.java`
* `T5ManifestPayloadPolicy.java`
* `T5RuntimePackage.java`
* T5-Tests mit `runtimeLoadable=false` Erwartungen.

Konkreter Zustand:

* `runtimeLoadable` wird im Writer hart auf `false` gesetzt.
* `runtimeLoadMode` enthält noch `t5-weights-loaded-runtime-not-implemented`.
* `T5ManifestPayloadPolicy.REASON` lautet „T5 runtime is not implemented yet“.

### Aufgaben

* Statusmodell schärfen:

    * `weightsLoadable`
    * `runtimeLoadable`
    * `executable`
    * `runtimeLoadMode`
    * `reason`
* T5 darf nur dann `runtimeLoadable=true` sein, wenn das Paket vom aktuellen T5-Runtime-Loader wirklich geladen werden kann.
* Wenn Generierung noch fachlich unvollständig ist, dann nicht über `runtimeLoadable=false` verstecken, sondern mit einem separaten ausführbaren Status/Reason ausdrücken.
* Tests entsprechend korrigieren.
* Workbench darf daraus klar ableiten:

    * Paket ist importiert.
    * Paket hat Payload.
    * Paket kann geladen werden.
    * Modell kann sinnvoll ausgeführt werden.

### Akzeptanz

* T5-Manifeste lügen nicht mehr mit „runtime not implemented“, wenn die Runtime tatsächlich laden kann.
* Workbench/Download-Status kann T5 korrekt anzeigen.
* T5-Tests sind auf die neue Statussemantik umgestellt.

---

## 7. T5 Engine fachlich fertigstellen oder bewusst abgrenzen

### Ziel

T5 soll entweder sauber produktiv werden oder klar als experimenteller/teilweiser Seq2Seq-Pfad abgegrenzt sein. Halbzustände dürfen nicht als fertige Engine erscheinen.

### Bekannte offene Engine-Themen

Betroffene Dateien/Bereiche:

* `T5WarpEncoderPipeline.java`
* `T5WarpDecoderPipeline.java`
* `T5SelfAttention.java`
* `T5CrossAttention.java`
* `T5LayerNorm.java`
* `T5GenerationLoop.java`
* `T5WarpLmHead.java`
* Tokenizer/Prompting für T5/CodeT5.

Aufgaben:

* Entscheiden und implementieren:

    * Welche Operationen sind WARP-Produktpfad?
    * Welche Operationen bleiben Java/reference?
    * Wo ist die Grenze eindeutig dokumentiert und getestet?
* Echte Seq2Seq-Funktion absichern:

    * Encoder-Ausgabe korrekt.
    * Decoder-Step korrekt.
    * Cross-Attention korrekt.
    * Cache/Memory-Struktur korrekt.
    * LM-Head korrekt.
    * Stop-Token-Vertrag korrekt.
    * Ausgabe nicht leer, wenn Modell/Prompt gültig sind.
* CodeT5 und google-t5 Varianten sauber trennen:

    * Tokenizer
    * Prefixe/Tasks
    * Modellfamilie/Architektur.

### Akzeptanz

* T5 kann für mindestens einen konkret unterstützten lokalen Modellkandidaten eine echte Textausgabe erzeugen.
* T5 hat eine klare Runtime-Status-Semantik.
* T5 nutzt gemeinsame `generation`-Typen.
* Keine T5-Sonderlösung dupliziert decoder-only Konzepte unnötig.

---

## 8. Gemeinsamen Generation-/Loadability-Vertrag in Workbench/API durchziehen

### Ziel

Die Engine hat bereits gemeinsame Typen; die Produktaußenkante soll sie auch nutzen.

### Aktueller Zustand

Gemeinsame Typen existieren:

* `GenerationSummary`
* `GenerationFinishReason`
* `RuntimeLoadability`

Workbench und Inference-Engines lesen aber noch teilweise familienweise Result-Felder und Strings.

Betroffene Dateien:

* `SummarizerPanel.java`
* `SmolLM2WorkbenchRuntimeRunner.java`
* `T5InferenceEngine.java`
* `QwenInferenceEngine.java`
* `InferenceResult.java`
* `Summary.java`

### Aufgaben

* Interner Standard:

    * Familienruntime erzeugt/konvertiert auf `GenerationSummary`.
    * Workbench zeigt `GenerationSummary` an, nicht family-spezifische Finish-Strings.
* Finish-Reasons vereinheitlichen:

    * stop token
    * max tokens / length
    * error
    * end turn, falls API-Kompatibilität erforderlich.
* `RuntimeLoadability` als interner Standard für Modellstatus verwenden.
* Alte String-Felder nur an API-Kompatibilitätsgrenzen erhalten.

### Akzeptanz

* Workbench muss nicht wissen, ob Result von Qwen, SmolLM2 oder T5 stammt, um Tokens/Finish/Usage anzuzeigen.
* Familien-Sondercode in Panels wird reduziert.
* Bestehende Ausgaben bleiben für Nutzer verständlich.

---

## 9. Model-/wdmlpack-Lifecycle vereinheitlichen

### Ziel

`model/` ist die zentrale Schicht für runtimefähige Packages. Familien dürfen nicht jeweils eigene Lebensdauer-, Mapping- oder Payload-Regeln erfinden.

### Aufgaben

* Prüfen und vereinheitlichen:

    * `RuntimeModelPackage`
    * `WdmlPackReader`
    * `WdmlPackManifest`
    * `RuntimeTensorCatalog`
    * `RuntimeTensor`
    * family-specific package wrappers.
* mmap-/ByteBuffer-Lifetime explizit machen:

    * Wer besitzt das Mapping?
    * Wie lange dürfen Tensor-Views leben?
    * Was passiert nach `close()`?
* RuntimeTensor soll der bevorzugte Lieferant für ByteBuffer-Slices sein.
* Familien-Tensorwrapper sollen nur noch dünne, typisierte Views sein.
* Paketstatus und Loadability sollen überall gleich entstehen.

### Akzeptanz

* T5 und SmolLM2 ziehen FP32-ByteBuffer-Quellen aus derselben model-Lifecycle-Semantik.
* Qwen-SafeTensors/wdmlpack-Sonderpfade werden nicht weiter divergierend ausgebaut.
* Keine dangling mmap-Views nach Package-Close in Tests.

---

## 10. Encoder/Reranker nur stabilisieren, nicht neu erfinden

### Ziel

MiniLM/E5/Reranker sind nach ER-1/ER-2 grundsätzlich stabil. Hier nur Architektur an Produktstatus und Validierung angleichen.

### Aufgaben

* `ModelAssetValidation` als einheitliche Statusquelle für Workbench-Download/Runtime verwenden.
* Status sauber anzeigen:

    * not downloaded
    * incomplete
    * corrupt
    * wrong variant
    * loadable
    * executable
* Keine neuen Encoder-Architekturen anfangen, solange CLS/MAX Pooling, Projection Heads und XLM-R/SentencePiece nicht bewusst geplant sind.

### Akzeptanz

* MiniLM/E5/Reranker bleiben grün.
* Workbench-Status widerspricht nicht dem Loader.
* Kein Einfluss auf decoder-only/T5 Engine.

---

## 11. Intel-/WARP-Profiling als Engine-Messung, nicht als Modellraterei

### Ziel

Performanceprobleme auf dem Intel-Notebook müssen über Engine-Profile verstanden werden, nicht über Vermutungen.

### Aufgaben

Für Qwen, SmolLM2 und T5, soweit runtimefähig:

* Load-Time messen.
* Prefill-Time messen.
* Decode-Time pro Token messen.
* LM-Head-Anteil messen.
* Fused-Projektions-Anteil messen.
* Attention/RoPE/Norm Java-Anteil messen.
* Heap vor/nach Load messen.
* Zahl der FP32-Fallbacks und INT4-Fusionen anzeigen.

Bestehende/naheliegende Orte:

* `DecoderOnlyWarpDecodeProfile`
* Qwen decode profile
* SmolLM2 profile output
* Workbench Runtime Runner Ausgabe

### Akzeptanz

* Für Qwen auf Intel-WARP ist klar, welcher Anteil dominiert:

    * LM-Head
    * INT4 MatMul
    * Java Attention
    * Dispatch/Fence
    * Prefill
    * Heap/Upload.
* Daraus folgt eine technische Priorität, nicht Bauchgefühl.

---

## 12. Schluss-Cleanup erst nach Engine-Stabilisierung

### Ziel

Erst wenn die Engine-Grenzen sauber sind, werden Doku, Logs und alte Pfade bereinigt.

### Aufgaben später

* Dokumentation geradeziehen:

    * Heap-light-Doku an tatsächlichen Stand anpassen.
    * Decoder-only-Homogenisierungsdoku aktualisieren.
    * T5-Status korrekt darstellen.
    * Workbench-Statusdoku aktualisieren.
* Logging finalisieren:

    * SLF4J/Logback.
    * laute Diagnoseausgaben reduzieren.
* Legacy-/Experimentpfade später entfernen:

    * Qwen Legacy erst nach längerer Stabilität.
    * alte Forschungsmodi erst nach Releaseentscheidung.
* Release-Härtung:

    * README
    * SUPPORTED_MODELS
    * Workbench Packaging
    * ChatGPT-compatible Release.

### Akzeptanz

* Kein Cleanup entfernt Diagnose, die noch zur Engine-Härtung gebraucht wird.
* Kein Legacy-Pfad wird vorzeitig gelöscht.

---

# Reihenfolge der Abarbeitung

1. Repository-Stand sauberziehen.
2. Runtime-Lifecycle/Close korrigieren.
3. Gemeinsamen Weight-Source-Vertrag einführen.
4. Fused Projection Builder family-neutral und heap-light machen.
5. Decoder-only Engine-Grenzen bereinigen.
6. Qwen INT4/Fusion stabilisieren und messbar machen.
7. T5 Manifest/Runtime-Status korrigieren.
8. T5 fachlich fertigstellen oder klar abgrenzen.
9. Generation-/Loadability-Vertrag in Workbench/API durchziehen.
10. Model-/wdmlpack-Lifecycle vereinheitlichen.
11. Encoder/Reranker-Status vereinheitlichen.
12. Intel-/WARP-Profiling durchführen.
13. Doku/Logging/Release-Cleanup.

# Harte Abnahmekriterien für den gesamten Block

* Qwen Default bleibt `decoder-only session`.
* Qwen Legacy bleibt über `-Dqwen.runtime=legacy` erreichbar.
* Produktstart bleibt ohne `--add-modules=jdk.incubator.vector` möglich.
* Compile/Test dürfen Vector-Modul weiter verwenden.
* Keine `.wdmlpack`-Formatänderung ohne explizite Entscheidung.
* T5/SmolLM2/Qwen teilen Engine-Bausteine, statt Familienkopien zu bilden.
* WARP-Upload-Pfade vermeiden unnötige Java-Heap-FP32-Kopien.
* Reference-/Diagnostic-Pfade bleiben korrekt.
* Workbench liest gemeinsame Runtime-/Generation-Status, nicht zufällige Familienstrings.
* MiniLM/E5/Reranker bleiben stabil.
* Alle betroffenen Tests grün:

    * qwen
    * decoderonly
    * smollm2
    * t5
    * generation
    * model
    * warp/windows-bindings
    * encoder/reranker
    * SmolLM2NativeWarpExecutorTest

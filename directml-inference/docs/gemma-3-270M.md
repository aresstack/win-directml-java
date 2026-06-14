# Plan für Copilot/Opus: Gemma 3 270M-it nativer Java/WARP-Pfad

## Ausgangslage

Aktuell läuft `google/gemma-3-270m-it` in der Workbench nur über den externen Python-/Transformers-Probe-Pfad:

```text
Workbench
→ external-python-transformers
→ HF-Dateien aus AppData
→ Ausgabe zurück in Workbench
```

Beispielstatus:

```text
Runtime mode: external-python-transformers
Model id: google/gemma-3-270m-it
Model directory: C:\Users\angel\AppData\Roaming\.directml\model\gemma-3-270m-it
```

Das ist für Evaluation okay, aber nicht das Produktziel.

Ziel ist ein echter nativer Pfad:

```text
HF / SafeTensors / config.json
→ Gemma3 Compiler
→ .wdmlpack
→ Java Runtime
→ DirectML/WARP
→ Workbench
```

Keine Python-Runtime, kein Transformers zur Laufzeit, keine ONNX Runtime, keine fremde Inferenzschicht.

---

## Harte Leitplanken

Gemma 3 darf **nicht** einfach in die bestehende Qwen-/SmolLM2-decoderonly-Schicht hineingepresst werden.

Die Feasibility hat gezeigt, dass Gemma 3 mathematisch deutlich abweicht:

```text
- zero-centered RMSNorm: x * (1 + w)
- GeGLU + GELU-tanh statt SwiGLU/SiLU
- QK-Norm
- Sandwich-Norms pro Layer
- local/global Attention
- sliding window
- duale RoPE-Parameter
- head_dim nicht zwingend hidden / heads
- großes 256k-Vokabular / großer LM-Head
```

Daher:

```text
Gemma bekommt eine eigene Familie:
directml-inference/src/main/java/com/aresstack/windirectml/inference/gemma/**
directml-inference/src/test/java/com/aresstack/windirectml/inference/gemma/**
```

Gemma darf aber gemeinsame untere Bausteine wiederverwenden:

```text
model/**
compiler/**
warp/WarpDenseProjection
D3D12 ByteBuffer Upload
RuntimeModelPackage / WdmlPackReader
GenerationSummary / GenerationFinishReason
Token-Selektion / Stop-Policy-Bausteine, sofern passend
```

Nicht erlaubt:

```text
- keine Qwen-Migration
- keine SmolLM2-Migration
- keine T5-Migration
- keine Entfernung des external-python-Probe-Pfads
- keine großen Refactorings
- keine Workbench-Neuarchitektur
- keine .wdmlpack-Formatänderung ohne separaten Slice
- kein Big-Bang "Gemma vollständig implementieren"
```

Jeder Slice bekommt einen eigenen Commit und danach Stopp.

---

## Wichtig vorab: Workbench-Prompt prüfen

Die aktuelle Workbench-Ausgabe:

```text
Okay, I'm ready. Please paste the longer text or prompt you want me to use as the basis for the workbench.
```

klingt so, als ob der eigentliche Summarizer-/Workbench-Text nicht sauber im Gemma-Prompt landet oder der externe Probe-Prompt zu allgemein ist.

Das ist unabhängig vom nativen WARP-Pfad. Der Fehler darf nicht in die native Runtime übernommen werden.

---

# Slice GEMMA-P0: Workbench-Prompt-Sanity für Gemma

## Ziel

Prüfen, ob die Workbench beim Modell `google/gemma-3-270m-it` wirklich den Benutzertext / Summarizer-Input an das Modell weitergibt.

## Erlaubt

* Nur minimale Diagnose im Gemma-External-Probe-Pfad.
* Prompt-Preview im Debug-/Verbose-Log, maximal gekürzt.
* Test/Doc, der zeigt, welches Promptformat an Gemma geht.

## Nicht erlaubt

```text
- keine native Gemma-Runtime
- keine WARP-Kernel
- keine Downloader-Änderung
- keine UI-Umgestaltung
- keine Qwen-/Smol-/T5-Änderung
```

## Akzeptanz

* Der externe Gemma-Pfad bekommt tatsächlich den Workbench-Eingabetext.
* Falls nicht, wird der Prompt-Pfad minimal korrigiert.
* External Python Probe bleibt lauffähig.
* Eigener Commit.
* Danach stoppen.

---

# Slice GEMMA-WARP-1: Native Gemma Family Shell + Inspect

## Ziel

Gemma 3 270M-it als eigene native Modellfamilie vorbereiten, ohne echte Generierung zu implementieren.

## Neue Struktur

Anlegen:

```text
directml-inference/src/main/java/com/aresstack/windirectml/inference/gemma/
directml-inference/src/test/java/com/aresstack/windirectml/inference/gemma/
```

Mögliche Klassen:

```text
Gemma3Config
Gemma3Architecture
Gemma3ModelIds
Gemma3SpecialTokens
Gemma3TensorRole
Gemma3TensorNameMapper
Gemma3LayoutManifest
Gemma3SafeTensorsInspector
Gemma3RuntimeLoadability
Gemma3RuntimePackage
Gemma3PromptTemplate
```

## Inspect-Aufgabe

Aus dem lokalen Modellverzeichnis auslesen:

```text
C:\Users\angel\AppData\Roaming\.directml\model\gemma-3-270m-it
```

Zu prüfen und zu dokumentieren:

```text
- config.json
- model.safetensors Tensor-Namen
- Tensor-Shapes
- dtype
- tokenizer.json
- tokenizer.model
- tokenizer_config.json
- special_tokens_map.json
- chat_template.jinja
- generation_config.json
```

Konkret bestätigen:

```text
- vocab size
- hidden size
- intermediate size
- layer count
- attention heads
- key/value heads
- head_dim
- context length
- sliding window
- local/global attention pattern
- RoPE-Parameter
- RMSNorm eps
- zero-centered RMSNorm
- activation
- tied embeddings / lm_head
- BOS/EOS/PAD IDs
- start_of_turn / end_of_turn Tokens
```

## Doku

Anlegen oder ergänzen:

```text
docs/gemma3-warp-runtime-plan.md
docs/model-candidate-gemma3-270m.md
```

## Nicht erlaubt

```text
- keine native Generierung
- keine WARP-Kernel
- keine .wdmlpack-Formatänderung
- keine Workbench-UI-Änderung
- keine Downloader-Änderung
- keine Qwen-/Smol-/T5-Änderung
```

## Akzeptanz

* Gemma-Konfiguration real aus lokalem Modell bestätigt.
* Tensorrollen mindestens als Mapping-Plan dokumentiert.
* Keine Runtime-Verhaltensänderung.
* Tests grün.
* Eigener Commit.
* Danach stoppen.

---

# Slice GEMMA-WARP-2: Gemma Tokenizer + Chat Template Validierung

## Ziel

Prüfen, ob der vorhandene Java-Tokenizer-Pfad `tokenizer.json` von Gemma korrekt nutzen kann.

Gemma darf keine neue native Tokenizer-Abhängigkeit einführen.

## Zu prüfen

```text
- tokenizer.json BPE/SentencePiece-Struktur
- tokenizer.model nur Fallback oder zwingend?
- Normalizer
- Byte fallback
- added tokens
- special tokens
- <bos>, <eos>, <pad>
- <start_of_turn>, <end_of_turn>
- chat_template.jinja
```

## Erlaubt

* Gemma-spezifische PromptTemplate-Klasse.
* Tokenizer-Kompatibilitätstests.
* Kleine Roundtrip-Tests mit bekannten Strings.
* Vergleich gegen Python/Transformers für 3–5 fixe Prompts, falls einfach möglich.

## Nicht erlaubt

```text
- keine Generierung
- keine WARP-Kernel
- keine Runtime-Integration in Workbench
- keine Änderung an Qwen-/Smol-/T5-Tokenizern
```

## Akzeptanz

* Gemma Prompt wird in Java korrekt gebaut.
* Token IDs stimmen für Testprompts gegen Transformers oder sind sauber dokumentiert.
* Chat-Template korrekt für `gemma-3-270m-it`.
* Eigener Commit.
* Danach stoppen.

---

# Slice GEMMA-WARP-3: Gemma .wdmlpack Compiler Shell

## Ziel

Gemma soll wie die anderen Modellfamilien nicht direkt aus Hugging-Face-Dateien zur Laufzeit laufen.

Ziel:

```text
HF SafeTensors + config.json
→ Gemma3WdmlPackCompiler
→ .wdmlpack
→ RuntimeModelPackage
```

Noch keine echte native Generierung.

## Neue Klassen

```text
Gemma3WdmlPackCompiler
Gemma3WdmlPackCompileTool
Gemma3SafeTensorsModelSource
Gemma3LayoutCompiler
Gemma3PackageMetadata
Gemma3CompileReport
```

## Verhalten

Der Compiler soll zunächst:

```text
- config.json lesen
- model.safetensors lesen
- Tensoren mappen
- Shape/DType validieren
- Manifest schreiben
- Payload aufnehmen oder referenzieren, gemäß bestehender .wdmlpack-Infrastruktur
- runtimeLoadable=false oder experimental=true setzen, solange keine Runtime existiert
```

## Nicht erlaubt

```text
- keine vollständige Runtime
- keine WARP-Kernel
- keine Workbench-UI
- keine Downloader-Änderung
- keine .wdmlpack-Formatänderung, außer zwingend und separat gemeldet
```

## Akzeptanz

* Gemma `.wdmlpack` kann erzeugt oder trocken inspiziert werden.
* Fehlende Tensoren / Shape-Fehler werden klar gemeldet.
* Runtime lädt das Paket noch nicht als ausführbar.
* Eigener Commit.
* Danach stoppen.

---

# Slice GEMMA-WARP-4: Gemma CPU-Reference-Math für Parity

## Ziel

Vor WARP-Kernels muss die Gemma-Mathematik korrekt verstanden werden.

Eine kleine Java-Reference darf existieren, aber nur als Test-/Parity-Helfer, nicht als Produktpfad.

## Zu implementieren

Gemma-spezifische Reference-Bausteine:

```text
Gemma3ReferenceRmsNormZeroCentered
Gemma3ReferenceGeluTanh
Gemma3ReferenceGeGluMlp
Gemma3ReferenceQKNorm
Gemma3ReferenceRoPE
Gemma3ReferenceAttention
Gemma3ReferenceLayer
```

## Zu validieren

Mit kleinen künstlichen Tensoren:

```text
- zero-centered RMSNorm
- GELU tanh approximation
- GeGLU
- QK-Norm
- RoPE local/global
- sliding-window mask
- causal mask
```

Optional gegen Python/Transformers für einzelne Zwischenwerte, falls leicht extrahierbar.

## Nicht erlaubt

```text
- kein Produkt-Fallback über Java-Reference
- keine Workbench-Integration
- keine Performance-Optimierung
- keine Qwen-/Smol-/T5-Änderung
```

## Akzeptanz

* Gemma-Math ist gerätefrei testbar.
* Tests zeigen die Unterschiede zu Qwen/Smol.
* Reference-Code bleibt klar als Test-/Diagnosepfad markiert.
* Eigener Commit.
* Danach stoppen.

---

# Slice GEMMA-WARP-5: WARP Primitive 1 — zero-centered RMSNorm + QK-Norm

## Ziel

Erste kleine echte Gemma-WARP-Bausteine implementieren.

Nicht den ganzen ForwardPass.

## Zu bauen

```text
Gemma3WarpRmsNormKernel
Gemma3WarpQKNormKernel
```

oder entsprechende reusable Elementwise-Kernel, falls bestehende Infrastruktur passt.

## Anforderungen

```text
- zero-centered RMSNorm: Gewicht als (1 + w) behandeln
- normale RMSNorm nicht kaputt machen
- QK-Norm getrennt testbar
- ByteBuffer-/heap-light Gewichtspfad nutzen, falls sinnvoll
```

## Tests

```text
- WARP vs Java Reference
- mehrere Shapes
- edge cases
- epsilon korrekt
```

## Nicht erlaubt

```text
- keine Attention
- kein MLP
- keine vollständige Generierung
- keine Workbench-Änderung
```

## Akzeptanz

* RMSNorm/QK-Norm WARP numerisch plausibel gegen Reference.
* Keine Regression bei Qwen/Smol/T5.
* Eigener Commit.
* Danach stoppen.

---

# Slice GEMMA-WARP-6: WARP Primitive 2 — GeGLU + GELU-tanh MLP

## Ziel

Gemma-MLP korrekt als WARP-Baustein implementieren.

Gemma nutzt nicht SwiGLU/SiLU wie Qwen/Smol.

## Zu bauen

```text
Gemma3WarpGeluTanhKernel
Gemma3WarpGeGluKernel
Gemma3WarpMlp
```

oder eine passende Kombination aus bestehenden DenseProjection + neuem Activation-Gate-Kernel.

## Anforderungen

```text
- gate_proj
- up_proj
- GeGLU
- GELU tanh approximation
- down_proj
```

## Tests

```text
- WARP vs Reference
- kleine Shapes
- realistische Shapes aus Gemma Config
- ByteBuffer Upload für FP32-Gewichte, falls Quelle FP32
```

## Nicht erlaubt

```text
- keine Attention
- keine vollständige Generierung
- keine Workbench-Änderung
```

## Akzeptanz

* MLP-Baustein numerisch gegen Reference.
* Qwen/Smol/T5 unverändert.
* Eigener Commit.
* Danach stoppen.

---

# Slice GEMMA-WARP-7: WARP Primitive 3 — RoPE local/global + Sliding Window Attention

## Ziel

Gemma-Attention-Bausteine vorbereiten.

## Zu bauen

```text
Gemma3RoPE
Gemma3LocalGlobalAttentionLayout
Gemma3SlidingWindowMask
Gemma3WarpAttentionStep
```

## Zu beachten

```text
- duale RoPE-Parameter
- local attention
- global attention
- sliding window
- causal masking
- GQA / kv-heads
- head_dim aus config, nicht hidden/heads hart ableiten
```

## Tests

```text
- Maskenlogik gerätefrei
- RoPE gegen Reference
- kleine Attention gegen Reference
- local/global Layer korrekt erkannt
```

## Nicht erlaubt

```text
- keine vollständige Gemma-Runtime
- keine Workbench-Integration
- keine Qwen-/Smol-/T5-Migration
```

## Akzeptanz

* Attention-Grundlogik ist separat testbar.
* Kein Einfluss auf andere Modelle.
* Eigener Commit.
* Danach stoppen.

---

# Slice GEMMA-WARP-8: Single-Layer Gemma WARP ForwardPass

## Ziel

Einen einzelnen Gemma-Layer nativ über WARP gegen Reference validieren.

Noch keine komplette Generierung.

## Zu bauen

```text
Gemma3WarpLayer
Gemma3WarpLayerWeights
Gemma3WarpForwardPass single-layer test harness
```

## Ablauf

```text
input hidden states
→ input RMSNorm
→ q/k/v projections
→ q/k norm
→ RoPE
→ attention
→ post-attention norm
→ MLP pre-norm
→ GeGLU MLP
→ post-ff norm
→ output hidden states
```

## Tests

```text
- tiny synthetic layer
- optional real Gemma layer 0 mit kurzem Prompt
- WARP vs Reference
```

## Nicht erlaubt

```text
- keine komplette Generation
- keine Workbench-Integration
- keine Default-Umschaltung
```

## Akzeptanz

* Ein Gemma-Layer läuft nativ über WARP.
* Numerische Abweichung dokumentiert.
* Eigener Commit.
* Danach stoppen.

---

# Slice GEMMA-WARP-9: Full Prefill ForwardPass ohne Decode Loop

## Ziel

Gemma vollständigen Prompt-Prefill nativ ausführen und Logits für die letzte Position erzeugen.

Noch kein Streaming-Decode.

## Zu bauen

```text
Gemma3WarpModel
Gemma3WarpForwardPass
Gemma3WarpWeights
Gemma3LmHead
```

## Wichtig

Der große LM-Head ist potenziell der Hauptkostentreiber.

Bitte heap-light arbeiten:

```text
- ByteBuffer Upload
- tied embedding/lm_head nicht unnötig doppelt materialisieren
- keine großen float[]-Kopien, wenn vermeidbar
```

## Validierung

Gegen Transformers:

```text
- gleicher Prompt
- Top-1 oder Top-K nächste Token vergleichen
- grobe Logit-Parity dokumentieren
```

## Nicht erlaubt

```text
- keine Workbench-Integration
- keine Default-Umschaltung
- kein external-python-Pfad entfernen
```

## Akzeptanz

* Native WARP Prefill erzeugt plausibles nächstes Token.
* Top-K gegen Transformers dokumentiert.
* Eigener Commit.
* Danach stoppen.

---

# Slice GEMMA-WARP-10: Decode Session + KV Cache

## Ziel

Gemma bekommt eine native DecodeSession mit KV Cache.

## Zu bauen

```text
Gemma3DecodeSession
Gemma3KvCache
Gemma3GenerationLoopAdapter
Gemma3StopTokenPolicy
Gemma3TokenSelector
```

Wiederverwendung prüfen:

```text
- DecoderOnlyGenerationLoop, falls math-unabhängig nutzbar
- GenerationSummary / GenerationFinishReason
- gemeinsamer Streaming-Consumer
```

## Wichtig

KV Cache muss Gemma-Attention korrekt abbilden:

```text
- local attention / sliding window
- global attention
- kv-heads
- head_dim
```

## Tests

```text
- prefill + decodeNext
- kurze Generierung
- Stop token
- max tokens
- streaming callback
```

## Nicht erlaubt

```text
- keine Workbench-Default-Umschaltung
- kein Entfernen external-python
- keine Qwen-/Smol-/T5-Migration
```

## Akzeptanz

* Gemma kann nativ mehrere Tokens generieren.
* Ergebnis grob gegen Transformers plausibel.
* Eigener Commit.
* Danach stoppen.

---

# Slice GEMMA-WARP-11: Workbench Native Runtime Integration hinter Flag

## Ziel

Gemma in der Workbench über native Java/WARP Runtime starten können, aber nur explizit.

## Verhalten

Neuer Runtime-Modus, z. B.:

```text
-Dgemma.runtime=native
-Dgemma.runtime=external
```

oder interne Auswahl:

```text
external-python-transformers
native-warp-experimental
```

Default zunächst:

```text
external-python-transformers
```

oder, falls gewünscht:

```text
native-warp-experimental nur bei explizitem Flag
```

## Workbench-Logging

Anzeigen:

```text
Runtime mode: native-warp-experimental
Model id: google/gemma-3-270m-it
Model directory: ...
Package: model.wdmlpack
Backend: WARP
```

## Nicht erlaubt

```text
- kein Default auf native ohne Freigabe
- external-python-Pfad nicht entfernen
- kein Fine-Tuning
- keine Downloader-Änderung
```

## Akzeptanz

* Workbench kann Gemma native WARP experimentell starten.
* External-Probe bleibt nutzbar.
* Fehler klar, wenn .wdmlpack fehlt.
* Eigener Commit.
* Danach stoppen.

---

# Slice GEMMA-WARP-12: Performance- und Heap-Vergleich

## Ziel

Entscheiden, ob native Gemma WARP überhaupt produktrelevant ist.

## Messen

Auf mindestens zwei Profilen, soweit verfügbar:

```text
- Hauptmaschine
- gehärtetes Intel-Notebook
```

Metriken:

```text
- load time
- prefill time
- decode tok/s
- total generation time
- heap usage grob
- native memory grob
- output tokens
- finish reason
```

Vergleich:

```text
Gemma native WARP
Gemma external Python
SmolLM2-360M WARP
Qwen2.5-Coder-0.5B WARP
```

## Entscheidung

```text
GO:
  Native Gemma ist schnell genug und qualitativ als Spezialmodell interessant.

WAIT:
  Runtime funktioniert, aber Qualität braucht Fine-Tuning.

NO-GO:
  LM-Head/Architektur macht Gemma trotz 270M zu langsam oder zu schwach.
```

## Nicht erlaubt

```text
- keine Default-Umschaltung
- keine Optimierung im Mess-Slice
- keine Codeänderung außer Doku/Harness
```

## Akzeptanz

* Messbericht in docs.
* Keine erfundenen Zahlen.
* Eigener Commit.
* Danach stoppen.

---

# Spätere optionale Blöcke

## MODEL-GEMMA-FINETUNE-1: Fine-Tuning-Plan

Nur Doku/Plan:

```text
docs/model-candidate-gemma3-270m-finetune-plan.md
```

Ziel:

```text
- Input-Echo vermeiden
- echte JSON-Werte statt Platzhalter
- Natural/ADABAS-Verständnis verbessern
- kurze kontrollierte Antworten
- technische deutsche Erklärungen
```

Kein Training im Repo.

## MODEL-GEMMA-DOWNLOADER-1: Produkt-Downloader härten

Nur später, wenn native Runtime oder Fine-Tune wirklich weiterverfolgt wird.

Themen:

```text
- HF_TOKEN sauber verwenden
- 401/403 verständlich erklären
- Gemma gated/Lizenzhinweis
- keine Gewichte bundlen
- keine Terms verstecken
```

## GEMMA-WARP-OPT-1: LM-Head Optimierung

Nur falls native Runtime funktioniert, aber LM-Head dominiert.

Mögliche Themen:

```text
- tied embedding/lm_head nicht doppelt hochladen
- ByteBuffer/mmap direkt nutzen
- top-k fused projection prüfen
- quantisierte LM-head-Variante prüfen
- QAT separat analysieren
```

---

# Globale Akzeptanzregeln für alle Slices

Jeder Slice:

```text
- eigener Commit
- kein Force-Push
- kein Rebase/Squash ohne Auftrag
- danach stoppen
- Bericht mit Scope, Tests, Risiken
```

Immer grün halten:

```text
qwen
decoderonly
smollm2
t5
generation
model
warp
encoder/reranker
SmolLM2NativeWarpExecutorTest
```

Nicht brechen:

```text
Qwen bleibt Default decoder-only session
Qwen Legacy bleibt -Dqwen.runtime=legacy
Produktstart bleibt ohne jdk.incubator.vector Modul möglich
external-python Gemma-Probe bleibt erhalten, bis native WARP bewiesen ist
```

---

# Strategische Entscheidung

Gemma 3 270M-it ist aktuell:

```text
Speed:
  interessant, gemessen schneller als SmolLM2-360M

Out-of-the-box-Qualität:
  nicht gut genug für fertigen Spezialassistenten

Fine-Tuning-Potenzial:
  interessant

Native Runtime:
  sinnvoll, aber nur inkrementell und experimentell
```

Daher nicht direkt:

```text
"Gemma als fertigen Qwen-Ersatz einbauen"
```

sondern:

```text
"Gemma als schnellen CPU-Spezialmodell-Kandidaten nativ lauffähig machen,
danach entscheiden, ob Fine-Tuning + Runtime produktwürdig sind."
```

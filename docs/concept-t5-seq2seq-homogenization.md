# Konzept: T5/seq2seq-Homogenisierung — Slice T5-1 (Bestandsaufnahme + Skelett)

> **Status:** Analyse-Slice. Kein Runtime-Umbau, keine Verhaltensänderung. T5 bleibt unverändert.
> **Scope:** `directml-inference` / `t5` ↔ gemeinsame untere Bausteine. Neue `seq2seq`-Paketstruktur vorbereitet.
> **Abgrenzung:** Der Qwen/SmolLM2-`decoderonly`-Block ist abgeschlossen (siehe `docs/decoderonly-homogenization.md`)
> und wird hier **nicht** angefasst.

## 1. Warum T5 NICHT `decoderonly` ist

T5 ist **Encoder-Decoder** (seq2seq):

```text
encoder(input) → encoder hidden states
loop: decoder step (self-attn + cross-attn auf encoder states) → lm head → token select → stop
```

Die Generierungs-**Loop** unterscheidet sich strukturell von decoder-only (Cross-Attention auf Encoder-Output,
separater Decoder-Cache, Relative-Position-Bias). `DecoderOnlyGenerationLoop` ist daher **nicht** wiederverwendbar.
Die Gemeinsamkeit liegt **unter** der Loop (Kernels, Projektionen, Package, Streaming, Profiling), nicht in ihr.

## 2. Ist-Stand T5 (Bestandsaufnahme)

T5 ist **kein Stub** — es ist eine reale, verdrahtete Runtime (einige Javadocs sagen noch „shell/future", der Code
ist aber echt: `T5GenerationLoop.generate` fährt Encoder→Decoder-Step→LM-Head→Select; `T5Runtime` hat Referenz- und
mehrere WARP-Boundary-Lademodi inkl. `loadWarp` als Produktions-Entrypoint; `T5InferenceEngine` + Tests existieren).

### 2a. Family-spezifisch — bleibt in `t5/` (nicht teilbar)
Encoder/Decoder-Pipelines (Referenz + WARP): `T5EncoderPipeline`, `T5DecoderPipeline`, `T5WarpEncoderPipeline`,
`T5WarpDecoderPipeline`, `T5EncoderRunner`/`T5DecoderRunner`, `T5EncoderBlock`/`T5DecoderBlock(Step)`; Attention:
`T5SelfAttention`, `T5CrossAttention`, `T5RelativePositionBias`, `T5*AttentionProjection/Memory*`; Zustand:
`T5DecoderCache`, `T5DecoderState`, `T5EncoderOutput`, `T5SelfAttentionMemory`; Bausteine: `T5LayerNorm` (T5-RMSNorm
ohne Bias/Mean-Subtraktion), `T5FeedForward`, `T5LmHead`/`T5WarpLmHead`/`T5LogitProjector`; Config/Layout/Tensorrollen:
`T5Config`, `T5Architecture`, `T5TensorRole/NameMapper`, `T5Layout*`, `T5Weights`/`T5WeightResolver`; Tokenizer:
`JsonT5Tokenizer`, `CodeT5Tokenizer`, `T5TextTokenizer`, `T5TokenizerLoader`; Generierungsloop `T5GenerationLoop`.

### 2b. Bereits geteilt (unterste Schicht — gut!)
- WARP-Kernel: `MatMulNBitsKernel` (Windows-Bindings) — von T5 **und** decoderonly genutzt.
- Streaming: `GenerationTokenSink` / `GeneratedToken` (inference-root) — T5 nutzt sie bereits.
- Prompt-Pipeline: `prompt/PromptStrategy` inkl. `T5PromptStrategy` — projektweit geteilt.
- Modell-/Tensor-Basis: `model/` (`RuntimeTensor`, `SourceTensorDataType`).

### 2c. Teilbar, aber heute dupliziert (Homogenisierungs-Ziele)
- **WARP-Dense-Projektion:** `T5WarpLinearProjection` und `DecoderOnlyWarpDenseProjection` umhüllen **denselben**
  `MatMulNBitsKernel.fromDequantizedWeights(...)` + `matvec`/`matmulBatch` — fast identischer Adapter (Namen,
  Input/Output-Validierung, Batch-Fallback). **Klarster geteilter Baustein.**
- **Token-Selektion (greedy/argmax):** `GreedyT5TokenSelector` (argmax + NaN-Sanitize) ↔ `DecoderOnlyMath.argmax`.
- **Stop-Handling:** `T5StopTokenPolicy` (Set-basiert, `shouldStop(int)`) ↔ `DecoderOnlyStopTokenPolicy` (funktional).
- **Result/Profiling:** `T5RuntimeResult` + `T5GenerationMetrics` ↔ `DecoderOnlyGenerationResult` (Form unterschiedlich:
  T5 hat Encoder/Decode/LM-Head/TokenSelect-Metriken — eher angleichbar als 1:1 teilbar).
- **Package/wdmlpack:** `T5WdmlPackCompiler`/`-CompileTool`/`T5RuntimePackage` ↔ analoge decoderonly/family-Lifecycles.

## 3. Wichtige Erkenntnis zur `seq2seq`-Schicht

Es gibt aktuell **nur eine** seq2seq-Familie: T5. „CodeT5" ist **kein** eigener Runtime, sondern eine
**Tokenizer-Variante** (`CodeT5Tokenizer`) auf derselben T5-Runtime. Eine generische `seq2seq`-Loop/Config/Session-
Abstraktion hätte daher **nur einen Implementierer** → das wäre tote Duplikation (genau das, was vermieden werden
soll). Anders als bei `decoderonly` (zwei echte Familien Qwen+SmolLM2) liegt der Homogenisierungswert hier **nicht**
in einer familienübergreifenden seq2seq-Schicht, sondern im **Teilen der unteren Bausteine** zwischen seq2seq(T5) und
decoderonly(Qwen/SmolLM2).

Daraus folgt für dieses Slice (bewusst „nicht alles blind anlegen"):

- **Angelegt:** `seq2seq`-Paket mit `package-info` als dokumentierte Heimat/Grenze der seq2seq-Schicht (T5/CodeT5).
- **NICHT angelegt:** `Seq2SeqConfig/Tokenizer/EncoderSession/DecoderSession/GenerationLoop/GenerationResult/
  StopPolicy/RuntimeMode` — T5 hat funktionierende Äquivalente; generische Parallelen jetzt wären Dead Code. Sie
  kommen erst, falls/ wenn ein zweiter seq2seq-Runtime entsteht oder ein konkreter geteilter Baustein sie erzwingt.

## 4. Vorgeschlagene Slice-Reihenfolge

| Slice | Inhalt | Risiko |
|------|--------|--------|
| **T5-2** | Neutralen geteilten **WARP-Dense-Projektion**-Baustein einführen; **T5** darauf umstellen (`T5WarpLinearProjection` → Adapter/Delegate). decoderonly bleibt vorerst unberührt (eigener späterer Slice), um den abgeschlossenen Block nicht anzufassen. | niedrig (reiner Adapter über denselben Kernel; T5-Tests + WARP-E2E als Gate) |
| **T5-3** | Geteiltes **Token-Selection-/Stop-Seam** für T5 (argmax + stop) — T5 adoptiert die neutralen Verträge. | niedrig |
| **T5-4** | **Result/Profiling**-Begriffe angleichen (gemeinsame Felder/Streaming), ohne T5-Metrik-Semantik zu verlieren. | mittel |
| **T5-5** | **Package/wdmlpack**-Lifecycle-Angleichung (CompilerSupport), familienspezifische Tensorrollen bleiben. | mittel |
| später | decoderonly auf denselben neutralen Baustein heben (eigener Slice, da der decoderonly-Block sonst angefasst wird). | mittel |

> Regel: jeweils klein, verhaltensneutral, mit T5-Tests + (wo Modell nötig) gated WARP-E2E als Gate. Kein generischer
> seq2seq-Loop-/Config-Umbau, solange T5 die einzige seq2seq-Familie ist.

## 5. Erste Anbindung & Risiken

- **Zuerst anbinden:** `T5WarpLinearProjection` (Slice T5-2). Klarste Duplizierung, reiner Adapter über
  `MatMulNBitsKernel`, gut testbar; berührt weder Loop noch decoderonly.
- **Risiken:**
  - T5 ist reif und hat **viele** WARP-Ausführungsmodi (`load*`-Kombinationen aus Encoder/Decoder/LM-Head); jeder
    geteilte-Baustein-Tausch muss alle T5-Tests **und** den WARP-Pfad byte-identisch lassen.
  - Veraltete „shell/future"-Javadocs dürfen nicht zu der Fehlannahme verleiten, T5 sei ein Stub — der Code ist real.
  - Echte T5-WARP-E2E-Verifikation braucht ein lokales T5-Modell (z. B. `flan-t5-small`/`codet5-*` im `.directml`-
    Cache) und WARP — analog zum Qwen-Runbook; nicht in normaler CI.
  - Ein geteilter Baustein, den nur T5 (nicht decoderonly) zunächst nutzt, ist kurzzeitig „nur ein Konsument" — das
    ist akzeptiert (decoderonly folgt separat), darf aber nicht den decoderonly-Block verändern.

## 6. Was nach T5-1 NICHT getan wurde (bewusst)

Kein Runtime-Tausch, kein geteilter Baustein extrahiert, keine T5-Klasse umgehängt, keine decoderonly-Änderung. Nur
Inventur + `seq2seq`-Paketgrenze + Slice-Plan.

## 7. Slice T5-2 — erster gemeinsamer WARP-Baustein (erledigt)

Erster geteilter unterer Baustein eingeführt: **`inference/warp/WarpDenseProjection`** — der modellfamilien-neutrale
Adapter über `MatMulNBitsKernel` (`y = W*x`, rank-2 `[output, input]`). API ist die Vereinigung der Familienbedarfe:
`project`/`projectInto` (matvec), `projectSequence`/`projectSequenceInto` (batched + per-row-Fallback), `kernel()`
(für späteres GPU-resident-Chaining), `close`/`isClosed`.

- **`T5WarpLinearProjection` ist jetzt ein dünner Adapter** darüber (behält die `T5LinearProjection`-API und die
  rank-2-/Größen-Validierungen). Verhalten unverändert: gleiche Shapes, gleicher Batch-Fallback, gleiche matvec-
  Ergebnisse (gleiche Kernel-Aufrufe in gleicher Reihenfolge). Die `T5WarpLinearProjectionFactory` (plain + fused
  self-/cross-attention) baut unverändert darüber.
- **decoderonly bleibt unangetastet** — `DecoderOnlyWarpDenseProjection` ist API-deckungsgleich mit dem neuen Baustein
  (selber Kernel, selbe Methoden `project/projectInto/projectSequenceInto/kernel/close`) und kann später **1:1** als
  eigener Slice darauf gehoben werden, ohne den abgeschlossenen Block jetzt zu berühren.
- Tests: `warp/WarpDenseProjectionTest` (device-frei: Shape-Validierung; gated WARP: matvec/sequence == Referenz) und
  `t5/T5WarpLinearProjectionTest` (device-frei: rank-2-Check; gated WARP: apply/applySequence == Referenz).

## 8. Slice T5-2b — decoderonly adoptiert den gemeinsamen Baustein (erledigt)

`DecoderOnlyWarpDenseProjection` ist jetzt ebenfalls ein **dünner Adapter** über `WarpDenseProjection` — damit ist der
Baustein **real gemeinsam** (T5 **und** decoderonly), nicht nur T5-intern. Die `DecoderOnlyDenseProjection`-API für
Aufrufer (Qwen/SmolLM2) ist unverändert: `fromRowMajorWeights`, `project`/`projectInto`/`projectSequenceInto`,
`kernel()`, `close`/`isClosed` verhalten sich identisch (gleiche Kernel-Aufrufe, gleicher Per-Row-Fallback, gleiches
`kernel()` für das MLP-Block-Chaining). `DecoderOnlyWarpFusedDenseProjection` und der MLP-Block wurden **nicht**
geändert. Verifiziert: `SmolLM2NativeWarpExecutorTest` (numerisch gleich) + decoderonly/qwen/smollm2/t5-Suiten grün;
device-freie Shape-Validierung ergänzt. Qwen-Default (decoder-only session) + Legacy-Fallback unverändert.

## 9. Slice T5-3 — Stop-Token-Ergebnisvertrag angeglichen (erledigt)

T5 nutzt jetzt denselben **Stop-Token-Ergebnisvertrag** wie decoderonly. Geändert wurde nur die Buchführung in
`T5GenerationLoop` (Stop-Check vor add/stream/cache-append) + ein Feld in `T5RuntimeResult`; Encoder/Decoder/Cross-
Attention/Token-Selektor/Stop-Policy bleiben unangetastet.

- **Stop-Token** beendet die Generierung, landet aber **nicht** mehr in `outputTokenIds`, wird **nicht** gestreamt.
- **Finish reason** bleibt `FinishReason.stop_token` (T5s eos-Äquivalent; das Enum wurde nicht umbenannt).
- **`T5RuntimeResult.finishTokenId()`** trägt das terminierende Stop-Token separat (sonst `NO_FINISH_TOKEN` = -1).
- Seams sind die bereits vorhandenen `T5TokenSelector` (greedy/argmax) + `T5StopTokenPolicy` (`shouldStop`); keine neue
  Klasse nötig, kein generischer Seq2Seq-Loop.

**Sichtbarer Output ist bei EOS jetzt ein Token kürzer** (das EOS-Token entfällt) — analog decoderonly/Qwen. Der
`max_tokens`-Fall ist unverändert (alle Tokens, `finishTokenId = -1`). `T5InferenceEngine` nutzt `greedyText`
(EOS-Unterdrückung für Mindestschritte), daher bleibt `completionTokens ≥ 1`. Gerätefreie Tests in
`T5GenerationLoopTest` decken greedy-Selektion, Stop-Erkennung, Nicht-Emission/Nicht-Streaming des Stop-Tokens und den
`length`-Fall ab.

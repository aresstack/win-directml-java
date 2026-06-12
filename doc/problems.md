# Problems / Blockers — Engine Cleanup Plan abarbeitung

Dieser Log sammelt Punkte aus `doc/todo.md`, die nicht (vollständig) erledigt werden konnten,
mit Grund und Lösungsalternativen. Erledigte Punkte stehen kurz mit Commit-Hash; offene mit Begründung.

Reihenfolge entspricht `doc/todo.md`.

---

## Item 3 — Fused Projection heap-light: T5-Fusion noch float[] (Teil offen)

**Stand:** Der **decoder-only/SmolLM2**-Teil ist erledigt (Commit folgt): SmolLM2 `qkv` und `gate_up` nutzen jetzt
`DecoderOnlyWarpFusedDenseProjection.fromWeightSourceParts` → bei all-FP32-ByteBuffer-Teilen einen
**Device-Region-Stack** (`MatMulNBitsKernel.fromFusedFp32ByteBuffers`, mehrere `CopyBufferRegion` in einen Ziel-Buffer,
kein Java-Heap-`float[]`-Fused-Array). Byte-identisch zum float[]-Concat verifiziert + SmolLM2-Native grün.

**Offen:** Die **T5-Fusion** (`T5FusedSelfAttentionProjection` / `T5FusedCrossAttentionMemoryProjection`) baut weiterhin
ein Host-`float[]` und verpackt es in `T5TensorData.reference(...)`; danach baut `T5WarpLinearProjectionFactory` daraus
eine Einzel-Projektion. Dadurch greift der ByteBuffer-Vorteil für T5-fused (Q/K/V, Cross-Attn K/V) noch nicht.

**Warum offen gelassen:** Die T5-Fusion ist über `T5WarpLinearProjectionFactory.createSelfAttentionProjection` /
`createCrossAttentionMemoryProjection` mit den T5-Attention-Projektionstypen verzahnt und T5 wird in den Items 6/7
ohnehin überarbeitet. Eine Umstellung jetzt würde die T5-Attention-Verdrahtung anfassen, bevor der T5-Engine-/
Status-Umbau (Item 6/7) entschieden ist — Risiko ohne Mehrwert.

**Lösungsalternative:** Im Zuge von Item 7 (T5-Engine) `T5WarpLinearProjectionFactory` so umbauen, dass die fused
Self-/Cross-Attn-Projektionen eine `List<WarpWeightSource>` aus den zugrunde liegenden `T5TensorData`-FP32-Quellen
bilden und `WarpDenseProjection.fromFusedFp32(...)` nutzen (statt `T5TensorData.reference(float[])`). Die Infrastruktur
(`fromFusedFp32`, `fromWeightSourceParts`) steht bereits und ist getestet.

---

## Status-Übersicht aller Items

| Item | Status | Commit / Hinweis |
|------|--------|------------------|
| 0 Repo-Stand | erledigt | HEAD==origin/main==c321b0a, sauber |
| 1 Lifecycle/close() | **erledigt** | 96d92fb |
| 2 Weight-Source-Vertrag | **erledigt** | 1babd51 |
| 3 Fused heap-light | **teilweise** | e55b524 (decoder-only/SmolLM2 fertig; T5-fused offen, s.o.) |
| 4 Decoder-only Grenzen | **teilweise** | s.u. (Qwen geteilt; SmolLM2-Loop/Stop noch family-local) |
| 5 Qwen INT4/Fusion | **teilweise** | 30e2274 (Diagnostik fertig; bit-shift/ByteBuffer-Fusion offen) |
| 6 T5 Manifest-Status | **offen** | s.u. (hängt an Item 7) |
| 7 T5 Engine | **offen** | s.u. (groß, braucht Modell + Korrektheit) |
| 8 Generation-Contract Workbench | **offen** | s.u. (medium, Workbench-Panels) |
| 9 model/wdmlpack-Lifecycle | **weitgehend erfüllt** | s.u. (durch H-Block) |
| 10 Encoder/Reranker-Status | **weitgehend erfüllt** | s.u. (durch ER-2) |
| 11 WARP-Profiling | **teilweise** | s.u. (Decode-Profile vorhanden; Heap/Fallback-Zähler ergänzt) |
| 12 Schluss-Cleanup | **bewusst später** | per Plan erst nach Engine-Stabilisierung |

---

## Item 4 — SmolLM2 hat noch family-local Generation-Loop/Stop-Vertrag

**Stand:** Qwen nutzt den geteilten `DecoderOnlyGenerationLoop` (+ `DecoderOnlyGenerationResult`) im Default
`decoder-only session`. **SmolLM2 dagegen** hat eigene `SmolLM2WarpGenerationLoop`, `SmolLM2ReferenceGenerationLoop`
und `SmolLM2StopTokenPolicy` — also eine zweite Loop-/Stop-Implementierung.

**Warum nicht erledigt:** Die Migration von SmolLM2 auf `DecoderOnlyGenerationLoop` + den geteilten Stop-Vertrag ist
ein **verhaltenssensitiver Refactor** (Token-Sampling, Stop-Handling, Streaming-Callback). Akzeptanz verlangt
„kein Verhalten ändert sich in den SmolLM2-Smokes" (greedy token-für-token + Streaming identisch). Das sauber +
verifiziert umzubauen ist eine eigene Slice-Serie; ein Teilumbau würde die SmolLM2-Generierung gefährden.

**Lösungsalternative:** Eigener Slice „SmolLM2 → DecoderOnlyGenerationLoop": SmolLM2-Sampler/Stop hinter die
decoderonly-Typen (`DecoderOnlyTokenSelector`, `DecoderOnlyStopTokenPolicy`) adaptieren, dann
`SmolLM2WarpGenerationLoop`/`SmolLM2ReferenceGenerationLoop` durch den geteilten Loop ersetzen, abgesichert über
`SmolLM2NativeWarpExecutorTest` (token-/streaming-identisch). Qwen dient als Vorlage (bereits migriert).

## Item 5 — Qwen INT4 Restarbeiten (über Diagnostik hinaus)

**Erledigt:** Sichtbare INT4-fused-vs-FP32-fallback-Zähler + WARN (30e2274).
**Offen:** (a) Bit-shift-Pfad für packed-nibble `zeroPoints` in `tryFuseQuantizedRowwise` (heute Fallback statt
implementiert); (b) ByteBuffer-/packed-source-Overloads von `MatMulNBitsKernel`, damit Qwen-Fusion ohne FP32-Heap
baut; (c) getrennte LM-Head-INT4-Analyse.
**Warum offen:** Qwen-INT4 ist laut Plan ein **eigener späterer Block**; diese Punkte sind native, paritäts-kritisch
(Token-Identität) und brauchen das echte Qwen-0.5B-Modell zur Verifikation.
**Alternative:** Eigener Qwen-INT4-Slice mit dem lokalen `qwen2.5-coder-0.5b-directml-int4` als Paritäts-Gate.

## Items 6 + 7 — T5 Manifest-Status + Engine

**Warum zusammen offen:** Item 6 (Manifest `runtimeLoadable`/`runtimeLoadMode`/`reason` korrigieren) hängt direkt an
Item 7: ob `runtimeLoadable=true` ehrlich gesetzt werden darf, hängt davon ab, ob die T5-Engine den jeweiligen
Kandidaten **wirklich** ausführen kann. Item 7 (T5 Encoder/Decoder/Cross-Attention/Cache fachlich fertigstellen oder
klar abgrenzen) ist die größte verbleibende Engine-Baustelle und braucht ein lokales T5-Modell (flan-t5-small/t5-small
liegen vor) plus echte Ausgabe-Verifikation.
**Warum nicht jetzt:** Groß, korrektheits- und modellabhängig; nicht in einem additiven Schritt sicher machbar, ohne
T5-Generierung zu riskieren. Außerhalb des sicheren Rahmens dieser autonomen Abarbeitung.
**Alternative (empfohlene Slice-Folge):**
1. T5 Statusmodell schärfen: getrennte Flags `weightsLoadable`/`runtimeLoadable`/`executable` + ehrlicher `reason`
   (statt „runtime not implemented" pauschal in `runtimeLoadable=false` zu verstecken). Tests entsprechend.
2. Einen konkreten lokalen T5-Kandidaten end-to-end verifizieren (echte, nicht-leere Ausgabe), Grenze WARP vs
   Reference dokumentieren/testen.
3. Erst dann T5-fused heap-light (Item 3 Rest) mitziehen.

## Item 8 — Gemeinsamer Generation-Contract in Workbench/API

**Stand:** Die gemeinsamen Typen (`GenerationSummary`, `GenerationFinishReason`, `RuntimeLoadability`) existieren; die
Familien-Runtimes liefern sie teils schon (Qwen/SmolLM2/T5 `toSummary()`), aber `SummarizerPanel` /
`SmolLM2WorkbenchRuntimeRunner` / `T5InferenceEngine` / `QwenInferenceEngine` lesen teils noch familienspezifische
Felder/Strings.
**Warum nicht jetzt:** Medium-Umbau über mehrere Workbench-Panels + Runner; Risiko, die Nutzer-sichtbare Ausgabe zu
verändern (Akzeptanz: „bestehende Ausgaben bleiben verständlich", keine UI-Neugestaltung). Braucht sorgfältige
Panel-für-Panel-Umstellung mit Sichtprüfung, die in dieser autonomen Runde nicht verifizierbar ist (Swing-GUI).
**Alternative:** Eigener Slice: Panels/Runner auf `GenerationSummary`/`RuntimeLoadability` umstellen, alte String-Felder
nur an API-Kompatibilitätsgrenzen behalten; Workbench-Smoke (headless, soweit möglich) + manuelle Sichtprüfung.

## Item 9 — model/wdmlpack-Lifecycle (weitgehend erfüllt durch H-Block)

`WdmlPackReader` ist die einzige mmap-Stelle; `RuntimeTensor` ist der bevorzugte ByteBuffer-Lieferant (T5/SmolLM2
ziehen FP32-Quellen daraus, Item 2/3). mmap-/ByteBuffer-Lifetime ist in `docs/heap-light-wdmlpack.md` (H1/H2)
explizit dokumentiert: Reader mmappt zero-copy, Familien-Wrapper sind dünne Views, Upload ist synchron während des
Ladens. **Offen/optional:** ein automatischer „kein dangling mmap-View nach Package-close"-Test wäre eine zusätzliche
Absicherung; aktuell ist die Lifetime durch synchronen Upload + Halten der Views in den Tensor-Wrappern abgedeckt.
**Alternative:** Bei Bedarf ein gezielter Lifetime-Test (Package öffnen, Views ziehen, close, Verhalten prüfen) als
kleiner Folge-Slice.

## Item 10 — Encoder/Reranker-Status (weitgehend erfüllt durch ER-2)

`ModelAssetValidation` (ER-2, commit 3829a10) ist die einheitliche, differenzierte Statusquelle (not downloaded /
incomplete / corrupt / wrong variant), genutzt von E5/Reranker-Loadern + dem Download-Vollständigkeitscheck. MiniLM/
E5/Reranker-Suiten waren in dieser Sitzung grün. **Offen/optional:** die Status-Stufen `loadable`/`executable` explizit
in die Workbench-Anzeige ziehen (überlappt mit Item 8). Keine neuen Encoder-Architekturen (bewusst).

## Item 11 — WARP-Profiling (teilweise)

`DecoderOnlyWarpDecodeProfile` misst bereits Decode-/MLP-/SwiGLU-/LM-Head-Anteile; SmolLM2/Qwen nutzen es. Item 5
ergänzt die INT4-vs-FP32-Fusion-Zähler. **Offen:** ein konsolidierter Profiling-Bericht (Load/Prefill/Decode-pro-Token/
LM-Head-Anteil/Heap vor-nach-Load/Fallback-Zähler) als ein Ausgabeblock pro Familie, plus die Heap-Messung. **Warum
nicht jetzt:** sinnvoll erst nach den Engine-Stabilisierungen (4/6/7/8); braucht reale Modelle zur Aussagekraft.
**Alternative:** Eigener Profiling-Slice nach 4/7, der die vorhandenen Profile + Item-5-Zähler + eine Heap-Messung in
einen Workbench-Runner-Bericht bündelt.

## Item 12 — Schluss-Cleanup

Per Plan **bewusst erst nach** Engine-Stabilisierung (Doku/Logging/Release-Härtung, Legacy-Entfernung). Nicht jetzt
angefasst — kein Cleanup soll Diagnose entfernen, die für 4/6/7/8/11 noch gebraucht wird, und kein Legacy-Pfad
vorzeitig gelöscht werden.


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
| 4 Decoder-only Grenzen | **erledigt** | WARP+Reference auf geteiltem DecoderOnlyGenerationLoop; SmolLM2StopTokenPolicy entfernt |
| 5 Qwen INT4/Fusion | **teilweise** | 30e2274 (Diagnostik fertig; bit-shift/ByteBuffer-Fusion offen) |
| 6 T5 Manifest-Status | **erledigt** | T5-1: weightsLoadable/runtimeLoadable/executable getrennt; keine „runtime not implemented"-Lüge mehr |
| 7 T5 Engine | **teilweise** | T5-2: Reference-Engine end-to-end auf echtem t5-small verifiziert → executable=true (Reference); WARP-T5-Pfad offen, s.u. |
| 8 Generation-Contract Workbench | **offen** | s.u. (medium, Workbench-Panels) |
| 9 model/wdmlpack-Lifecycle | **weitgehend erfüllt** | s.u. (durch H-Block) |
| 10 Encoder/Reranker-Status | **weitgehend erfüllt** | s.u. (durch ER-2) |
| 11 WARP-Profiling | **teilweise** | s.u. (Decode-Profile vorhanden; Heap/Fallback-Zähler ergänzt) |
| 12 Schluss-Cleanup | **bewusst später** | per Plan erst nach Engine-Stabilisierung |

---

## Item 4 — Decoder-only Engine-Grenzen: ERLEDIGT

**Stand:** SmolLM2 hat **keine zweite Generation-/Stop-Engine** mehr.
- **WARP:** `SmolLM2WarpGenerationLoop` ist (war bereits) ein dünner Adapter über `DecoderOnlyGenerationLoop`
  (`new DecoderOnlyGenerationLoop(forwardPass.delegate(), ...)`).
- **Reference:** `SmolLM2ReferenceGenerationLoop` enthält **keine eigene for-Schleife** mehr — neuer Adapter
  `SmolLM2ReferenceDecoderOnlyForwardPass implements DecoderOnlyForwardPass` + `SmolLM2ReferenceDecodeSession
  implements DecoderOnlyDecodeSession` (prefill/decodeNext geben die **volle** Sequenz an
  `SmolLM2ReferenceForwardPass.logitsForLastToken(tokenIds, kvCache)`, wegen `kvCache.completedTokenCount()`;
  `lastCallLmHeadNanos` = LM-Head-Delta; idempotenter close). Der Loop wird vom geteilten
  `DecoderOnlyGenerationLoop` gefahren; das Ergebnis wird auf `SmolLM2TokenRuntimeResult` gemappt.
- **Stop:** beide nutzen `DecoderOnlyStopTokenPolicy`. `SmolLM2StopTokenPolicy` (produktiv tot) wurde **gelöscht**;
  der einzige Test (`SmolLM2RuntimeTest.stopTokenPolicyStopsOnEosToken`) auf `DecoderOnlyStopTokenPolicy` umgestellt.

SmolLM2-spezifisch bleiben nur Sampler-Factory/Options, Result-Mapping, ForwardPass-/DecodeSession-Adapter und das
Reference-Hotspot-Profil-Mapping. **Verifikation grün:** `SmolLM2NativeWarpExecutorTest` (Top-1/Logits/greedy/Streaming
+ close-Idempotenz), `SmolLM2ReferenceGenerationLoopTest`, `SmolLM2RuntimeTest`, smollm2/decoderonly/generation +
qwen/t5/model/warp + encoder. Qwen Default decoder-only session + Legacy unverändert; Produktstart ohne Vector-Modul
weiterhin möglich.

## Item 5 — Qwen INT4 Restarbeiten (über Diagnostik hinaus)

**Erledigt:** Sichtbare INT4-fused-vs-FP32-fallback-Zähler + WARN (30e2274).
**Offen:** (a) Bit-shift-Pfad für packed-nibble `zeroPoints` in `tryFuseQuantizedRowwise` (heute Fallback statt
implementiert); (b) ByteBuffer-/packed-source-Overloads von `MatMulNBitsKernel`, damit Qwen-Fusion ohne FP32-Heap
baut; (c) getrennte LM-Head-INT4-Analyse.
**Warum offen:** Qwen-INT4 ist laut Plan ein **eigener späterer Block**; diese Punkte sind native, paritäts-kritisch
(Token-Identität) und brauchen das echte Qwen-0.5B-Modell zur Verifikation.
**Alternative:** Eigener Qwen-INT4-Slice mit dem lokalen `qwen2.5-coder-0.5b-directml-int4` als Paritäts-Gate.

## Item 6 — T5 Manifest/Runtime-Status: ERLEDIGT (Slice T5-1/T5-2)

Ehrliche Trennung statt pauschalem `runtimeLoadable=false`/„runtime not implemented":
- **weightsLoadable** = Payload + komplettes Layout + unterstützte Runtime-Dtypes (unverändert berechnet).
- **runtimeLoadable** = `weightsLoadable` — `T5RuntimePackage.open()` baut bei ladbaren Gewichten `T5Weights` + die
  Runtime-Strukturen, also IST das Paket runtime-loadable (vorher hart `false`).
- **executable** = neues Manifest-Feld + `T5RuntimePackage.executable()`; ist für komplette, runtime-ladbare Payloads
  **true**, weil die Default-Reference-Engine in T5-2 end-to-end auf einem echten lokalen `t5-small` verifiziert wurde
  (nicht-leere, EOS-terminierte Ausgabe).
- **runtimeLoadMode/reason** (zentral in `T5ManifestPayloadPolicy`): `t5-manifest-only` /
  `t5-weights-not-loadable` / `t5-executable` — mit klaren Gründen. Die alten Strings
  „not-implemented"/„t5-weights-loaded-runtime-not-implemented"/„T5 runtime is not implemented yet" sind aus Writer,
  `T5SafeTensorsLayoutCompiler`, `T5LayoutManifest` und `T5RuntimePackage.fromMetadata` entfernt.

Der `executable=true`-Claim ist ausdrücklich auf die Default-Reference-Engine begrenzt. `T5Runtime.UNSUPPORTED_MESSAGE`
(WARP-Pfad) bleibt bewusst unverändert — das ist die separate Item-7-Restbaustelle und blockiert nicht die verifizierte
Reference-Ausführung. Tests umgestellt: `T5RuntimePackageLoadabilityTest`, `T5RuntimeTest`,
`T5SafeTensorsLayoutCompilerTest`, `T5WdmlPackCompilerTest` (Dry-Run-Tool-Test bleibt `runtimeLoadable=no`, da Dry-Run
kein Payload schreibt).

## Item 7 — T5 Engine (teilweise: Reference verifiziert, WARP offen)

**Erledigt (Slice T5-2):** Die **Reference**-T5-Engine ist end-to-end auf einem **echten lokalen t5-small** verifiziert.
Neuer gegateter Test `T5RealModelReferenceTest.t5SmallReferenceProducesNonEmptySummary` (`@EnabledIf`, Pfad via
`-Dt5.testModelDir` oder `model/t5-small`) fährt den vollen Pfad `T5InferenceEngine` (reference) → Tokenizer →
`T5Runtime`/`T5RuntimePackage` → Encoder (input ids/mask) → Decoder-Step-Schleife → Cross-Attention → LM-Head-Logits →
Greedy-Token-Auswahl → EOS/Stop → Detokenisierung. Ergebnis: **echte, nicht-leere, EOS-terminierte Ausgabe**
(`"the quick brown fox jumps over the lazy dog ."`, 13 Tokens, FinishReason EOS) für Eingabe
`"summarize: The quick brown fox jumps over the lazy dog."`. Damit ist **Akzeptanz A** erfüllt.

Folge: `executable=true` wird jetzt für runtime-ladbare Pakete sauber begründet gesetzt (`T5WdmlPackManifestWriter`:
`executable = runtimeLoadable`, Modus `t5-executable`, Reason „T5 reference generation verified end-to-end (non-empty,
EOS-terminated output)"). Der Claim ist **auf die Default-Reference-Engine** begrenzt; der separate **WARP-T5-Pfad**
(`T5Runtime.UNSUPPORTED_MESSAGE`) bleibt bewusst **nicht** zertifiziert und berührt diesen Paket-Status nicht.
Tests umgestellt: `T5RuntimeTest` (runtimeLoadable+executable true), `T5WdmlPackCompilerTest`
(`marksPackageAsRuntimeLoadableAndExecutable`). Manifest-only-/Dry-Run-Pfade bleiben `executable=false`.

**Offen (WARP-T5-Engine):** Der WARP-(GPU/DirectML-)Pfad für T5 ist weiterhin nicht implementiert
(`T5Runtime.UNSUPPORTED_MESSAGE`). Das ist die große, korrektheits- und performance-kritische Restbaustelle —
WARP-Encoder/Decoder/Cross-Attention + Cache mit Paritäts-Gate gegen die jetzt verifizierte Reference-Ausgabe.
**Warum nicht jetzt:** Groß, nativ, paritäts-kritisch; nicht in einem additiven Schritt sicher machbar. Erst dann darf
`executable` auch für den WARP-Pfad beansprucht werden.
**Alternative (empfohlene Slice-Folge):**
1. ✅ (T5-1) Statusmodell geschärft: `weightsLoadable`/`runtimeLoadable`/`executable` + ehrlicher `reason`.
2. ✅ (T5-2) Reference end-to-end auf lokalem t5-small verifiziert (echte, nicht-leere Ausgabe); Grenze WARP vs
   Reference dokumentiert/getestet.
3. T5-fused heap-light (Item 3 Rest) im WARP-Zug mitziehen, sobald die WARP-T5-Engine angegangen wird.
4. WARP-T5-Engine mit Paritäts-Gate gegen die Reference-Ausgabe.

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

**Folgearbeit (Artifact-Lifecycle-Block, Slice W5):** Encoder/Reranker (MiniLM/E5/Reranker) und Phi-3 haben **bewusst
noch keinen `.wdmlpack`-Compiler**. Im einheitlichen Artifact-Lifecycle laufen sie als **`PACKAGE_LEGACY_DIRECT`**
(`LegacyDirectLifecycle`): die Runtime lädt direkt aus den SafeTensors/ONNX-Quelldateien, das DownloadPanel zeigt einen
deaktivierten „Legacy (direct)"-Button + den ehrlichen Status „direct SafeTensors legacy path (package compiler not
implemented yet)" — **keine Fake-Konvertierung, kein implizites Package-Schreiben**. Der Release nutzt dort bewusst den
direct-SafeTensors-Legacy-Pfad. **Eigener Folgepunkt (nicht in diesem Block umgesetzt):** ein
Bert/MiniLM/E5-Encoder-`wdmlpack`-Compiler + ein CrossEncoder/Reranker-`wdmlpack`-Compiler + eine RuntimePackage-Facade,
damit Encoder/Reranker denselben `download → convert → run-from-package`-Ablauf wie Qwen/SmolLM2/T5 bekommen.

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


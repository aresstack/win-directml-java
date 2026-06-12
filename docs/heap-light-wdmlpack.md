# heap-light `.wdmlpack` — H1: Bestandsaufnahme + Load-Path-Mapping

> **Status:** reine Analyse/Doku. Kein Code geändert, kein Format geändert, kein Runtime-Verhalten geändert.
> **Frage:** Wo entstehen auf dem `.wdmlpack`-Ladepfad vermeidbare Host-Heap-Kopien, und welche Familie lässt sich
> am risikoärmsten zuerst heap-light machen?

## 1. Zielbild (langfristig, NICHT H1)

```
.wdmlpack  →  FileChannel/mmap ByteBuffer-Region  →  UploadBuffer  →  DeviceBuffer  →  WARP/DirectML
```

statt heute:

```
.wdmlpack  →  mmap ByteBuffer  →  float[] (decode/dequantize, teils + clone)  →  uploadFloats(float[])  →  DeviceBuffer
```

## 2. Wer öffnet/liest `.wdmlpack`

| Klasse | Rolle | Heap? |
|--------|-------|-------|
| `model/WdmlPackReader` | einzige Klasse, die das Paket **mmappt** und Payload-Offsets interpretiert | **zero-copy** (mmap `MappedByteBuffer`, read-only Slices) |
| `model/RuntimeModelPackage` | schlanker Deskriptor (Header + Manifest + Größe) | kein Payload-Heap |
| `model/WdmlPackManifest` | Manifest-Zugriff (`tensors[]`: name, dataType, dims, byteLength, payloadOffset, payloadLength) | klein (JSON-Maps) |
| `model/RuntimeTensor` | benannter Tensor = `name`, `dims`, `dataType`, **read-only `ByteBuffer`-View** + `rawByteLength` | **zero-copy** (View auf mmap) |
| `model/RuntimeTensorCatalog` | Name→`RuntimeTensor`; `payloadBytes()`; Adapter `toSourceTensorCatalog()`/`toTensorCatalog()` | Adapter erzeugen `SourceTensor.inline(... ByteBuffer)` — weiterhin **zero-copy** |

**Befund:** Die gemeinsame `model/`-Lese-/Katalogschicht ist bereits heap-light. `WdmlPackReader.mapPayloadTensors`
liefert `RuntimeTensor` mit `ByteBuffer`-Slices direkt in die gemappte Datei (kein `byte[]`/`float[]`). Auch die
Adapter (`SourceTensor`, Qwens `OnnxTensor`-Adaption in `QwenWdmlPackModelSource.adaptRuntimeTensors`) reichen nur
`ByteBuffer`-Views weiter.

## 3. Wo die Host-Kopien wirklich entstehen (Familie → Kernel-Boundary)

| Familie | Klasse / Stelle | Materialisierung | Quell-dtype |
|---------|------------------|------------------|-------------|
| **SmolLM2** | `smollm2/SmolLM2DenseTensor.decodeValues` | mmap `ByteBuffer` → **`float[elementCount]`** (Element-für-Element FLOAT/FP16/BF16-Decode) … | FP32/FP16/BF16 |
| **SmolLM2** | `smollm2/SmolLM2DenseTensor` ctor (`values.clone()`) | … danach **zweite** volle `float[]`-Kopie (defensiv) | — |
| **T5** | `t5/T5TensorData.readFloatValues` | mmap `ByteBuffer` → **`float[expectedElements]`** (FP32/FP16-Decode) | FP32/FP16 |
| **Qwen** | `qwen/QwenGpuKernels` (fused QKV / gate-up) → `MatMulNBitsKernel.dequantizeInt4` | INT4 → **`float[N*K]`** (+ Konkatenation für Fusion) | INT4 (AWQ block-128) |
| **alle WARP-Dense** | `warp/WarpDenseProjection.fromDequantizedWeights(float[])` → `MatMulNBitsKernel.fromDequantizedWeights` | nimmt **volle FP32-`float[output×input]`**, lädt sie via `D3D12Bindings.uploadFloats(float[])` in einen **Device-FP32-Buffer** | FP32 |

Decoder-only teilt sich diesen Pfad: `DecoderOnlyWarpDenseProjection.fromRowMajorWeights(float[])` /
`DecoderOnlyWarpFusedDenseProjection` und `SmolLM2WarpForwardPass` rufen alle `WarpDenseProjection.fromDequantizedWeights`.
T5 nutzt `T5WarpLinearProjection`/`T5WarpLmHead` über denselben `fromDequantizedWeights`-Eingang.

### Upload-Boundary (DirectML)
- `MatMulNBitsKernel`:
  - **FP32-Pfad** (`fromDequantizedWeights`, `useInt4Gpu=false`): Host-`float[]` → `uploadFloats(...)` → Device-FP32
    `weightBuf [N,K]`. (Doku im Kernel: „Dequantize INT4→FP32 once at preparation time, upload".)
  - **INT4-GPU-Pfad** (`new MatMulNBitsKernel(wb,N,K,qWeight,scales,zp,…)`): gepackte INT4 + FP32-Scales + ZP direkt
    auf Device (`int4WeightBuf`/`int4ScalesBuf`/`int4ZpBuf`) — **deutlich heap-leichter**, kein voller FP32-`float[]`.
- Die Upload-API ist heute `D3D12Bindings.uploadFloats(float[])` / `uploadBytes(byte[])` — sie **erzwingt ein
  Host-Array**; es gibt (noch) keine `upload(ByteBuffer)`/`upload(MemorySegment)`-Variante.

## 4. Größte Heap-Treiber (Reihenfolge)

1. **Volle FP32-`float[output×input]` pro Projektion** an `fromDequantizedWeights` — der dominante Treiber. Für jede
   Linear-/LM-Head-Projektion wird die komplette Gewichtsmatrix als FP32 auf dem Host gehalten, bis sie hochgeladen ist.
2. **SmolLM2 doppelte Kopie**: `decodeValues` (`new float[]`) **plus** ctor-`values.clone()` → 2× volle FP32-Matrix.
3. **Qwen-Fusion**: `dequantizeInt4` dreier Projektionen → `float[]` + Konkatenation (QKV, gate-up) vor dem Upload.
4. **FP16/BF16→FP32-Decode** (SmolLM2/T5): Element-für-Element-Schleife erzeugt eine FP32-Matrix, obwohl die Quelle
   halb so groß ist.

Der Reader/Katalog selbst trägt **nichts** dazu bei (zero-copy mmap).

## 5. Welche Familien betroffen sind

- **Qwen, SmolLM2, T5** liegen auf dem `.wdmlpack`-Pfad und sind betroffen.
- **MiniLM / Reranker / E5** laden **SafeTensors direkt** (`encoder/...`, `BertCpuEncoderWeights.load(modelDir,cfg)`)
  und berühren `.wdmlpack` **nicht** → für diesen Block **nicht relevant** (eigener, späterer Pfad falls je nötig).

## 6. Absichernde Tests (Ist-Ladepfad)

- `model/WdmlPackReaderTest`, `model/WdmlPackWriterTest`, `model/RuntimeModelPackageTest`,
  `model/SourceTensorCatalogTest`, `model/RuntimeLoadabilityTest` — Format/Lesepfad.
- `smollm2/SmolLM2RuntimePackageWeightsTest`, `smollm2/SmolLM2WdmlPackCompileToolTest`,
  `smollm2/SmolLM2NativeWarpExecutorTest`, `smollm2/SmolLM2PackageStalenessTest` — SmolLM2-Last/WARP.
- `qwen/QwenWdmlPackModelSourceTest`, `qwen/QwenWdmlPackCompilerTest`, `qwen/QwenWdmlPackCompileToolTest`,
  `qwen/QwenWdmlPackModelSourceTest` — Qwen-Paket/Source.
- T5: `t5/*` Compile-/Runtime-Package-Tests.

Diese Tests fixieren Manifest-Felder, Offsets und Token-/Numerik-Ergebnisse — ein heap-light-Umbau muss sie
unverändert grün lassen (Numerik byte-identisch).

## 7. Dateiformat-/Manifest-Annahmen (bleiben in H2 unverändert)

- Header trägt `payloadOffset`/`payloadLength`; pro Tensor im Manifest: `name`, `dataType` (ONNX-Code),
  `dims`, `byteLength`, `payloadOffset` (relativ zum Payload), `payloadLength`.
- Payload ist **roh little-endian** abgelegt (kein Container-internes Re-Encoding); deshalb ist der mmap-Slice bereits
  das exakte Upload-Byte-Bild für FP32-Tensoren.
- Reader-Limit: aktuell `fileSize > Integer.MAX_VALUE` nicht unterstützt (mmap als ein `MappedByteBuffer`).

**Folge:** Heap-light braucht **keine Formatänderung** — nur die Host-Konsumseite (Decode/Upload) muss den mmap-Slice
direkt verwenden statt ihn in `float[]` zu materialisieren.

## 8. Sichere Low-Risk-Slices (Vorschlag, additiv)

1. **Upload-Seam additiv erweitern (shared, gerätenah):** `D3D12Bindings.uploadFloats(ByteBuffer)` /
   `upload(MemorySegment)` **zusätzlich** zur `float[]`-Variante; `MatMulNBitsKernel.fromDequantizedWeights(ByteBuffer)`
   + `WarpDenseProjection.fromDequantizedWeights(ByteBuffer)` als **Overloads**. Bestehende `float[]`-Pfade bleiben.
2. **FP32-Pilotfamilie (T5 oder SmolLM2):** für den **FP32**-Fall den mmap-`ByteBuffer`-Slice direkt hochladen statt
   `float[]`-Decode + `clone()`. FP16/BF16 und Qwen-INT4 bleiben zunächst auf dem `float[]`-Pfad.
3. **SmolLM2 Doppel-`clone()` entschärfen** (nur für den Upload-Pfad), ohne den Reference/CPU-`float[]`-Pfad zu entfernen.

## 9. Risiken

- **mmap-Lebensdauer:** Heute wird sofort in `float[]` dekodiert, die Mapping-Lebensdauer ist irrelevant. Beim
  Direkt-Upload aus dem mmap-Slice muss das Mapping **bis zum Abschluss des GPU-Uploads** leben (FileChannel/Mapping
  nicht vorzeitig freigeben). Klare Lifetime-Verantwortung nötig.
- **Endianness:** mmap-Slices sind `LITTLE_ENDIAN`; `uploadFloats` schreibt heute Host-nativ. Eine `ByteBuffer`-Upload-
  Variante muss LE garantieren (x86 = LE, dennoch explizit prüfen) → Numerik byte-identisch.
- **Numerik-Gleichheit:** FP32-Direkt-Upload muss dasselbe Device-Bild erzeugen wie `float[]`→`uploadFloats`
  (identische LE-FP32-Bytes — erfüllt). FP16/BF16-Decode darf bei späterer Migration nicht abweichen.
- **Qwen-INT4-Fusion:** QKV/gate-up brauchen FP32-Konkatenation; nicht trivial auf `ByteBuffer` umstellbar → später.
- **Reference-/CPU-Pfade** (`SmolLM2DenseTensor.multiplyVector/dotRow`, T5-Reference) brauchen weiter `float[]` →
  `float[]`-Pfad **nicht entfernen** (ist auch Nicht-Ziel/verboten).

## 10. Nicht-Ziele (H-Block insgesamt, bestätigt für H1)

- Keine `.wdmlpack`-Formatänderung. Kein Entfernen bestehender `float[]`-Pfade. Kein mmap-Umbau in H1.
- Keine Kernel-/Runtime-Verhaltensänderung, keine Performance-Optimierung, keine Qwen-Legacy-Entfernung.

## 11. Empfehlung für H2

**Start an der gemeinsamen Schicht, nicht in einer Familie:** zuerst den **additiven Upload-Seam**
(`D3D12Bindings.upload(ByteBuffer)` + `MatMulNBitsKernel.fromDequantizedWeights(ByteBuffer)` +
`WarpDenseProjection.fromDequantizedWeights(ByteBuffer)` als Overloads, `float[]`-Pfade bleiben). Danach **T5 als
FP32-Pilot** (kleinste, am klarsten gekapselte Familie über `T5WarpLinearProjection`/`T5WarpLmHead`) auf den
`ByteBuffer`-Upload umstellen — nur FP32, mit Lifetime-Garantie und Byte-Identitäts-Test gegen den alten Pfad.

**Keine Formatänderung nötig.** **Reihenfolge der Heap-Ersparnis:** FP32-Familien (T5/SmolLM2) zuerst (direkter
ByteBuffer-Upload eliminiert 1–2 volle FP32-Kopien), Qwen-INT4-Fusion zuletzt (komplexer, eigener Slice).

## 12. Slice H2a — ByteBuffer-Upload-Naht (additiv, nur die Naht)

**Umgesetzt (rein additiv, keine Familienmigration, kein Format-/Runtime-Change):** Ein heap-leichter Upload-Pfad
existiert jetzt von einem `ByteBuffer` (inkl. read-only direkter mmap-Slices) bis in den Device-FP32-Buffer, ohne
vorher ein volles `float[]` zu materialisieren. Die bestehenden `float[]`-Pfade sind unverändert nutzbar.

| Schicht | Neue API (additiv) | Verhalten |
|---------|--------------------|-----------|
| `windows/D3D12Bindings` | `uploadBytes(device, queue, dst, ByteBuffer data, arena)` | mappt den Upload-Heap, kopiert `[position,limit)` **verbatim** via `MemorySegment.ofBuffer` → `copy` → `CopyBufferRegion`; position/limit unverändert; LE erzwungen |
| `windows/MatMulNBitsKernel` | `fromDequantizedWeights(wb, N, K, ByteBuffer fp32WeightsLe)` | FP32-ByteBuffer-Ctor; `prepareGpu` teilt sich jetzt einen `WeightUploader`-Body (float[]- **und** ByteBuffer-Upload identisch) |
| `inference/warp/WarpDenseProjection` | `fromDequantizedWeights(wb, name, out, in, ByteBuffer fp32WeightsLe)` | Shape+LE gerätefrei validiert, delegiert an den Kernel-ByteBuffer-Ctor |

**Upload-Mechanik:** direkter Byte-Copy aus dem `ByteBuffer` (keine temporäre `byte[]`/`float[]`-Host-Kopie). Der
Kernel `duplicate()`t den übergebenen Buffer, damit der Upload unabhängig von Position/Limit/Mark des Aufrufers ist
(kein Datenkopieren — nur ein leichter Buffer-View).

**Read-only mmap-Slices:** unterstützt — `MemorySegment.ofBuffer` akzeptiert read-only direkte Buffer; der Test nutzt
bewusst einen `allocateDirect(...).asReadOnlyBuffer()` als mmap-Stellvertreter.

**Endianness:** explizit behandelt — `uploadBytes(ByteBuffer)` und beide ByteBuffer-Overloads **verlangen
`LITTLE_ENDIAN`** (sonst `IllegalArgumentException`); Bytes werden verbatim kopiert, daher auf x86 byte-identisch zum
`float[]`-Pfad.

**Verifikation:**
- gerätefrei: `WarpDenseProjectionTest.byteBufferOverloadRejectsInvalidShapeAndEndianness…` (falsche Größe, BE,
  degenerierte Shape → IAE).
- WARP: `WarpDenseProjectionTest.warpByteBufferUploadMatchesFloatArrayUpload` — `float[]`-Upload vs ByteBuffer-Upload
  erzeugen **byte-identische** Projektionsergebnisse (Toleranz `0.0f`); Quell-Buffer-Position/Limit bleiben unberührt.
- Regression: windows-bindings + inference (qwen/decoderonly/smollm2/t5/generation/model/warp/phi3 + SmolLM2-Native) +
  encoder-Suiten grün; der geteilte `prepareGpu`-Body hält den bestehenden FP32-`float[]`-Pfad numerisch identisch.

**`WarpDenseProjection` ist damit bereit für den T5-FP32-Piloten (H2b):** der mmap-`ByteBuffer`-Slice eines FP32-T5-
Gewichts kann direkt an `fromDequantizedWeights(..., ByteBuffer)` gereicht werden. **Sicherste nächste Familie:** T5
(kleinste, klar gekapselt über `T5WarpLinearProjection`/`T5WarpLmHead`, FP32-Quelle) — mit mmap-Lifetime-Garantie und
Byte-Identitäts-Test gegen den alten Pfad. SmolLM2 danach (zusätzlich Doppel-`clone()` entschärfen); Qwen-INT4 zuletzt.

**Noch NICHT in H2a:** keine Familie nutzt die ByteBuffer-Naht produktiv — T5/SmolLM2/Qwen laden weiter über `float[]`.

## 13. Slice H2b — T5 FP32 ByteBuffer-Pilot

**Umgesetzt:** T5 ist die erste Familie, die FP32-Gewichte über die H2a-ByteBuffer-Naht hochlädt. Keine
Pipeline-/Generierungs-/Encoder-/Decoder-/Cross-Attention-Änderung, kein Format-Change, `float[]`-Fallback erhalten.

**Geänderte T5-Klassen (WARP-Upload nutzt ByteBuffer):**
- `t5/T5TensorData` — FP32-Decode ist jetzt **lazy**: aus einem FLOAT32-`RuntimeTensor` wird der rohe LE-mmap-Slice
  behalten (`fp32LittleEndianSource()`); das `float[]` entsteht erst bei `values()`/`at()`. **Validierung bleibt
  eager** (payload vorhanden, dtype, Länge → gleiche Exceptions wie bisher). FLOAT16- und `reference(...)`-Tensoren
  behalten eager `float[]` und liefern **kein** FP32-Source (→ float[]-Fallback).
- `t5/T5WarpLinearProjection.from` — nutzt bei vorhandenem `fp32LittleEndianSource()` den
  `WarpDenseProjection.fromDequantizedWeights(…, ByteBuffer)`-Pfad, sonst `weight.values()` (unverändert).
- `t5/T5WarpLmHead.from` — gleicher FP32-ByteBuffer-Pfad für die **größte** Matrix (vocab×hidden), FP16-Fallback.

**Welche T5-WARP-Gewichte jetzt ByteBuffer-Upload nutzen:** alle **nicht-fusionierten** FP32-Linears (Attention-Output
„o", Feed-Forward „wi/wo") **und der LM-Head**. **Fusionierte** Projektionen (Self-Attn Q/K/V, Cross-Attn K/V) gehen
weiter über `float[]`, weil die Konkatenation (`T5Fused…Projection.fuseWeights` → `T5TensorData.reference`) ein
zusammengesetztes Host-Array braucht — wie Qwens INT4-Fusion. LayerNorm/Bias/Embedding nutzen weiter `float[]`
(kleine Tensoren, Reference-Math).

**`float[]`-Fallback erhalten:** der alte Pfad bleibt voll funktionsfähig und wird genutzt, sobald kein FP32-Source
vorliegt (FP16, fusionierte/Reference-Tensoren) oder im Reference-Runtime (dort liefert die Factory weiter
`float[]`-basierte Projektionen).

**Wird `T5TensorData.readFloatValues` im WARP-Pfad noch gebraucht?** Für nicht-fusionierte FP32-Linears + LM-Head:
**nein** — durch den lazy Decode wird ihr `float[]` nie materialisiert. Für fusionierte Projektionen und für
LayerNorm/Bias/Embedding: ja (dort wird `values()`/`at()` aufgerufen und der lazy Decode greift).

**mmap-/Buffer-Lifetime abgesichert:** `T5TensorData` hält den `fp32SourceLe`-View (und damit das mmap-Mapping über die
geteilte Backing) am Leben; jeder `fp32LittleEndianSource()`-Aufruf liefert einen **unabhängigen** `duplicate()`-View
(Position/Limit isoliert). Der Upload ist **synchron** im Kernel-Konstruktor (`uploadBytes` → `executeAndWait`),
solange die `T5TensorData`/der Tensor-Katalog während des Ladens erreichbar ist — danach besitzt der Device-Buffer die
Daten. Kein vorzeitiges Freigeben des Mappings.

**Heap-Kopien messbar reduziert oder nur strukturell?** **Messbar reduziert** für die abgedeckten Gewichte: deren volle
FP32-`float[]`-Host-Kopie entfällt vollständig (vorher: 1× Decode-`float[]` + 1× `values()`-`Arrays.copyOf` vor dem
Upload → jetzt 0). Der LM-Head (größte Einzelmatrix) und alle nicht-fusionierten Linears profitieren. Fusionierte
Projektionen + Reference-Tensoren bleiben unverändert.

**Verifikation:**
- gerätefrei: `T5TensorDataTest` (FP32 → LE-Source + lazy Decode == Originalwerte; unabhängige Views; reference →
  kein Source, aber `values()` korrekt).
- WARP: `T5WarpLinearProjectionTest.fp32RuntimeTensorTakesByteBufferPathAndMatchesFloatArrayPath` — ByteBuffer-Pfad
  **byte-identisch** zum `float[]`-Pfad (Toleranz `0.0f`); rank-2-Validierung device-frei erhalten.
- Regression: t5/qwen/decoderonly/smollm2/generation/model/warp + encoder + SmolLM2-Native grün.

**Nächste Familie:** **SmolLM2** ist sinnvoll als Nächstes (FP32-Quelle, dieselbe `DecoderOnlyWarpDenseProjection`-
Naht; zusätzlich `SmolLM2DenseTensor`'s Doppel-`clone()` entschärfbar). **Qwen** zuletzt (INT4-Fusion + gepackter
INT4-GPU-Pfad sind komplexer und brauchen einen eigenen Slice). FP16/BF16-Tensoren bleiben in allen Familien zunächst
auf dem `float[]`-Decode.

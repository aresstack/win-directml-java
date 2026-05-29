# Qwen-Performance — Folge-Optimierungen nach HYBRID-Rollout

> Stand: 2026-05-29
> Betrifft: `directml-inference`, `directml-windows-bindings`
> Voraussetzung: HYBRID-Backend (= "max GPU offload" inkl. lm_head) ist eingebaut
> und liefert real gemessen 3.5 sec/Token auf Intel-iGPU (vorher 151 sec).

## 1 Ausgangslage (gemessen)

Profile-Snapshot eines 16-Token-Decode-Runs auf Intel UHD-iGPU, HYBRID-Modus:

| Stage        | Zeit/Token   | Anteil | Wo                                                       |
|--------------|--------------|--------|----------------------------------------------------------|
| Projections  | 2 041 ms     | 58 %   | GPU (48 Submissions: 24 Layer × `qkvFused` + `batchMlp`) |
| LM head      | 633 ms       | 18 %   | GPU (1 Submission, vocab=151 936)                        |
| Attention    | 907 ms       | 26 %   | **CPU** (GQA mit Java Vector API)                        |
| Norms+RoPE   | 0.6 ms       | 0 %    | CPU                                                      |
| SwiGLU       | 1.1 ms       | 0 %    | CPU                                                      |
| **Σ Decode** | **3 500 ms** | 100 %  |                                                          |

Prefill (Prompt seqLen ≈ 150 Tokens):

| Layer         | Kumulativ | Per-Layer                   |
|---------------|-----------|-----------------------------|
| 1             | 10 s      | 10 s (Warmup)               |
| 2..24         | 129 s     | ~5.2 s/Layer (steady state) |
| **Σ Prefill** | **129 s** |                             |

Per Layer Prefill: **600 GPU-Submissions** (seqLen × 4 Projektionen,
sequenziell per Token-Position). Bei ~8 ms iGPU-Fence-Wait = ~4.8 s/Layer.

## 2 Diagnose

Die fundamentale Engstelle auf Intel-iGPU ist **Per-Submission-Fence-Wait**
(typisch 30–50 ms im Decode-Pfad, 8 ms in der Prefill-Schleife wegen
Pipelining mehrerer Dispatches in einer Submission). Pro Submission läuft
nur ein winziger Tensor-Op über ~896 oder 4864 Elemente — die GPU ist
zu 99 % idle, der CPU-Treiber serialisiert. Jede Submission, die wir
einsparen, ist direkt 30–50 ms weniger Decode-Latenz.

Submission-Budget je Token (aktuell):

```
Decode  : 24 × 2 + 1 (lmHead) = 49 Submissions/Token
Prefill : 24 × 4 × seqLen     = 4 × seqLen Submissions/Layer × 24 Layer
                              = 96 × seqLen Submissions Total
```

Das ist die einzige Größe, die wir noch reduzieren können. Alle weiteren
Optimierungen drehen sich darum, **Submissions zu eliminieren**, nicht
darum, einzelne Kernels zu beschleunigen.

## 3 Geplante Optimierungen

### Opt-A: Batched GEMM im Prefill (Submission-Count seqLen → 1 pro Projektion)

**Aufwand**: ~1 Tag · **Risiko**: niedrig · **Wirkung erwartet**: Prefill 129 s → ~10–15 s
· **Wirkung gemessen**: Prefill 129 s → **77 s** (Faktor 1.55, nicht 8–13 wie erwartet)
· **Status: ✅ implementiert 2026-05-29** (siehe `MatMulNBitsKernel.matmulBatch`,
`QwenGpuPipeline.qkvFusedBatch/oProjBatch/gateUpFusedBatch/downProjBatch`,
`Qwen2Runtime.processLayerPrefill`)

#### 2026-05-29 — Warum nur 1.55× und nicht 8–15×?

Die Annahme "Submission-Count fällt auf 1/seqLen → wall-clock fällt proportional"
war falsch. Auf Intel-iGPU bestimmt **Compute-Zeit**, nicht Submission-Zahl,
die batched-GEMM-Latenz, sobald M ≥ 64. Konkret:

| Pfad               | Submissions/Layer | GPU-Compute/Layer          | Wall/Layer |
|--------------------|-------------------|----------------------------|------------|
| Per-Token (vorher) | 600 (150 × 4)     | 150 × 4 × ~3 ms ≈ 1 800 ms | ~5 200 ms  |
| Batched (jetzt)    | 4                 | 4 × ~700 ms ≈ 2 800 ms     | ~3 200 ms  |
| Differenz          | −596              | +1 000 ms                  | −2 000 ms  |

Eingespart wurden ~2 s/Layer Fence-Overhead — aber die naive HLSL-Schleife
(`FP32_MATMUL_BATCH_HLSL` mit unrolled-by-4) skaliert deutlich schlechter als
DML's optimierter `MatMulNBits` Operator. Konkrete Ursachen:

1. **Kein Tiling, kein groupshared**: jeder Thread lädt die ganze K-Spalte
   von X aus dem HBM für sich. Bei M=150, N=4864, K=896 sind das
   150 × 4864 × 896 × 4 Byte = ~2.6 GB Reads gegen ~5 MB unique Daten —
   ~500× over-fetch.
2. **FP32-Weights statt INT4**: `qkvFused` und `gateUpFused` laufen nach
   `fromDequantizedWeights` als FP32-Buffer (~17 MB für gateUp) statt 4 MB INT4.
   Memory-Bandbreite saturiert die iGPU.
3. **DML's MatMulNBits ist ein hand-tuned Kernel**: Intel-DirectML-Backend
   nutzt vermutlich SIMD16-Wave-Ops + cooperative loads, die unser naiver
   Shader nicht hat.

#### Follow-up Opt-A-2: HLSL-Tiling + groupshared für FP32-Batch-Shader

**Korrektur 2026-05-29**: Ursprungs-Plan war "DML batched MatMulNBits".
Das existiert nicht: im Code (`MatMulNBitsKernel.java`) ist der Name irreführend
— es gibt nur einen `DML_OPERATOR_GEMM` mit hartem `A=[1,1,1,K]` (M=1) für
Decode-Matvec. Der Batch-Pfad (`matmulBatch`) läuft über zwei eigene
HLSL-Compute-Shader (`INT4_MATMUL_BATCH_HLSL` und `FP32_MATMUL_BATCH_HLSL`),
beide naiv mit 1 Thread pro Output-Zelle ohne Tiling.

Der realistische, code-minimale 0.5-Tag-Win ist daher: **die existierenden
Batch-Shader durch eine getilte Variante ersetzen**.

**Problem im aktuellen `FP32_MATMUL_BATCH_HLSL`**:

- 1 Thread = 1 Output-Zelle (m, n)
- Jeder Thread liest K Elemente von X und K Elemente von W aus HBM
- Keine Datenkooperation zwischen Threads, die dieselbe X-Zeile oder W-Spalte brauchen
- Für gateUp (M=150, N=9728, K=896): ~10 GB HBM-Reads bei nur ~5 MB unique Daten
  → ~2 000× over-fetch, Memory-Bandwidth saturiert

**Lösung: `FP32_MATMUL_BATCH_TILED_HLSL`** mit Thread-Group-Tiling 16×16
und TILE_K=16 in groupshared:

- 1 Thread-Group berechnet einen [16, 16] Output-Tile (256 Threads)
- Pro K-Iteration: 256 Threads laden kooperativ ein [16, 16] X-Tile und ein
  [16, 16] W-Tile nach groupshared (1 Load/Thread/Iteration), dann jeder
  Thread macht 16 Multiply-Adds aus dem groupshared
- HBM-Reads pro Output-Tile: 16×K X + K×16 W = 32·K = 28 672 Bytes
  statt 256·1792 = 458 752 Bytes → **~16× weniger HBM-Traffic**
- Dispatch: 1D gehalten (über `SV_GroupID.x`, intern decoded zu (gx, gy))
  — so bleibt `GpuComputeKernel.recordDispatch` unverändert

**Aufwand**: ~0.5 Tag

- Neuer HLSL-Konstanten-String `FP32_MATMUL_BATCH_TILED_HLSL` (~40 Zeilen)
- Im `MatMulNBitsKernel.ensureBatchCapacity` FP32-Branch:
    - Shader-Name auf neue Variante umstellen
    - `groupSize` von 64 auf 256
    - `numRootConsts` von 3 auf 4 (zusätzlich `nTiles = (N+15)/16` als Konstante,
      damit der Shader Group-ID intern in (gx, gy) zerlegen kann)
- In `matmulBatch` FP32-Branch:
    - Constants-Array: `{N, K, M, (N+15)/16}` statt `{N, K, M}`
    - Dispatch-`elementCount`: `((M+15)/16) * ((N+15)/16) * 256` statt `M*N`

**Risiko**: niedrig

- Korrektheits-Test: bestehender Smoke-Test (Qwen2.5-Coder-0.5B-Instruct mit
  festem Prompt) muss byte-identische Token-Sequenz liefern wie der
  pre-Opt-A-2-Run. Wenn Shader-Bug, divergiert Output sofort sichtbar.
- INT4-Shader (oProj, downProj) bleibt zunächst unverändert, weil INT4-Tiling
  durch die per-Block-128-Scale-Layout-Komplikation aufwendiger ist und in
  Opt-A-3 separat behandelt wird.
- M=1 wird vom Tiled-Shader korrekt verarbeitet (TILE-Padding via Bounds-Check
  `if (m < M && n < N)` vor dem `Y.Store`).

**Wirkung erwartet**: Prefill 77 s → **~15–25 s** (Faktor 3–5).

- gateUp ist dominanter Kostenträger (N=9728 sehr groß) und am stärksten
  bandwidth-bound → profitiert am meisten
- qkvFused profitiert moderat (N=1152)
- oProj/downProj bleiben unverändert (INT4-Pfad), tragen aber weniger zur
  Gesamtzeit bei

**Akzeptanz**:

1. Profile-Log nach Opt-A-2 zeigt Prefill < 30 s (vorher 77 s).
2. Token-Sequenz für Smoke-Prompt byte-identisch zu Pre-Opt-A-2-Run.
3. Build + Smoke-Test grün.

#### Messung Opt-A-2 v1 (2026-05-29, Intel UHD Notebook, seqLen=49)

Gemessen: Prefill 58.5 s für seqLen=49. Nur 7 % besser als der Pre-Tile-Run,
statt erwarteter 3–5×. **Root cause via Per-Stage-Log gefunden**
(`Qwen2Runtime.profPrefillQkvBatchNs` etc., committed im selben Run):

```
Prefill per-stage (24 layers, seqLen=49):
  qkvBatch  (FP32) =  3 899 ms     7 %    (von Opt-A-2 angefasst)
  oProjBatch(INT4) =  3 708 ms     7 %    (unverändert)
  gateUpBatch(FP32)= 20 447 ms    35 %    (von Opt-A-2 angefasst, größte Stage)
  downBatch (INT4) = 11 347 ms    19 %    (unverändert)
  attn      (CPU)  = 18 271 ms    31 %    (CPU GQA, wächst quadratisch mit seqLen)
  sumStages        = 57 674 ms   ≈ Prefill total
```

Im selben Log: **96 separate `Compiled compute shader 'fp32_matmul_batch_tiled'`-
Zeilen** (24 Layer × 4 Projektions-Kernel = 96 `MatMulNBitsKernel`-Instanzen,
jede instanz-eigene `fp32BatchShader`-Field, jede ruft `ensureBatchCapacity`
und kompiliert auf eigene Faust). Bei ~150–900 ms pro Compile = **~30–40 s
purer DXC-Overhead im Prefill**. Mein Tile-Shader IST schneller als der alte
(compute-wise), aber der Compile-Overhead war im alten Pfad ebenfalls da und
wird durch den größeren Tile-Shader nur sichtbarer.

#### Opt-A-2 v2 (Shader-Sharing, 2026-05-29) — implementiert

**Aufwand**: ~30 min · **Risiko**: niedrig · **Status: ✅ implementiert 2026-05-29**

Fix: `int4BatchShader` und `fp32BatchShader` als `private static volatile
GpuComputeKernel sharedInt4BatchShader / sharedFp32BatchShader` mit
Double-Checked-Locking deklarieren. Erster `matmulBatch`-Aufruf je Modus
kompiliert den Shader, alle 95 weiteren Kernel-Instanzen benutzen dieselbe
GpuComputeKernel-Instanz wieder.

PSO + RootSignature sind D3D12-Device-weit gültig. Die gecachten
MethodHandles in `GpuComputeKernel` arbeiten via Vtable-Slot — funktionieren
für jede `ID3D12GraphicsCommandList` desselben Typs, also kann eine
Kernel-Instanz die GpuComputeKernel der ersten benutzen.

Die static Shader werden **intentional NICHT in `close()` freigegeben** —
sie leben für JVM-Lifetime. Speicherkosten: ~10 KB pro PSO+RootSig
(vernachlässigbar). Vorteil: zweites Modell-Load profitiert direkt vom
bereits kompilierten Shader.

**Wirkung erwartet**: Prefill 58 s (seqLen=49) → **~25–30 s** — sparen die
~30 s Compile-Overhead direkt ein. Compute-Zeit unverändert.

**Akzeptanz**:

1. Log zeigt nur **2** "Compiled compute shader"-Zeilen statt 96
   (einmal `int4_matmul_batch`, einmal `fp32_matmul_batch_tiled`).
2. Prefill < 35 s für seqLen=49.
3. Token-Sequenz byte-identisch.

#### Was Opt-A-2 NICHT lösen kann

Die Per-Stage-Messung zeigt: selbst nach Eliminierung des Compile-Overheads
bleibt **gateUpBatch ~14 s** (24 Layer × ~580 ms compute) und **attn(CPU)
~18 s**. Bei seqLen=150 wäre CPU-Attention quadratisch teurer (≈ 168 s
allein!). Bedeutung: **Opt-B (GPU-Attention) wird ab Prompt-Länge seqLen ≥ 80
die einzige wirksame weitere Optimierung sein**, weil CPU-Attention sonst
alles dominiert.

Folgereihenfolge nach Opt-A-2 v2:

1. Opt-A-2 v2 testen → erwartet Prefill ~25–30 s @ seqLen=49
2. Wenn ja: B+C-Paket angehen. Opt-B-Schritt 6 (Prefill mit GPU-Attention)
   wird für längere Prompts entscheidend.
3. Wenn nein: weitere Diagnose mit per-stage-Zahlen.

#### Messung Opt-A-2 v2 (2026-05-29, Intel UHD, seqLen=49)

v2 ist **eindeutig aktiv** (1× `Compiled compute shader 'fp32_matmul_batch_tiled'`,
close() schließt Shader nicht mehr). Aber: Prefill ist nur minimal besser.

| Stage            | v1 (ms) | v2 (ms) | Δ                                      |
|------------------|---------|---------|----------------------------------------|
| qkvBatch FP32    | 3 899   | 3 234   | −665                                   |
| oProjBatch INT4  | 3 708   | 3 131   | −577                                   |
| gateUpBatch FP32 | 20 447  | 24 604  | +4 157 (Mess-Varianz, Logik identisch) |
| downBatch INT4   | 11 347  | 11 713  | +366                                   |
| attn CPU         | 18 271  | 20 991  | +2 720                                 |
| **sum**          | 57 674  | 63 676  | +6 002                                 |

Fazit: Compile-Overhead war **viel kleiner** als 30 s geschätzt — die 95
vermiedenen Compiles haben nur ~1.2 s gespart. Die `300-900 ms`-pro-Compile-
Schätzung war ~30× zu hoch. **Opt-A-2 ist ausgereizt.**

#### Opt-A-3: INT4-Fused-Kernels für QKV und gate/up (2026-05-29, **WURZEL**)

**Aufwand**: ~1 Tag · **Risiko**: niedrig-mittel · **Status: 🚧 in Arbeit**

Wurzel-Befund im Profil: `gateUpBatch` läuft **FP32** (24.6 s, 39% der
Prefill-Zeit), `downBatch` läuft **INT4** (11.7 s). Beide haben INT4-Gewichte
im Modell. Grund in `QwenGpuKernels.createFusedGateUp` (Z. 164-178):

```java
float[] gDeq = dequantOrExtract(lw.gateProj());   // INT4 → FP32 dequant
float[] uDeq = dequantOrExtract(lw.upProj());     // INT4 → FP32 dequant
float[] fused = new float[2 * intermediate * hidden];   // concat
return MatMulNBitsKernel.fromDequantizedWeights(wb, fusedN, hidden, fused);
// ↑ setzt useInt4Gpu = false → FP32 GEMM Pfad
```

Gleicher Bug in `createFusedQKV` (Z. 142-158). Das war architektonisch
motiviert (1 fused submission statt 2 oder 3), aber kostet Faktor 8 an
Speicher-Bandbreite (FP32 = 32 MB/Layer durch L3 statt 4 MB INT4).

**Bandbreitenrechnung** (gate+up, Qwen2.5-0.5B):

| Format         | Bytes/Layer | Prefill (24 Layer, seqLen=49)             |
|----------------|-------------|-------------------------------------------|
| FP32 (heute)   | 33 MB       | 24.6 s                                    |
| INT4 (Opt-A-3) | 4 MB        | **erwartet ~5 s** (8× weniger Bandbreite) |

Gleiche Logik für QKV: 1.1 MB FP32 → 0.14 MB INT4 (3.2 s → ~1 s).

**Fix:** Helper-Methode `fuseQuantizedRowwise(QuantizedWeight... parts)`
in `QwenGpuKernels` die `qWeight`, `scales`, `zeroPoints` zeilenweise
konkateniert (kein FP32-Dequant). Resultat: ein größeres `QuantizedWeight`
mit `N = N1 + N2 + ...`, dann an den primären `MatMulNBitsKernel`-
Konstruktor übergeben → INT4-GPU-Pfad.

**Layout-Validierung** (Qwen2.5-0.5B, blockSize=32, K=896 → blocksPerRow=28):

- qWeight: `[N × K/2]` bytes, row-major → trivialer zeilenweiser concat
- scales:  `[N × blocksPerRow]` floats, row-major → trivialer concat
- zeroPoints: `[(N × blocksPerRow + 1) / 2]` bytes, global packed nibbles.
  Aber: 896 × 28 = 25 088 (gerade), 128 × 28 = 3 584 (gerade),
  4 864 × 28 = 136 192 (gerade), 9 728 × 28 = 272 384 (gerade)
  → **alle Teilmatrizen byte-aligned** → byte-arrays direkt konkatenierbar.

Safety-Check im Helper: assert `(N_part × blocksPerRow) % 2 == 0` für jeden
Teil außer dem letzten, sonst RuntimeException mit Fallback-Hinweis auf
bit-shift-Pfad (für andere Modelle ggf. nachrüsten).

**Pre-flight Bedingungen** (alle parts müssen gelten):

1. Alle `parts[i].K() == parts[0].K()` (gleiche Input-Dim)
2. Alle `parts[i].blockSize() == parts[0].blockSize()` (gleicher Block)
3. `(parts[i].N() * blocksPerRow) % 2 == 0` für i < parts.length−1

**Wirkung erwartet**: Prefill 64 s → **~30–35 s** @ seqLen=49.
Decode unverändert (gleiche Anzahl Submissions, gleiche FLOPS).
Token-Sequenz byte-identisch (INT4-Pfad numerisch identisch zu FP32 Dequant,
weil die FP32-Variante intern denselben Dequant macht — nur im Vorfeld).

**Akzeptanz**:

1. Log zeigt `INT4 GPU mode [9728, 896]` für gateUp (statt heute Upload-Log
   `Uploaded weight [9728, 896] to GPU (33 MB)`).
2. `gateUpBatch` < 8 s in Per-Stage-Messung.
3. Token-Sequenz byte-identisch zu v2 (FP32-fused).

**Risiko**: Wenn Token-Sequenz abweicht → ein Concat-Index ist falsch.
Dann mit Single-Layer-Test (Layer 0 only) gegen FP32-Referenz vergleichen.

#### Follow-up Opt-A-3: HLSL-Tiling + groupshared für FP32-Pfad

Falls Opt-A-2 für FP32-Kernel (`qkvFused`, `gateUpFused`) nicht funktioniert
(diese sind nicht INT4-GPU-resident), zweite Option: HLSL umschreiben mit
Thread-Group-Tiling — pro Thread-Group `[16, 16]` Output-Tile, X-Tile in
groupshared lädt jeder Thread nur 1/256 der K-Spalte:

```hlsl
[numthreads(16, 16, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    groupshared float xTile[16][TILE_K];  // 16 rows × TILE_K cols of X
    groupshared float wTile[TILE_K][16];  // TILE_K rows × 16 cols of W
    float acc = 0;
    for (uint kk = 0; kk < K; kk += TILE_K) {
        // cooperative load: 256 threads load 16 × TILE_K X-tile in 1/16 calls each
        // cooperative load: 256 threads load TILE_K × 16 W-tile
        GroupMemoryBarrierWithGroupSync();
        for (uint k = 0; k < TILE_K; k++) acc += xTile[ltid.y][k] * wTile[k][ltid.x];
        GroupMemoryBarrierWithGroupSync();
    }
    Output[(gid.y*16 + ltid.y) * N + (gid.x*16 + ltid.x)] = acc;
}
```

**Aufwand**: ~1 Tag · **Wirkung erwartet**: zusätzlich ×2–3 (Memory-Bandwidth
reduziert sich um Faktor TILE_K, hier 32).

#### Follow-up Opt-A-4: FP32 → INT4 vereinheitlichen

`qkvFused` und `gateUpFused` sind nur deshalb FP32, weil sie aus mehreren
ONNX-Tensoren konkateniert werden (Q,K,V bzw. gate,up). Wenn wir die
`MatMulNBitsKernel.fromDequantizedWeights`-Konstruktion durch eine
INT4-Re-Quantisierung der konkatenierten Weights ersetzen, läuft der
schnellere INT4-Pfad mit ~4× weniger Memory-Traffic.

**Aufwand**: ~1 Tag · **Risiko**: mittel (Re-Quant-Fehler kann Output
beschädigen — Test gegen CPU-Referenz nötig) · **Wirkung erwartet**: ×2.

**Empfohlene Reihenfolge**: A-2 (DML-Batched) zuerst — geringster Aufwand,
höchste Wirkung, kein neuer Kernel-Code. Wenn Prefill danach immer noch
> 30 s, dann A-3 + A-4.

#### Problem

`Qwen2Runtime.processLayerPrefill` ruft `gpuPipeline.qkvFused(...)` einmal
**pro Sequenzposition** in einer `for (s = 0; s < seqLen; s++)`-Schleife.
Jede Submission ist ein eigenständiger Fence-Wait. Bei 150 Tokens × 4
Projektionen × 24 Layer = **14 400 Submissions** für ein einziges Prefill
gegen ~10–15 effektive große GEMM-Calls.

#### Lösung

DML-MatMulNBits unterstützt `M > 1` (Batch-Dimension) nativ. Wir erweitern:

1. `MatMulNBitsKernel` um einen `matmulBatch(float[] x, float[] out, int M)`
   Pfad, der den Eingangsbuffer auf `M × K`, den Ausgangsbuffer auf `M × N`
   dimensioniert und den Dispatch mit Thread-Group-Count `M` statt `1` startet.
   Implementierungsweg: Die existierende Kernel-Konstruktion legt
   Buffer-Größen statisch auf M=1 fest — wir parametrisieren `maxBatch`
   beim Bau der Kernels (default = 1 für Decode, höher für Prefill).
2. `QwenGpuPipeline.qkvFusedBatch(int layerIdx, float[] xBatch, float[] qkvBatch, int M)`
   für den batched Pfad.
3. `Qwen2Runtime.processLayerPrefill` ersetzt die per-Token-`for`-Schleifen
   durch je einen `qkvFusedBatch(...)` / `oProj` / `gateUpFusedBatch(...)` /
   `downProj`-Aufruf pro Layer.

**Submission-Count danach**: 4 pro Layer × 24 Layer = **96 Submissions Total**
(statt 14 400). Per-Layer-Prefill ~5 s → ~0.4 s. Ein zweiter
`batch_size`-Wert beim Kernel-Bau braucht es vermutlich nicht — die
M-Dimension wird im Dispatch-Constant ausgelesen.

Attention bleibt im Prefill weiterhin auf CPU (parallel über seqLen × numHeads
mit SIMD), das ist schon schnell genug (~50 ms für seqLen=150).

#### Risiken

- DML-Operator muss `M > 1` korrekt verarbeiten (in der Praxis ja, da es
  ein normales GEMM ist).
- Buffer-Größen müssen für `M_max` reserviert werden — auf iGPU mit
  geteilten VRAM ist das unkritisch (4864 × 256 × 4 = 5 MB pro
  GateUp-Output-Buffer).
- Bestehende `matvec`-Aufrufe im Decode-Pfad dürfen nicht regressieren —
  einfach durch `matmulBatch(x, out, 1)` ersetzen mit identischem
  Verhalten oder via separater Code-Pfad.

### Opt-B: Attention auf GPU mit GPU-residentem KV-Cache (Decode 3.5 s → ~2.5 s/Token)

**Aufwand**: 2–3 Tage · **Risiko**: mittel · **Status**: 🚧 in Arbeit 2026-05-29

**Wirkung**:

- Spart 907 ms CPU-Attention/Token (gemessen im aktuellen HYBRID-Profil)
- Spart 24 Readbacks (Q|K|V → CPU) und 24 Uploads (attnOut → GPU)
- Saldiert: ~700–900 ms Decode-Latenz → erwartet **2.5–2.8 s/Token**
- Submission-Count bleibt 49 (qkvFused-Submission absorbiert Attention)

#### Architektur (an existierende `batchMlp` angelehnt)

Der existierende `QwenGpuPipeline.batchMlp` (siehe oben) zeigt das Muster:
6 GPU-Ops in 1 Submission, Zwischenergebnisse GPU-resident in den
Kernel-eigenen Input/Output-Buffern, einzige Roundtrips sind
Upload(hiddenInput) am Anfang und Readback(hiddenOut) am Ende.

Opt-B macht dasselbe für Submission 1:

```
Submission 1 (NEU — qkvAttnFused):
  upload(normedHidden)
  → dispatch qkvFused GEMM   (Out: oProj.inputBuf[qSize+2kvSize] in qkv-Layout)
  → dispatch rope_qk         (in-place auf Q-Range + K-Range im qkv-Buffer, pos im CB)
  → dispatch kv_append       (kopiert K,V aus qkv-Buffer an Position pos im KVCache)
  → dispatch gqa_attention   (liest Q + KVCache[0..pos] → schreibt attnOut nach oProj.inputBuf)
  → readback NUR für Debug-Modus; sonst attnOut bleibt GPU-resident

Submission 2 (existing batchMlp, leicht modifiziert):
  // attnOut liegt bereits in oProj.inputBuf — KEIN upload mehr
  → dispatch o_proj          ... (Rest unverändert)
```

Submission 1 ersetzt: 1 upload + 1 dispatch + 1 readback (vorher) + 38 ms CPU-Attention.
Neue Submission 1 hat: 1 upload + 4 dispatches + 0 readbacks = ein einzelner Fence-Wait,
ca. 30–50 ms statt vorher ~70 ms (Submission + CPU-Attention).

#### Implementierungs-Schritte (granular)

**Schritt 1: GPU-residenter KV-Cache (`QwenGpuKvCache.java`, neue Klasse)**

Pro Layer ein `[maxSeqLen, kvHeads * headDim]`-Buffer für K und einer für V
im `D3D12_HEAP_TYPE_DEFAULT`. Größe für Qwen 0.5B: `4096 × 2 × 64 × 4 = 2 MB / Layer / K
= 96 MB für 24 Layer × (K+V)`. Cap auf 4096 Positionen (statt config
maxPos=32k) — dynamisches Realloc beim Überschreiten.

API:

```java
class QwenGpuKvCache {
    QwenGpuKvCache(WindowsBindings wb, int numLayers, int kvHeads, int headDim, int maxPos);
    MemorySegment kCache(int layer);  // GPU-Buffer
    MemorySegment vCache(int layer);
    void reset();                      // pos=0 (keine memory-clear nötig, indexed Reads)
    int currentPos();
    void advance();                    // pos++
}
```

**Schritt 2: HLSL-Shader (`QwenAttentionShaders.java`)**

3 neue Shader, alle nach dem `Phi3ComputeShaders.RMSNORM_HLSL`-Muster:

```hlsl
// rope_qk.hlsl  — in-place RoPE auf Q (qHeads) und K (kvHeads) im QKV-Buffer
RWByteAddressBuffer QKV : register(u0);
ByteAddressBuffer Cos : register(t0);   // [headDim/2]
ByteAddressBuffer Sin : register(t0);
cbuffer CB : register(b0) { uint qSize; uint kvSize; uint headDim; uint pos; };
// Thread = (head_idx, dim_in_half). Apply Givens rotation on (x[i], x[i+halfDim]).
```

```hlsl
// kv_append.hlsl  — K|V aus qkv-Buffer an Position pos im KVCache schreiben
ByteAddressBuffer QKV : register(t0);
RWByteAddressBuffer KCache : register(u0);
RWByteAddressBuffer VCache : register(u1);
cbuffer CB : register(b0) { uint qSize; uint kvSize; uint pos; };
// Thread = i in [0..kvSize). KCache[pos*kvSize + i] = QKV[qSize + i]
```

```hlsl
// gqa_attention.hlsl  — GQA causal attention, online softmax
ByteAddressBuffer Q : register(t0);         // [qHeads * headDim]  (= QKV[0..qSize])
ByteAddressBuffer KCache : register(t1);    // [maxPos, kvHeads, headDim]
ByteAddressBuffer VCache : register(t2);
RWByteAddressBuffer Out : register(u0);     // [qHeads * headDim]
cbuffer CB : register(b0) { uint qHeads; uint kvHeads; uint headDim; uint pos; float scale; };
// 1 Thread-Group pro Q-Head; jeder Thread des Group bearbeitet einen Range
// von KV-Positionen, online softmax mit groupshared max/sum-Reduktion.
[numthreads(64, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    uint qh = gid.x;
    uint kvh = qh / (qHeads / kvHeads);
    groupshared float gMax;
    groupshared float gSum;
    // Pass 1: max over scores  → gMax
    // Pass 2: sum exp(score - gMax)  → gSum
    // Pass 3: out = Σ (exp - gMax)/gSum * V
    // Numerisch stabil + GQA-Mapping korrekt.
}
```

**Schritt 3: `qkvAttnFused` in `QwenGpuPipeline`**

```java
public void qkvAttnFused(int layerIdx, float[] normedHidden, int pos) {
    pipeline.begin();
    var cl = pipeline.getCommandList();
    var qkvK = kernels.qkvFused(layerIdx);
    var oK = kernels.oProj(layerIdx);
    // 1. upload normedHidden → qkvK.inputBuf  +  dispatch qkv GEMM → qkvK.outputBuf
    qkvK.recordBatchFromCpu(pipeline, normedHidden);
    // 2. RoPE in-place on qkvK.outputBuf
    attnShaders.rope().recordDispatch(cl, {qkvBuf, cosBuf, sinBuf}, {qSize, kvSize, headDim, pos}, dispatchN);
    // 3. KV-append: copy K|V slices from qkvBuf into kvCache.kBuf, vBuf at row pos
    attnShaders.kvAppend().recordDispatch(cl, {qkvBuf, kvCache.k(layerIdx), kvCache.v(layerIdx)}, {qSize, kvSize, pos}, kvSize);
    // 4. GQA → oK.inputBuf (= attnOut)
    attnShaders.gqa().recordDispatch(cl, {qkvBuf, kvCache.k(layerIdx), kvCache.v(layerIdx), oK.inputBuf}, {qHeads, kvHeads, headDim, pos, scaleBits}, qHeads);
    // KEINE readback — attnOut bleibt in oK.inputBuf, batchMlp liest ihn dort
    pipeline.submitAndWait();
}
```

**Schritt 4: `batchMlp` ohne upload-Step**

Neue Variante `batchMlpGpuResident(hiddenInput, hiddenOut, layerIdx)`: überspringt
den `oK.recordBatchFromCpu(pipeline, attnOutput)`-Step (attnOut ist schon
GPU-resident), Rest identisch.

**Schritt 5: Decode-Pfad in `Qwen2Runtime.decodeSingleToken`**

Wenn `qwen.gpu.attention=true`:

```java
if (gpuAttention && useGpuForDecode) {
    gpuPipeline.qkvAttnFused(layerIdx, decNormed, pos);  // Submission 1 (GPU attn)
    gpuPipeline.batchMlpGpuResident(hiddenIo, hiddenIo, layerIdx);  // Submission 2 (existing)
    return;
}
```

**Schritt 6: Prefill-Anbindung**

Prefill muss den GPU-KV-Cache populieren, sonst startet Decode mit leerem
Cache. Zwei Varianten:

- (a) Prefill bleibt CPU-Attention, schreibt am Ende des Prefills den
  vollständigen K/V-Tensor per Upload in den GPU-KV-Cache (1 Upload pro
  Layer am Prefill-Ende).
- (b) Prefill nutzt ebenfalls `qkvAttnFused`-Batched, K/V landen direkt im
  GPU-Cache (benötigt M-dim im rope/kvAppend/gqa, aufwendiger).

Variante (a) ist niedrigeres Risiko und liefert die volle Decode-Wirkung.
In dieser Iteration zuerst (a), später (b) als Folge-Opt.

#### Risiken

- **Online-Softmax in HLSL muss bit-nah zur CPU-Referenz sein** — sonst
  Divergenz in Token-Sequenz nach 5–10 Tokens. Test-Pfad: `qwen.gpu.attention=true`
  vs `false` mit deterministischem Prompt, beide müssen identische Token-IDs
  liefern.
- **GQA-Mapping**: Q-Head `h` liest KVCache mit `kvh = h / (qHeads/kvHeads)`.
  In HLSL als Integer-Division.
- **KV-Cache-Reset zwischen Generationen**: `pos` wird beim `resetCache()` auf 0
  zurückgesetzt, Buffer-Inhalt darf alt bleiben (wird nicht gelesen wegen
  causal mask + `pos`-Bound).
- **RoPE-Tabelle auf GPU**: einmalig beim Pipeline-Start hochladen
  (halfDim × maxPos × 4 Byte = 256 KB für maxPos=4096, headDim=64).
- **Buffer-Größe**: 96 MB GPU-KV-Cache für 24 Layer × 4096 maxPos. iGPU
  teilt sich VRAM mit Host-RAM — unkritisch.
- **Backward-Kompat**: Feature-Flag `qwen.gpu.attention` default `false`,
  damit der validierte 3.5 s/Token-HYBRID-Pfad nicht regrediert.

### Opt-C: Single-Dispatch Decoder Layer (Decode 3.5 s → ~1 s/Token)

**Aufwand**: ~1 Woche · **Risiko**: hoch · **Wirkung**:
49 Submissions/Token → **25 Submissions/Token** (1 pro Layer + 1 lm_head)

#### Lösung

Ein einziger HLSL-Compute-Shader pro Layer, der die komplette Layer-Logik
in einer Submission ausführt:

```
input:  hiddenIn [hidden]
        per-layer weights (alle Kernels GPU-resident)
        KV-Cache (von Opt-B vorausgesetzt)

step 1: RMSNorm(hiddenIn) → normedH
step 2: qkv = quantMatMul(normedH, qkvFusedWeights)
step 3: Q, K, V split; RoPE(Q, K, pos)
step 4: KVAppend
step 5: attnOut = GQA(Q, K-cache, V-cache, pos)
step 6: oProj = quantMatMul(attnOut, oWeights)
step 7: residual = hiddenIn + oProj
step 8: normedR = RMSNorm(residual, postNormW)
step 9: gateUp = quantMatMul(normedR, gateUpWeights)
step 10: mlpAct = SwiGLU(gateUp)
step 11: down = quantMatMul(mlpAct, downWeights)
step 12: out = residual + down
output: out [hidden]
```

Alle Zwischenergebnisse in `groupshared` oder GPU-Buffern, keine
CPU-Roundtrips innerhalb des Layers.

**Submission-Count**: 1 pro Layer × 24 + 1 lm_head = **25 Submissions/Token**.
Bei ~30 ms iGPU-Fence = ~750 ms/Token reine Submission-Latenz, plus
echte Compute-Zeit ~200 ms → **~1 s/Token**.

#### Risiken

- INT4-Dequantisierung in HLSL muss bit-exakt zur CPU-Referenz sein
  (sonst Divergenz mit Phi3/Qwen-Tests).
- DML-MatMulNBits ist kein direkter HLSL-Kernel — wir müssten entweder
    - die DML-Op als Sub-Submission lassen (Pipeline-State-Switch teuer) ODER
    - eine eigene INT4-Matmul-HLSL schreiben (groß, aber machbar — alle
      Bausteine sind im Repo vorhanden: Dequant-Code in `MatMulNBitsKernel.dequantizeInt4`).
- Shader-Größe wird groß genug, dass GPU-Register-Allocation kritisch wird.
  Test-Pfad nötig: ein Layer-für-Layer Vergleich mit CPU-Referenz.

## 4 Implementierungs-Reihenfolge (revidiert 2026-05-29)

Die ursprüngliche Reihenfolge A → B → C hat sich nach realer Messung
geändert. Aktueller, vereinbarter Plan:

| # | Optimierung                        | Aufwand  | Status  | Wirkung erwartet                |
|---|------------------------------------|----------|---------|---------------------------------|
| 1 | Opt-A v1: Batched GEMM Prefill     | 1 Tag    | ✅ done  | Prefill 129 s → 77 s (gemessen) |
| 2 | Opt-A-2: HLSL Tiling + groupshared | 0.5 Tag  | ⏳ next  | Prefill 77 s → ~15–25 s         |
| 3 | Opt-B + Opt-C als ein Paket        | 3–4 Tage | 🚧 wait | Decode 3.5 s → ~1 s/Token       |

Nach Opt-A-2: 200 Tokens in ~20 s Prefill + 200 × 3.5 s Decode ≈ **12 min**.
Nach Opt-A-2 + B+C: 200 Tokens in ~20 s + 200 × 1 s ≈ **3.7 min**.

### Warum B+C als Paket statt einzeln

Wir wollten ursprünglich C als unabhängige Folge-Optimierung
("Single-Dispatch Decoder Layer, 1 Woche"). Bei genauer Analyse zeigt sich:
**C ist algorithmisch eine Obermenge von B**, nicht ein Nachfolge-Schritt.
Steps 3–5 des Single-Dispatch-Layers (RoPE auf Q/K, KV-Append, GQA-Attention)
sind exakt Opt-B. Ohne GPU-residenten KV-Cache und GQA-HLSL-Shader kann C
kein "Single Dispatch" sein, weil zwangsweise ein CPU-Attention-Roundtrip
in der Mitte des Layers nötig wäre — das wäre dann faktisch wieder eine
2-Submission-Lösung.

Die ursprüngliche "1-Woche"-Schätzung für C ging davon aus, dass C *ohne*
B-Infrastruktur gebaut wird — als monolithischer Shader inkl. eigener
INT4-Dequant-Reimplementierung in HLSL. Das ist groß und riskant.

Wenn wir B sauber bauen, ist C nur noch das **Mergen zweier
Command-List-Recording-Sections in eine einzige Submission** — eine
~50-Zeilen-Refaktorierung in `QwenGpuPipeline.batchMlp` / `qkvAttnFused`,
kein neuer HLSL-Code. Aufwand für C *nach* B: ~0.5 Tag.

### Konsolidierte B+C-Schritte (für den späteren Implementierungs-Block)

1. **B-Step 1**: GPU-KV-Cache-Buffer (`QwenGpuKvCache.java`,
   D3D12 DEFAULT-Heap, 24 × 2 × (4096 × kvHeads × headDim × 4 B) ≈ 96 MB).
   ✅ **implementiert 2026-05-29** (`directml-inference/.../QwenGpuKvCache.java`).
   Layout `[kvHead → pos → d]`, ein D3D12 DEFAULT-Buffer pro Layer pro K/V,
   `uploadFromCpu(layer, seqLen, cpuK, cpuV)` baut staging-Array mit Tail=0
   und uploadet via `D3D12Bindings.uploadFloats`. Cached GPU-VAs pro Layer.
2. **B-Step 2**: HLSL-Shader (`QwenAttentionShaders.java`): `rope_and_append`
   (RoPE auf Q + rotiertes K + V in KCache/VCache schreiben),
   `gqa_attention_decode` (online-softmax, 1 Threadgroup pro Q-Head,
   headDim Threads kooperieren über Positionen).
   ✅ **implementiert 2026-05-29** (`directml-windows-bindings/.../QwenAttentionShaders.java`).
   GPT-J-style RoPE (Pairing `(x[i], x[halfDim+i])`), `sincos()` on-the-fly
   (kein cos/sin-Table-Upload nötig). FlashAttention-Recurrence:
   `m_new = max(m, score); alpha = exp(m-m_new); s = s*alpha + exp(score-m_new);
   outVal = outVal*alpha + e*V`.
   ⚠️ **Isoliert testen gegen CPU-Referenz BEVOR End-to-End-Integration** —
   sonst sind Token-Sequenz-Divergenzen im Smoke-Test nicht auf B vs. CPU
   rückführbar (siehe Selbstdisziplin-Notiz).
3. **B-Step 3**: `qkvAttnFused(layerIdx, normedHidden, pos)` in
   `QwenGpuPipeline` als eigene Submission (attnOut bleibt GPU-resident in
   `oProj.inputBuf`).
4. **B-Step 4**: `batchMlpGpuResident(...)` Variante von `batchMlp`, die
   den `attnOutput`-Upload-Step (`oK.recordBatchFromCpu`) überspringt.
5. **B-Step 5**: Decode-Pfad-Wiring in `Qwen2Runtime.decodeSingleToken`
   hinter Feature-Flag `qwen.gpu.attention=true` (default `false`, damit
   der validierte HYBRID-Pfad nicht regrediert).
6. **B-Step 6**: Prefill-Anbindung: KV-Cache am Ende des Prefills per
   Upload aus dem CPU-Cache befüllen (1 Upload/Layer, vor erstem
   Decode-Token). Variante (a) aus Opt-B-Beschreibung.
7. **B-Akzeptanz**: Decode-Profile zeigt Attention < 100 ms/Token (vorher
   907 ms), Total ≤ 2.8 s/Token. Token-Sequenz für Smoke-Prompt **byte-
   identisch** zu HYBRID-Run.
8. **C-Step 1**: `qkvAttnFused` + `batchMlpGpuResident` in EINE
   Command-List schreiben, EIN `submitAndWait()` am Ende. Verifizieren,
   dass alle GPU-Buffer (qkv-Buffer, oProj-Buffer, residualBuf,
   downProj-Buffer) zwischen den Sections via UAV-Barriers korrekt
   synchronisiert sind.
9. **C-Akzeptanz**: Submission-Count fällt von 49 → 25 (über Fence-
   Counter-Logging messbar). Decode-Total ≤ 1.2 s/Token. Token-Sequenz
   weiterhin identisch zum CPU-Referenz-Run.

### Warum nicht direkt C ohne B (Diskussion 2026-05-29)

Getestet im Konzept-Review: ohne B-Schritte 1–4 würde der "Attention
im Shader"-Step zu CPU-Attention-Fallback degenerieren, und der Resubmit
zwischen Step 5 und Step 6 macht aus dem geplanten "1 Dispatch" effektiv
2 Dispatches. Das HLSL-Korrektheits-Risiko (Online-Softmax, GQA-Mapping)
ist in B und C identisch und muss in beiden Fällen einmal gelöst werden.
Ein "direkt C"-Pfad spart also keinen Aufwand ein — er verhindert nur
die isolierte Testbarkeit von B's GPU-Attention-Code.

### Selbstdisziplin-Notiz für künftige Sessions

Vor jedem neuen Implementierungs-Schritt **diese Tabelle und die
B+C-Schritt-Liste oben lesen**. Insbesondere:

- Opt-A-2 ist die *nächste* Aufgabe, nicht Opt-B. Reihenfolge ist absichtlich
  so gewählt, weil Opt-A-2 risiko-arm und sofort UX-wirksam ist.
- B-Step 2 (Shader-HLSL) muss vor B-Step 5 (Decode-Wiring) gegen
  CPU-Referenz validiert sein. Wer das überspringt, debuggt 3 Tage lang
  Token-Sequenz-Divergenzen, ohne zu wissen, ob der Shader oder das Wiring
  schuld ist.
- B-Step 6 (Prefill-KV-Cache-Upload) ist die kleinste, aber notwendigste
  Lücke: ohne sie startet Decode mit leerem KV-Cache und produziert nur
  Self-Attention auf den letzten Token statt auf den vollen Prompt-Kontext.
- C kommt **erst nach B-Akzeptanz**. Wer C vorzieht, wenn B noch nicht
  byte-identisch zur Referenz ist, hat keinen Validierungs-Anker mehr.

## 5 Akzeptanzkriterien

Jede Optimierung wird mit dem bestehenden Profile-Log validiert:

```
Decode progress: 16 tokens, last 16 took X.XX s (Y.YY tok/s) — profile so far:
[Qwen2 Decode Profile] 16 tokens, NNNN ms total, NN ms/token
  Mode:          ...
  Projections:   NN ms avg (XX%)
  Attention:     NN ms avg (XX%)
  ...
```

Für jede Optimierung:

1. Profile-Vergleich vorher/nachher (gleicher Prompt, gleiche Token-Anzahl)
2. Output-Vergleich (Token-IDs identisch zu Vor-Optimierungs-Run — wir
   ändern nur die Performance-Charakteristik, nicht die Numerik)
3. CPU-Referenz-Test (`QwenInferenceEngine` mit Backend `cpu` vs. Backend
   `hybrid`) muss identische Token-Sequenzen liefern (gegen ein etabliertes
   Smoke-Test-Modell wie Qwen2.5-Coder-0.5B-Instruct).

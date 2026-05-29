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

**Aufwand**: ~1 Tag · **Risiko**: niedrig · **Wirkung**: Prefill 129 s → ~10–15 s
· **Status: ✅ implementiert 2026-05-29** (siehe `MatMulNBitsKernel.matmulBatch`,
`QwenGpuPipeline.qkvFusedBatch/oProjBatch/gateUpFusedBatch/downProjBatch`,
`Qwen2Runtime.processLayerPrefill`)

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

**Aufwand**: 2–3 Tage · **Risiko**: mittel · **Wirkung**:

- Spart 900 ms CPU-Attention/Token
- Spart 24 Readbacks (Q|K|V → CPU) und 24 Uploads (attnOut → GPU)
- Saldiert: ~24 weniger Submissions (qkvFused darf gleich in die Attention
  weiterfließen, ohne zwischendurch readback)

#### Lösung

1. **GPU-residenter KV-Cache**: pro Layer ein `[maxSeqLen, kvHeads, headDim]`
   `D3D12_HEAP_TYPE_DEFAULT`-Buffer für K und V. Größe für Qwen 0.5B:
   `4096 × 2 × 64 × 4 = 2 MB / Layer / K = 96 MB für K+V × 24 Layer`.
   Passt locker in iGPU-VRAM.
2. **HLSL-Compute-Shader `gqa_attention`**:
    - Input: Q `[qHeads, headDim]` (GPU-Buffer von qkvFused),
      K-Cache `[seqLen, kvHeads, headDim]`,
      V-Cache `[seqLen, kvHeads, headDim]`
    - Output: attnOut `[qHeads, headDim]`
    - Pro Q-Head: Dot(Q, K[p]) für alle p ≤ pos, Softmax, Σ w·V[p].
    - Thread-Group-Layout: ein Group pro Q-Head, jeder Thread arbeitet
      auf einem Range von KV-Positionen.
3. **KV-Cache-Append-Kernel**: nach RoPE auf K/V die neuen Werte an Position
   `pos` ins Cache-Buffer schreiben (1 winziger Dispatch oder per
   `D3D12_COPY_REGION` zwischen GPU-Buffern).
4. **Pipeline-Integration**: in `QwenGpuPipeline` neue Methode
   `qkvAttnFused(layerIdx, normedHidden, attnOut)`, die qkvFused +
   RoPE-Apply + KV-Append + Attention in einer Command-List mit einem
   einzigen `submitAndWait()` aufzeichnet.

Per Layer Decode reduziert sich damit auf:

- 1 Submission: qkvFused + RoPE + KVAppend + Attention
- 1 Submission: batchMlp (unverändert)
- **= 2 Submissions/Layer × 24 Layer + 1 lm_head = 49 → 49 Submissions** ❌

Hmm — Submission-Count bleibt gleich, weil batchMlp schon vorher eigene
Submission war. Was wir sparen: die 900 ms CPU-Attention und die zwei
Readback/Upload-Cycles in jeder Submission. Net ~700–900 ms Decode-Latenz.

#### Risiken

- HLSL-Shader für GQA + dynamische Softmax muss numerisch korrekt sein
  (max-substraction für Stabilität, FP32-Akkumulator).
- RoPE-Application auf GPU-residenter Q/K — kann als separater Mini-Shader
  oder direkt im qkv-Output-Shader inline laufen.
- KV-Cache-Reset zwischen Generationen muss sauber funktionieren.
- Buffer-Größen wachsen mit `maxPositionEmbeddings` — bei Qwen 0.5B mit
  32k Context-Window wären das ~768 MB. Daher cap auf realistisches
  Maximum (z.B. 4096) und dynamisch reallozieren.

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

## 4 Implementierungs-Reihenfolge

| # | Optimierung                     | Aufwand  | Wirkung               |
|---|---------------------------------|----------|-----------------------|
| 1 | Opt-A: Batched-GEMM Prefill     | 1 Tag    | Prefill 129 s → ~10 s |
| 2 | Opt-B: GPU-Attention + KV-Cache | 2–3 Tage | Decode 3.5 s → ~2.5 s |
| 3 | Opt-C: Single-Dispatch Layer    | ~1 Woche | Decode 2.5 s → ~1 s   |

Reihenfolge ist intentional: Opt-A ist klein, low-risk, hoher ROI (Prefill
ist *jetzt* der dominante UX-Schmerzpunkt nach dem HYBRID-Fix). Opt-B
liefert dann den nächsten Decode-Schub, ohne die Architektur für Opt-C
zu blockieren. Opt-C ist die teuerste, aber auch die einzige, die das
Decode-Budget unter 1.5 s/Token drückt — sinnvoll nur, wenn der
Use-Case lange Generationen (> 200 Tokens) braucht.

Nach Opt-A + Opt-B: 200 Tokens in ~10 s + 200 × 2.5 s ≈ **8.5 min**.
Nach Opt-C: 200 Tokens in ~10 s + 200 × 1 s ≈ **3.5 min**.

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

# GPU Pipeline — Aktueller Stand (V2.0)

## Architektur

**129 GPU-Submissions pro Token → 65** via MLP-Batch.

Pro Layer: 1× QKV-GEMM (einzeln) + 1× MLP-Batch (7 GPU-Ops in 1 Submission).
Plus 1× LM-Head am Ende.

---

## Submission-Vergleich

| Version | Submissions/Token | Fence-Waits | CPU↔GPU Transfers |
|---------|-------------------|-------------|-------------------|
| V1.x | 129 | 129 | 258 (upload+readback) |
| **V2.0** | **65** | **65** | **~100** |

---

## V2.0 MLP-Batch

Pro Layer werden 7 GPU-Operationen in EINER Command-List gebatcht:

```
pipeline.begin()
  O-GEMM(attnOutput → oK.output)                    [DML]
  Upload(hiddenInput → residualBuf)                  [COPY]
  Add(oK.output + residualBuf → residualBuf)         [compute]
  RMSNorm(residualBuf → guK.input)                   [compute]
  GateUp-GEMM(guK.input → guK.output)                [DML]
  SwiGLU+Scale(guK.output → downK.input)             [compute]
  Down-GEMM(downK.input → downK.output)              [DML]
  Add(residualBuf + downK.output → residualBuf)      [compute]
  Readback(residualBuf → CPU)                        [COPY]
pipeline.submitAndWait()
```

### HLSL Compute Shader (V2.0)

| Shader | UAVs | Constants | Funktion |
|--------|------|-----------|----------|
| `element_add` | 3 (A, B, C) | count | Elementweise Addition |
| `rms_norm` | 3 (In, Weight, Out) | dim, eps | RMSNorm mit Group-Shared Reduction |
| `swiglu` | 3 (GateUp, Scale, Out) | intermediate | SwiGLU + Scale fused |
| `scale` | 3 (X, Scale, Out) | count | Elementweises Multiply |

---

## Decode-Pfad (pro Token)

```
Embedding-Lookup (CPU)

for layer 0..31:
  RMSNorm (CPU)
  QKV GEMM → pipeline.matvec (1 submission)
  RoPE (CPU)
  KV-Cache Store (CPU, per-head layout)
  Attention Score+Softmax+V-Sum (CPU, parallel heads)
  MLP-Batch (1 submission — 7 GPU ops)

Final RMSNorm (CPU)
LM-Head → pipeline.matvec (1 submission)
Argmax (CPU)
```

---

## Benchmark-Kommando

```
/benchmark
```

In der `Phi3ChatCLI` ausführen. Misst 3 Runs à 32 Tokens nach 2 Warmup-Runs.

---

## Performance-Optimierungen (implementiert)

- **V1.3**: Per-Head KV-Cache Layout (Stride 12 KB → 384 Bytes)
- **V1.4**: Parallele Attention-Heads (`IntStream.parallel()`, 32 Heads)
- **V1.4**: 4-Accumulator ILP Dot Product
- **V2.0**: MLP-Batch (7 GPU-Ops → 1 Submission, 129 → 65 Submissions/Token)

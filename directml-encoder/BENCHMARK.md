# `embedBatch` Benchmark — Baseline (V2.0)

Host: Windows 11, DirectML in-box, MiniLM only (E5 weights not on disk).
Command:

```
gradle :directml-encoder:runEmbedBatchBenchmark --args="model minilm both 1 10,50,100"
```

Sweep: `N ∈ {10, 50, 100}`, warmup = 1×`maxN` batch.
Corpus: 4 round-robined templates (short / short / long / long) → spans pad buckets `64` and `128`.

## Ergebnisse

| backend | model  |   N |   loopMs |  batchMs | loopPerMs | batchPerMs | speedup |
|---------|--------|----:|---------:|---------:|----------:|-----------:|--------:|
| cpu     | minilm |  10 |  2484.96 |  2520.79 |   248.496 |    252.079 |   0.99× |
| cpu     | minilm |  50 | 13990.69 | 13879.30 |   279.814 |    277.586 |   1.01× |
| cpu     | minilm | 100 | 28595.91 | 28343.04 |   285.959 |    283.430 |   1.01× |
| dml     | minilm |  10 |   443.18 |   438.63 |    44.318 |     43.863 |   1.01× |
| dml     | minilm |  50 |  2213.25 |  2239.53 |    44.265 |     44.791 |   0.99× |
| dml     | minilm | 100 |  4438.07 |  4723.57 |    44.381 |     47.236 |   0.94× |

## Beobachtungen

1. **CPU**: `embedBatch == loop`. Erwartet, weil `CpuBertEncoder` keinen
   batched Pfad überschreibt und der Default in `EmbeddingModel` schlicht
   `embed()` in einer Schleife aufruft. Per-text ≈ 280 ms.

2. **DirectML**: **kein Speedup** durch Batching. Per-text bleibt bei
   ~44 ms unabhängig von `N`. Der GPU-Mean-Pool + GPU-L2-Refactor ist
   funktional drin (Readback ist nur `[N,H]`), aber der dominierende
   Kostenanteil ist offensichtlich der Encoder-Stack selbst – und der
   skaliert in der aktuellen Implementierung linear mit `N`.

3. Das `[N,1,S,H]`-Form-Binding der GEMMs in `DirectMlBertEncoderStack`
   liefert die Korrektheit, aber DirectML scheint die Batch-Achse nicht
   in eine einzige GPU-Welle zu falten – jeder Layer wartet weiterhin
   per `executeAndWait` auf seine Fence.

## Daraus abgeleiteter nächster Schritt

Datenlage spricht **gegen** weitere Mean-Pool/L2-Optimierungen und
**für** den Encoder-Stack-Layer-Pipeline-Hebel:

- **Submission-Coalescing pro Layer**: alle DML-Sub-Ops eines
  Encoder-Layers in eine Command-List, eine Fence pro Layer statt
  pro Sub-Op (analog zum V2.0-MLP-Batch im Phi3-Pfad,
  `gpu-pipeline-baseline.md`).
- **Asynchrone Dispatches**: `executeAndWait` ersetzen durch
  Submit + Fence-Wait nur am Stack-Ausgang.
- Erst danach lohnt sich **Batch-Coalescing** (8/16/32/...), weil
  Batching erst greift, wenn die per-Layer-Fence-Kosten reduziert sind.

Worker-/Threading-Modell ist sekundär: solange der Single-Stream-Pfad
nicht ausgequetscht ist, würde Parallelität nur die Submission-Queue
serialisieren.


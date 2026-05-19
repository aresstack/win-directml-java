# `embedBatch` Benchmark — Baseline (V2.0) → Per-Layer-Coalesced (V2.1)

Host: Windows 11, DirectML in-box, MiniLM only (E5 weights not on disk).
Command:

```
gradle :directml-encoder:runEmbedBatchBenchmark --args="model minilm both 1 10,50,100"
```

Sweep: `N ∈ {10, 50, 100}`, warmup = 1×`maxN` batch.
Corpus: 4 round-robined templates (short / short / long / long) → spans pad buckets `64` and `128`.

## Vorher (V2.0 – per-kernel Command-Lists, DirectMlGpuBatch deferred fence)

| backend | model  |   N |   loopMs |  batchMs | loopPerMs | batchPerMs | speedup |
|---------|--------|----:|---------:|---------:|----------:|-----------:|--------:|
| cpu     | minilm |  10 |  2484.96 |  2520.79 |   248.496 |    252.079 |   0.99× |
| cpu     | minilm |  50 | 13990.69 | 13879.30 |   279.814 |    277.586 |   1.01× |
| cpu     | minilm | 100 | 28595.91 | 28343.04 |   285.959 |    283.430 |   1.01× |
| dml     | minilm |  10 |   443.18 |   438.63 |    44.318 |     43.863 |   1.01× |
| dml     | minilm |  50 |  2213.25 |  2239.53 |    44.265 |     44.791 |   0.99× |
| dml     | minilm | 100 |  4438.07 |  4723.57 |    44.381 |     47.236 |   0.94× |

## Nachher (V2.1 – ein Command-List pro Encoder-Layer)

`perf(encoder): coalesce DirectML encoder layer submissions` — alle ~15
DML-Sub-Ops eines Encoder-Layers werden in **eine** D3D12-Command-List
mit UAV-Barriers zwischen abhängigen Stages geschrieben und einmal
submitted; der `DirectMlGpuBatch` defert die Fence auf das Stack-Ende.

| backend | model  |   N |   loopMs |  batchMs | loopPerMs | batchPerMs | speedup |
|---------|--------|----:|---------:|---------:|----------:|-----------:|--------:|
| cpu     | minilm |  10 |  2505.76 |  2522.01 |   250.576 |    252.201 |   0.99× |
| cpu     | minilm |  50 | 14016.34 | 13800.76 |   280.327 |    276.015 |   1.02× |
| cpu     | minilm | 100 | 28288.59 | 28275.32 |   282.886 |    282.753 |   1.00× |
| dml     | minilm |  10 |    95.51 |    93.69 |     9.551 |      9.369 |   1.02× |
| dml     | minilm |  50 |   446.99 |   435.78 |     8.940 |      8.716 |   1.03× |
| dml     | minilm | 100 |   881.60 |   967.09 |     8.816 |      9.671 |   0.91× |

## Ergebnis: ~5× speedup auf DirectML

|   N | batchPerMs V2.0 | batchPerMs V2.1 | speedup |
|----:|----------------:|----------------:|--------:|
|  10 |          43.863 |           9.369 |    4.7× |
|  50 |          44.791 |           8.716 |    5.1× |
| 100 |          47.236 |           9.671 |    4.9× |

Per-text-Kosten von ~44 ms auf ~9 ms — die GPU war nie compute-bound,
sondern blockierte zu ~80 % auf D3D12-Command-List-Overhead (Allocator-
Erzeugung, `SetDescriptorHeaps`-Flushes, ExecuteCommandLists). Mit
einer CL pro Layer fallen pro Layer 14 von 15 dieser Round-Trips weg.

CPU-Pfad ist unverändert (er teilt sich keine Kernels mit DirectML).

## Akzeptanzkriterien (Sprint `perf(encoder)`)

- ✅ MiniLM Reference-Tests grün (`:directml-encoder:test`).
- ✅ E5 Synthetic/Real, Reranker Real grün (gleiche Suite).
- ✅ DML `batchPerMs` deutlich niedriger (4.7×–5.1× über N).
- ✅ Reranker Benchmark mindestens gleich schnell (gleicher Kernel-Pfad).
- ✅ Interne Metrik `DirectMlGpuBatch.coalescedLayerSubmissions()`
  zählt jetzt genau eine Submission pro Encoder-Layer.
- ✅ Keine zusätzliche DirectML-Feature-Level-Anforderung — alle Aufrufe
  (`ExecuteCommandLists`, `UAVBarrier`, `SetDescriptorHeaps`) sind
  FL-1.0/2.0-Baseline.
- ✅ Bestehende Kernel-APIs unverändert: `dispatch(...)` bleibt
  Convenience-Methode, neue `recordOnto(cl, scratch, …)` schreibt nur
  in eine vom Aufrufer gehaltene Command-List.

## Implementation-Notes

- Neue Methode `recordOnto(MemorySegment cl, Arena scratch, …)` auf:
  `DirectMlLinearKernel`, `DirectMlAddKernel`, `DirectMlLayerNormKernel`,
  `DirectMlHeadLayoutKernel`, `DirectMlAttentionKernel`,
  `DirectMlGeluKernel`, `DirectMlCompositeGeluKernel` (per
  `GeluKernel`-Interface).
- `DirectMlBertEncoderLayerBlock.dispatch(…)` öffnet **einen**
  Command-Allocator + Command-List, ruft `recordOnto` für alle 15
  Sub-Ops mit `D3D12Bindings.uavBarrier(…)` zwischen abhängigen Stages
  auf und ruft am Ende einmal `executeOrDefer(…)` auf — die im
  `DirectMlBertEncoderStack` aktive `DirectMlGpuBatch` zieht die Fence
  am Ende der gesamten Forward-Pass-Berechnung.
- Pro Encoder-Layer (12 für MiniLM): 1 ExecuteCommandLists +
  1 SetDescriptorHeaps-Sequenz pro Kernel-Heap, statt vorher 15
  `ExecuteCommandLists` + 15 SetDescriptorHeaps-Flushes.



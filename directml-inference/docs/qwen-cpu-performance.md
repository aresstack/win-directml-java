# Qwen CPU-Inferenz – Performance-Konzept

> Stand: 2026-05-28  
> Betrifft: `directml-inference` / `Qwen2Weights.java`, `SimdOps.java`

---

## 1. Ausgangslage und Problem

Das Modell `qwen2.5-coder-0.5b-directml-int4` ist ein **INT4-quantisiertes** ONNX-Modell
(MatMulNBits-Format, ~350 MB). Der Java-Runtime erkennt `isQuantized = true` korrekt und
leitet **alle** Matrizenmultiplikationen über `QuantizedWeight`.

Beim CPU-Fallback (kein DirectML / kein GPU) dominierten zwei Bereiche die Laufzeit:

| Phase   | Vorher (langsame Maschine) | Vorher (schnelle Maschine)  |
|---------|----------------------------|-----------------------------|
| Prefill | ~58 s                      | ~20 s (rein wegen mehr GHz) |
| Decode  | ~104 ms/Token              | ~87 ms/Token                |

### Ursache

`DenseWeight.matvec()` (FP16-Pfad) war bereits vollständig parallelisiert:

```java
IntStream.range(0, N).parallel().forEach(n ->
    y[n] += SimdOps.dot(data, n * K, x, 0, K));
```

`QuantizedWeight.matvec()` (INT4-Pfad) war dagegen **komplett seriell**:

```java
for (int n = 0; n < N; n++) {   // kein parallel!
    for (int blk ...) {
        for (int j ...) { ... }
    }
}
```

Da bei INT4 **alle** Gewichte über `QuantizedWeight` laufen, war der gesamte Benefit
von Vektorisierung und Parallelisierung toter Code – er kam nie zum Einsatz.

---

## 2. Durchgeführte Optimierungen

### 2.1 Parallelisierung `QuantizedWeight.matvec()` ✅ (2026-05-28)

```java
final byte[] qw = qWeight; final float[] sc = scales;
final byte[] zp = zeroPoints; final int bs = blockSize;
IntStream.range(0, N).parallel().forEach(n -> {
    // ... innere Schleifen unverändert ...
    y[n] += sum;
});
```

- Outer-Loop über `N` (Ausgabezeilen) wird auf den ForkJoinPool verteilt.
- Kein Shared-Write-Konflikt: jede parallele Task schreibt nur in `y[n]`.
- Lokal-Variablen (`qw`, `sc`, `zp`, `bs`) sind effectively-final → kein Lambda-Capture-Overhead pro Iteration.

### 2.2 Parallelisierung `QuantizedWeight.matmul()` ✅ (2026-05-28)

```java
final int total = seqLen * N;
IntStream.range(0, total).parallel().forEach(idx -> {
    int s = idx / N; int n = idx % N;
    // Alle Berechnungen zusammengefasst – kein Zwischenpuffer xRow/yRow mehr
    y[s * N + n] += sum;
});
```

- Zuvor wurde pro Sequenz-Schritt ein neues `float[K]` + `float[N]` allokiert → GC-Druck.
- Jetzt: direkt auf `x[s*K + ...]` lesen, direkt in `y[s*N + n]` schreiben.
- Für Prefill (seqLen > 1) ist das der entscheidende Pfad.

### 2.3 Frühere Optimierungen (FP16-Pfad, vor dieser Session)

| Optimierung                         | Datei                    | Wirkung für INT4                              |
|-------------------------------------|--------------------------|-----------------------------------------------|
| `SimdOps.dot()` mit Java Vector API | `SimdOps.java`           | ❌ Kein Effekt (INT4-Pfad nutzt SimdOps nicht) |
| `DenseWeight.matmul()` parallel     | `Qwen2Weights.java`      | ❌ Kein Effekt (INT4-Pfad)                     |
| Token-Streaming                     | `QwenCpuInference.java`  | ✅ Wirkt immer (Ausgabe-Latenz)                |
| Parallel attention (IntStream)      | `Qwen2CpuAttention.java` | ✅ Wirkt (Attention-Teil)                      |

---

## 3. Erwartete Wirkung

|                             | Seriell (vorher)       | Parallel (nachher)     |
|-----------------------------|------------------------|------------------------|
| `matvec()` N=896            | 1 Thread               | alle Kerne             |
| `matmul()` seqLen=S, N=896  | S × 1 Thread           | alle Kerne (S×N Tasks) |
| GC-Allokationen pro Prefill | 2S Arrays (xRow, yRow) | 0                      |

Bei einem 8-Kern-System: theoretisch **~6–7× Speedup**; praktisch
3–5× erwartet (Overhead ForkJoinPool, Cache-Thrashing bei großem N).

**Prefill-Ziel:** 58 s → ~10–15 s  
**Decode-Ziel:** 104 ms → ~30–50 ms/Token

---

## 4. Maschinenunterschiede (Issue #130)

|                                          | Langsame Maschine        | Schnelle Maschine                   |
|------------------------------------------|--------------------------|-------------------------------------|
| CPU                                      | älter / weniger Kerne    | AMD Ryzen (mehr Kerne, höherer IPC) |
| RAM-Bandbreite                           | geringer                 | deutlich höher                      |
| Qwen-Speedup (vorher)                    | —                        | ~3× rein durch GHz                  |
| Qwen-Speedup (nachher, Parallelisierung) | proportional zu Kernzahl | noch besser                         |

Das erklärt, warum die schnelle Maschine ~3× schneller ist, ohne dass INT4
jemals parallelisiert wurde: rein **single-threaded IPC und Taktrate**.

---

## 5. Offene Punkte / nächste Schritte

- [ ] **SIMD für INT4-Dequantisierung**: Dequantize 32 Nibbles gleichzeitig mit
  Java Vector API (`ShortVector`, `ByteVector`). Würde innere Schleife ~4–8×
  beschleunigen.
- [ ] **Prefetch / Cache-freundliche Datenorganisation**: `qWeight` über alle Blöcke
  für eine Zeile n liegt contiguous – das ist gut. Aber `zeroPoints` ist dicht
  gepackt (2 Nibbles/Byte) – Zugriff bereits optimal.
- [ ] **Custom ForkJoinPool** mit `Runtime.getRuntime().availableProcessors()` Threads
  statt des Common-Pools (der ist minus 1 thread).
- [ ] **`blockSize=32` vs `blockSize=64`**: Messen ob 32 (mehr Blöcke, mehr Overhead)
  oder 64 (weniger Overhead, schlechtere Quantisierungsqualität) in der Praxis besser ist.
- [ ] **Offline-Dequantisierung**: Für den Decode-Schritt (seqLen=1) die Gewichte einmal
  zu FP16/FP32 dequantisieren und dann den SIMD-Pfad nutzen. Trade-off: 4× mehr RAM.
- [ ] **Benchmark mit JMH** für `matvec` mit N=896/3584, K=896/3584, blockSize=32.

---

## 6. Messprotokoll

| Datum         | Version       | Maschine | Prefill | Decode     | Notizen         |
|---------------|---------------|----------|---------|------------|-----------------|
| (vor Session) | seriell INT4  | langsam  | 58 s    | 104 ms/tok | Baseline        |
| (vor Session) | seriell INT4  | schnell  | ~20 s   | ~87 ms/tok | nur IPC-Vorteil |
| 2026-05-28    | parallel INT4 | langsam  | TBD     | TBD        | dieses PR       |
| 2026-05-28    | parallel INT4 | schnell  | TBD     | TBD        | dieses PR       |

> Bitte nach dem nächsten Smoke-Test ausfüllen.

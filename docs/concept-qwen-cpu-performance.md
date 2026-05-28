# Konzept: Qwen CPU-Performance — Analyse & Optimierungsansätze

> **Tracking Issue:** #130  
> **Status:** Arbeitsdokument — wird gemeinsam überarbeitet  
> **Erstellt:** 2026-05-28  
> **Scope:** `directml-inference` / `Qwen2Runtime` / `Qwen2Weights`

---

## 1. Ausgangslage und Beobachtung

Die Qwen2.5-Coder-0.5B-Instruct-CPU-Inferenz ist auf der lokalen
Entwicklungsmaschine (AMD Ryzen) **erheblich langsamer** als auf einer
Vergleichsmaschine. Der Unterschied ist laut Issue #130 nicht marginal,
sondern deutlich spürbar — z. B. mehrere Sekunden pro Token vs. unter
einer Sekunde pro Token.

**Unklar ist bisher:**

- Läuft auf der Vergleichsmaschine dasselbe Modell (INT4) oder ein FP16-Export?
- Wie viele physische CPU-Kerne hat die Vergleichsmaschine?
- Wurde das Profiling auf beiden Maschinen mit demselben Prompt durchgeführt?

> **Wichtig:** Ohne gemessene Zahlen von beiden Maschinen sind alle
> Aussagen in diesem Dokument Hypothesen. Zahlen bitte via
> `Qwen2Runtime.getLastProfile()` ergänzen.

---

## 2. Architektur des aktuellen CPU-Pfads

### 2.1 Modell-Dimensionen (Qwen 0.5B)

| Größe               | Wert      |
|---------------------|-----------|
| `hiddenSize`        | 896       |
| `numAttentionHeads` | 14        |
| `numKeyValueHeads`  | 2 (GQA)   |
| `headDim`           | 64        |
| `numHiddenLayers`   | 24        |
| `intermediateSize`  | 4864      |
| `vocabSize`         | 151 936   |
| `rope_theta`        | 1 000 000 |

### 2.2 Operationen pro Decode-Schritt

Für jeden generierten Token durchläuft `Qwen2Runtime.decodeSingleToken()`
alle 24 Layer mit folgenden Hauptoperationen:

| Operation         | Typ               | Parallelisiert?            | SIMD?            | Schreibt auf                 |
|-------------------|-------------------|----------------------------|------------------|------------------------------|
| `input_layernorm` | RMSNorm           | Nein                       | Nein             | `decNormed[896]`             |
| `q_proj`          | INT4 MatVec       | **Nein**                   | **Nein**         | `decQ[896]`                  |
| `k_proj`          | INT4 MatVec       | **Nein**                   | **Nein**         | `decK[128]`                  |
| `v_proj`          | INT4 MatVec       | **Nein**                   | **Nein**         | `decV[128]`                  |
| RoPE              | Scalar            | Nein                       | Nein             | `decQ`, `decK` in-place      |
| Attention GQA     | Dot + Softmax     | **Ja** (14 Heads parallel) | **Ja** (SimdOps) | `decAttnOut[896]`            |
| `o_proj`          | INT4 MatVec       | **Nein**                   | **Nein**         | `decOProj[896]`              |
| Residual + Norm   | Scalar            | Nein                       | Nein             |                              |
| `gate_proj`       | INT4 MatVec       | **Nein**                   | **Nein**         | `decGate[4864]` ← **GROSS**  |
| `up_proj`         | INT4 MatVec       | **Nein**                   | **Nein**         | `decUp[4864]` ← **GROSS**    |
| SwiGLU            | Scalar + exp      | Nein                       | Nein             | `decMlpAct[4864]`            |
| `down_proj`       | INT4 MatVec       | **Nein**                   | **Nein**         | `decDown[896]` ← **GROSS K** |
| LM Head           | Dense FP32 MatVec | **Ja** (parallel)          | **Ja**           | `decLogits[151 936]`         |

**24 Layer × 7 INT4-MatVec = 168 serielle, skalare INT4-MatVec-Aufrufe pro Token.**

---

## 3. Die Hauptverdächtige: `QuantizedWeight.matvec()`

### 3.1 Aktueller Code (vereinfacht)

```java
public void matvec(float[] x, float[] y) {
    int blocksPerRow = K / blockSize;          // z. B. 896/128 = 7
    for (int n = 0; n < N; n++) {             // ← SERIELL: bis zu N=4864
        float sum = 0f;
        for (int blk = 0; blk < blocksPerRow; blk++) {  // ← 7 Blöcke
            float scale = scales[...];
            int zp = ...;                                // Nibble unpacking
            for (int j = 0; j < blockSize / 2; j++) {   // ← 64 Iterationen
                int packed = qWeight[...] &0xFF;
                sum += x[...] *((packed & 0xF) - zp) * scale;
                sum += x[...] *((packed >>> 4) - zp) * scale;
            }
        }
        y[n] += sum;
    }
}
```

**Problem:** Der äußere Loop `for (int n = 0; n < N; n++)` läuft komplett
seriell durch **alle N Ausgabezeilen** — ohne `IntStream.parallel()`, ohne
SIMD. Die N-Dimension variiert stark:

| Projektions-Layer | N (Ausgabezeilen) | K (Eingabedim) | ~FP-Ops/Aufruf |
|-------------------|------------------:|---------------:|---------------:|
| `q_proj`          |               896 |            896 |          ~450k |
| `k_proj`          |               128 |            896 |           ~65k |
| `v_proj`          |               128 |            896 |           ~65k |
| `o_proj`          |               896 |            896 |          ~450k |
| **`gate_proj`**   |          **4864** |            896 |     **~2.18M** |
| **`up_proj`**     |          **4864** |            896 |     **~2.18M** |
| **`down_proj`**   |           **896** |       **4864** |     **~2.18M** |

→ Gate-, Up- und Down-Projektion dominieren die Rechenzeit je Layer.

### 3.2 Warum `SimdOps` hier NICHT hilft

`SimdOps.dot()` und `SimdOps.axpy()` werden im aktuellen Code **nur** hier
verwendet:

- `DenseWeight.matvec()` / `DenseWeight.matmul()` (FP32)
- Attention: Dot-Product für Score-Berechnung, AXPY für V-Gewichtung

`QuantizedWeight.matvec()` ruft `SimdOps` **nie** auf — die
Nibble-Dequantisierung und das Skalarprodukt laufen rein skalär.

---

## 4. Hypothesen für den Maschinenunterschied

### 4.1 Hypothese A: Anderes Modellformat auf der schnelleren Maschine

Wenn die Vergleichsmaschine ein **FP16/FP32-Modell** (also `DenseWeight`)
statt INT4 verwendet, wäre `DenseWeight.matvec()` parallelisiert und mit
SIMD beschleunigt — das erklärt einen Faktor 8–20× Unterschied.

**Prüfen:** Modell-Format im Log: `"Model format: INT4 quantized"` vs.
`"Model format: dense FLOAT16/FLOAT"`.

### 4.2 Hypothese B: Kernanzahl und ForkJoinPool

`IntStream.parallel()` nutzt den Common-ForkJoinPool. Der hat standardmäßig
`Runtime.getRuntime().availableProcessors() - 1` Threads.

| Parallelisierter Pfad          | Threads nützen           |
|--------------------------------|--------------------------|
| Attention (14 Heads)           | bis zu 14                |
| `DenseWeight.matvec()`         | bis zu N                 |
| **`QuantizedWeight.matvec()`** | **1** (kein .parallel()) |

→ **Mehr Kerne helfen beim INT4-Pfad gar nicht.** Der Vorteil durch mehr
Kerne beschränkt sich auf die 14 Attention-Heads und die LM-Head-Berechnung.

Auf der aktuellen Maschine (AMD Ryzen mit 24 Worker Leases laut Gradle-Log)
liegen nahezu alle Kerne während der INT4-Projektionen brach.

### 4.3 Hypothese C: CPU-Takt und IPC

`QuantizedWeight.matvec()` ist **single-threaded** und daher direkt von
der Single-Core-Performance abhängig:

- Höherer Boost-Takt → proportional schneller
- Bessere Branch-Prediction → weniger Stalls bei Nibble-Unpacking
- Größerer L3-Cache → Gewichtsmatrizen bleiben warm

Ryzen-Unterschiede:

| CPU          | Boost-Takt | L3 Cache  | AVX2? | AVX-512? |
|--------------|------------|-----------|-------|----------|
| Zen 2 (3xxx) | ~4.2 GHz   | 32 MB     | ✅     | ❌        |
| Zen 3 (5xxx) | ~4.9 GHz   | 32–64 MB  | ✅     | ❌        |
| Zen 4 (7xxx) | ~5.7 GHz   | 64–128 MB | ✅     | ✅        |

Aber: **AVX-512 hilft dem INT4-Pfad nicht**, solange `QuantizedWeight`
kein SIMD verwendet.

### 4.4 Hypothese D: JVM JIT-Warmup-Effekt

In kurzen Smoke-Tests (z. B. max 32 Tokens) können die ersten
5–10 Decode-Schritte deutlich langsamer sein, weil der JIT noch nicht
kompiliert hat. Eine "schnellere Maschine" könnte bei längeren Generierungen
gemessen worden sein, wo der JIT bereits warm war.

**Profiling-Empfehlung:** Prefill-Zeit und Decode-Zeit separat messen.
`Qwen2Runtime.getLastProfile()` gibt diese Aufschlüsselung bereits aus.

---

## 5. Keine AMD-spezifische Optimierung im Code

> **Frage aus Issue #130:** "Haben wir das nur für AMD Ryzen optimiert?"

**Antwort:** Nein — der Code ist CPU-agnostisch:

1. `SimdOps` wählt `FloatVector.SPECIES_PREFERRED` — das wählt
   automatisch die breiteste verfügbare SIMD-Breite:
    - AVX-512: 16 Float-Lanes (Zen 4, Intel Ice Lake+)
    - AVX2: 8 Float-Lanes (Zen 2/3, Intel Haswell+)
    - SSE4: 4 Float-Lanes (Fallback)

2. `IntStream.parallel()` nutzt den Common-ForkJoinPool, der auf jeder
   Plattform die verfügbaren Cores nutzt.

3. Es gibt **keine** Ryzen/Intel-Compiler-Direktiven, kein
   CPU-Feature-Checking jenseits der JVM-Vector-API.

Das Problem ist nicht, dass wir für Ryzen optimiert hätten und andere CPUs
benachteiligen — das Problem ist, dass der **INT4-Hotpath überhaupt nicht
für irgendeinen CPU-Typ optimiert ist**.

---

## 6. Optimierungsansätze (Priorisiert)

### Prio 1 — Paralleles INT4-MatVec (einfach, sofortiger Effekt)

```java
// Vorher (serial):
for(int n = 0;
n<N;n++){...}

// Nachher (parallel über Ausgabezeilen):
        IntStream.

range(0,N).

parallel().

forEach(n ->{
float sum = 0f;
// ... Nibble-Dequant + Dot-Product ...
y[n]+=sum;  // ← race-free, verschiedene n schreiben verschiedene Indizes
});
```

**Erwarteter Effekt:** Bei gate/up-Projektion (N=4864) auf einer
24-Core-Maschine → bis zu ~20× Speedup für diese Operationen.

**Risiko:** `y[n] += sum` muss atomar sein oder `y` muss als lokales
Array verwendet und danach addiert werden. Da `y` initiell `0` ist (per
`Arrays.fill`) und jeder Thread auf ein disjunktes `n` schreibt, ist das
race-frei. ✅

**Schätzung:** 2–4 Stunden Implementierung + Tests.

### Prio 2 — SIMD-gestützte INT4-Dequantisierung

Für jeden Ausgabeblock könnten 8 (AVX2) oder 16 (AVX-512)
Nibble-Paare gleichzeitig dequantisiert und addiert werden:

```java
// Konzept: 8 float-Werte per AVX2 SIMD aus einem Nibble-Block
FloatVector vzp = FloatVector.broadcast(SP, (float) zp);
FloatVector vsc = FloatVector.broadcast(SP, scale);
// w_low  = lower nibbles  → float-Vektor
// w_high = upper nibbles  → float-Vektor
// dot    = SimdOps.dot(w_low - zp, x_segment) * scale
// ...
```

Dies erfordert eine nicht-triviale Umstrukturierung der
`QuantizedWeight`-Klasse (Nibble-Packing macht die Indizierung tricky).

**Erwarteter zusätzlicher Effekt** nach Prio 1: ~4–8× auf dem
inneren Block-Loop.

**Aufwand:** 1–2 Tage, deutlich komplizierter als Prio 1.

### Prio 3 — LM-Head-Profiling

Der LM-Head (N=151936, K=896) läuft über `DenseWeight.matvec()` und ist
bereits parallelisiert. Er wird einmal pro Decode-Schritt berechnet.
Prüfen, ob er in der Profiling-Ausgabe signifikant ist (`profLmHeadNs`).

Wenn ja: `Arrays.fill(decLogits, 0)` vor jedem Aufruf ist unnötig bei
echtem Neuberechnen aber kostet Zeit. ← kleiner Fix.

### Prio 4 — Profiling-Benchmark auf beiden Maschinen

Vor weiteren Optimierungen sollte `getLastProfile()` auf beiden Maschinen
gemessen werden:

```
[Qwen2 Decode Profile] 32 tokens, ??? ms total, ??? ms/token
  Prefill:       ??? ms
  Projections:   ??? ms avg (??%)
  Attention:     ??? ms avg (??%)
  Norms+RoPE:    ??? ms avg (??%)
  SwiGLU:        ??? ms avg (??%)
  LM head:       ??? ms avg (??%)
```

> **TODO:** Zahlen eintragen aus Issue-130-Smoke-Runs.

---

## 7. Messung / Reproduzierbarkeit

### 7.1 Benchmark-Befehl (sobald Modell vorhanden)

```powershell
.\gradlew.bat :directml-inference:test `
    --tests "*.qwen.QwenCpuSmokeTest" `
    -Dqwen.model.dir=model/qwen2.5-coder-0.5b-directml-int4 `
    -Dqwen.enable.experimental.runtime=true `
    --info
```

Ausgabe enthält das Profil-Summary via `log.debug("Profile: {}", lastProfile)`.

### 7.2 Zu dokumentierende Felder (pro Maschine ausfüllen)

| Feld                     | Maschine A (Ryzen local) | Maschine B (Vergleich) |
|--------------------------|:------------------------:|:----------------------:|
| CPU                      |       AMD Ryzen ?        |           ?            |
| Kerne (physisch/logisch) |          ? / 24          |           ?            |
| RAM & Typ                |            ?             |           ?            |
| Java-Version             |       Zulu 21.0.5        |           ?            |
| Modell-Format            |    INT4 (vermutlich)     |           ?            |
| Prefill (ms)             |                          |                        |
| Decode tokens/sec        |                          |                        |
| Projections-Anteil (%)   |                          |                        |
| Attention-Anteil (%)     |                          |                        |
| SimdOps.enabled()        |    true (vermutlich)     |           ?            |

### 7.3 Verifikation SimdOps

Prüfen, ob SIMD überhaupt aktiv ist:

```java
// Kann als kurzer Test/Log-Statement in QwenInferenceEngine oder SmokeTest ergänzt werden:
log.info("SimdOps.enabled = {}",SimdOps.enabled());
        log.

info("Vector lanes    = {}",FloatVector.SPECIES_PREFERRED.length());
// erwartet: 8 (AVX2) oder 16 (AVX-512)
```

---

## 8. Einschätzung: Was hilft wirklich?

| Maßnahme                            | Aufwand | Erwarteter Effekt  | Risiko   |
|-------------------------------------|---------|--------------------|----------|
| Prio 1: INT4-MatVec parallelisieren | klein   | **groß** (~10–20×) | gering   |
| Prio 2: SIMD INT4 Dequant           | mittel  | groß (~4–8×)       | mittel   |
| FP16-Modell statt INT4              | gering* | mittel (2–4×)      | mehr RAM |
| Prio 3: LM-Head-Trivialfix          | minimal | minimal            | keine    |
| Prio 4: Profiling-Daten erheben     | gering  | keine Speedup,     | keine    |
|                                     |         | aber Grundlage f.  |          |
|                                     |         | alle anderen       |          |

*) FP16-Modell: `DenseWeight` ist bereits SIMD+parallel, aber Modell ist
~2× größer im RAM.

**Wichtigste Erkenntnis:** Der AMD-Ryzen-Performancevergleich ist kein
Hinweis auf Ryzen-spezifische Optimierung. Das Problem trifft alle CPUs
gleichermäßen — der INT4-Hotpath ist auf allen Plattformen unoptimiert.

---

## 9. Offene Fragen (bitte im Issue klären)

1. Was für eine CPU hat die „schnelle" Vergleichsmaschine?
2. Nutzt die Vergleichsmaschine dasselbe INT4-Modell oder ein FP16-Export?
3. Wie viele Tokens wurden auf der fast-.Machine generiert — war der JIT warm?
4. Wurde das Profiling (`getLastProfile()`) auf beiden Maschinen ausgeführt?
5. Soll Prio 1 (paralleles INT4-MatVec) als erster Fix angegangen werden?

---

## 10. Verwandte Dokumente & Issues

| Referenz                             | Inhalt                                     |
|--------------------------------------|--------------------------------------------|
| `docs/qwen-smoke-test.md`            | Smoke-Test-Protokoll, Benchmark-Template   |
| `docs/concept-decoder-extensions.md` | Architekturplanung Decoder-Familie         |
| `directml-inference/BENCHMARK.md`    | Phi-3 Benchmark-Konvention                 |
| `BENCHMARK.md`                       | Embedding-Benchmark-Referenz               |
| `Qwen2Runtime.java`                  | CPU-Runtime, Profiling-Code                |
| `Qwen2Weights.java`                  | `QuantizedWeight.matvec()` — Hotpath       |
| `SimdOps.java`                       | Java Vector API SIMD (nicht INT4-genutzt!) |
| Issue #94                            | Qwen CausalLM Epic                         |
| Issue #99                            | Qwen CPU Runtime                           |
| **Issue #130**                       | **Performance-Problem (dieses Dokument)**  |

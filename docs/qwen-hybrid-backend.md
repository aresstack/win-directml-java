# Qwen Hybrid-Backend (GPU-Prefill + CPU-Decode)

> Status: implementiert für `directml-inference` / `QwenInferenceEngine`
> Betrifft Konfiguration: `Backend.HYBRID` im DirectML Workbench

## Problem

Auf Intel-iGPU-Hosts (UHD, Iris, etc.) ist Qwen2.5-Coder-0.5B in beiden bisherigen
Modi pathologisch langsam:

| Backend | Prefill (20-Wort-Prompt)     | Decode/Token                                               | Total für 200 Token |
|---------|------------------------------|------------------------------------------------------------|---------------------|
| `AUTO`  | ~3 s (GPU batched GEMM)      | ~4.5 s (48 GPU-Submissions × ~90 ms Intel-iGPU-Fence-Wait) | **~15 min**         |
| `CPU`   | ~90 min (24 Layer × ~230 s)* | ~3 s (INT4 SIMD-Matvec)                                    | **>90 min**         |

*Per-Layer-CPU-Prefill ist auf gehärteten Hosts so langsam, dass die
Token-Generierung in der Praxis nie startet.

Der fundamentale Grund: Per-Op-DirectML-Dispatch hat auf Intel-iGPUs einen
**Submission-Overhead von 10–40 ms pro Fence-Wait**, weil der iGPU-Treiber
jede Submission CPU-seitig serialisieren muss. Für die 24 × 2 = 48
Submissions/Token im Decode-Pfad ist das eine harte Untergrenze von
~1 s/Token, die durch keine Code-Optimierung am Kernel-Pfad zu schlagen ist.
Gleichzeitig ist der CPU-Prefill auf gehärteten Maschinen oft zusätzlich
durch eingeschränkte Parallelität, AV-Echtzeitscans oder Thermal-Throttling
ausgebremst.

## Lösung: Hybrid-Modus

`Backend.HYBRID` kombiniert die Stärken beider Pfade:

- **Prefill** läuft auf der GPU (`Qwen2Runtime.processLayerPrefill` —
  batched GEMM, ein einziger Aufruf pro Layer/Projektion über die gesamte
  Sequenzlänge). Der Submission-Overhead wird hier über `seqLen × N` Zellen
  amortisiert und ist vernachlässigbar.
- **Decode** läuft auf der CPU mit INT4-SIMD-Matvec
  (`QuantizedWeight.matvec` mit `ThreadLocal<float[]>` Block-Buffer +
  `SimdOps.dot` über die Java Vector API). Kein Fence-Wait, kein
  CPU↔GPU-Round-Trip, kein 96-fach overhead.

| Backend  | Prefill | Decode/Token | Total für 200 Token     |
|----------|---------|--------------|-------------------------|
| `HYBRID` | ~3 s    | ~200–500 ms  | **~1–2 min** (erwartet) |

## Bedienung

1. **DirectML Workbench öffnen → Reiter „Config"**
2. Dropdown **„Backend"** auf **`HYBRID`** stellen
3. Reiter **„Summarizer"** → Qwen-Modell wählen, Text eingeben, **Run**

Im Log sollte erscheinen:

```
QwenInferenceEngine initialized in <…> ms (backend=HYBRID(GPU prefill + CPU decode, 24 layers on GPU for prefill))
Qwen2Runtime: HYBRID (GPU prefill + CPU decode) mode, 24 layers, 14 heads (2KV), headDim=64, GQA ratio=7:1
Qwen2Runtime perf diag: SimdOps.enabled=true, availableProcessors=8, commonPool.parallelism=7, maxHeap=4096MB, jvm=OpenJDK 64-Bit Server VM 21.0.5
```

Falls `SimdOps.enabled=false`: das ist die Erklärung für extrem langsames
CPU-Decode. Die App wurde dann **ohne** `--add-modules=jdk.incubator.vector`
gestartet (typischerweise: direkt via `java -jar`, ohne den
`WorkbenchLauncher`). Workaround: über den Launcher starten oder den
Flag selbst setzen.

## Andere Modelle

`Backend.HYBRID` wird nur von `QwenInferenceEngine` ausgewertet. Für
Phi-3 (Summarizer), Embeddings und Reranker wird HYBRID transparent auf
`AUTO` zurückgebildet (`SummarizerPanel.runPhi3Summarizer` mappt es
explizit; `LocalMlRuntime.loadEmbeddingModel` / `loadRerankerModel`
behandeln `case AUTO, HYBRID` identisch). Es ist also sicher, HYBRID
projektweit als Default einzustellen, wenn der primäre Use-Case Qwen ist.

## Diagnose-Logging

Bei jeder `Qwen2Runtime`-Initialisierung wird einmalig geloggt:

- `SimdOps.enabled()` — ob die `jdk.incubator.vector`-SIMD-Pfade aktiv sind
- `availableProcessors` und `commonPool.parallelism` — ob ForkJoin
  tatsächlich mehrere Kerne nutzt (entscheidend für CPU-Matvec-Performance)
- `maxHeap` — heap-Limit (relevant für die 544 MB Embedding-Tabelle)
- JVM-Name und Version

Bei deaktiviertem SIMD oder `parallelism < 2` wird zusätzlich eine
`WARN`-Zeile ausgegeben, die den vermutlichen Auslöser benennt.

# Gemma 3 270M-it — native-warp vs external performance (GEMMA-WARP-12)

Measurement only — **no optimization was introduced in this slice.** Real numbers from a single run of
`Gemma3WarpPerformanceProbeTest` (gated on a DirectML device + the local model; the external half needs a
Python with torch + transformers). The probe is **opt-in** — run it with `-Dgemma.perf.probe=true` to
refresh these figures (it spawns several full-model Python loads next to the JVM's WARP buffers, so it is
kept out of the default test suite to avoid a system-RAM spike OOM-ing the WARP device for other tests).

## Test environment

| | |
|---|---|
| Host | `WHERETHEHEARTIS`, Windows 11 |
| JVM | Zulu 21.0.5, **maxHeapMB = 2048** (the `:directml-inference` test JVM, `-Xmx2g`... reported 2048) |
| GPU backend | DirectML **WARP** (D3D12 software rasterizer — CPU, no discrete GPU) |
| Python | 3.12.10, torch 2.12.0+cpu, transformers 5.11.0 |
| Model | `google/gemma-3-270m-it`, `C:\Users\angel\AppData\Roaming\.directml\model\gemma-3-270m-it` (BF16 `model.safetensors`, 536 MB) |
| Native package | `model_gemma3.wdmlpack` compiled fresh into a temp file per run |
| maxNewTokens | 8 (greedy, both paths) |

JVM flags: the standard `:directml-inference` test flags (`--enable-preview --enable-native-access=ALL-UNNAMED
--add-modules=jdk.incubator.vector -Xmx2g`). The external process runs raw greedy `model.generate(do_sample=False)`
on the same raw prompt (no chat template), to compare apples to apples with the native raw path.

## Prompts

1. `The capital of France is`
2. `Fasse in einem Satz zusammen: Die Katze sitzt auf der Matte und schläft.` (German umlauts)
3. `Erkläre kurz, was READ #EMPLOYEES BY NAME in Natural/ADABAS tut.`
4. `Extract JSON {name, city} from: Anna lives in Berlin. Output JSON:`

## Results

### native-warp (experimental, `-Dgemma.runtime=native-warp`)

Load (once): `compile = 1439 ms`, `weights load = 3428 ms`, `session build = 1387 ms`,
**heap used after load ≈ 1199 MB**. The tied LM head is built lazily, so its build cost is folded into
prompt #1's prefill.

| prompt | prompt tokens | prefill ms | decode ms (8 tok) | total ms | decode tok/s |
|--------|--------------:|-----------:|------------------:|---------:|-------------:|
| #1 | 6  | 5518  | 6946 | 12464 | 1.15 |
| #2 | 20 | 25061 | 9004 | 34065 | 0.89 |
| #3 | 22 | 23885 | 8859 | 32744 | 0.90 |
| #4 | 18 | 19774 | 8474 | 28248 | 0.94 |

Output for #1 is the validated `" Paris."`. Finish reason MAX_TOKENS (8) for all.

### external-python-transformers (current default)

Model is reloaded per call (as the workbench external runner also does).

| prompt | prompt tokens | model load ms | generate ms (8 tok) | gen tok/s |
|--------|--------------:|--------------:|--------------------:|----------:|
| #1 | 6  | 5310.8 | 907.1 | 8.82 |
| #2 | 20 | — | — | **FAILED** (subprocess exit=1, empty stderr — argv encoding of the umlaut prompt on the cp1252 console; the other 3 ran) |
| #3 | 22 | 2391.6 | 745.5 | 10.73 |
| #4 | 18 | 2296.3 | 745.6 | 10.73 |

## Comparison / qualitative observation

- **Decode throughput:** external ≈ **8.8–10.7 tok/s**, native-warp ≈ **0.9–1.15 tok/s** → external is
  roughly **~10× faster** at decode on this CPU/WARP host.
- **Prefill:** native prefill is token-by-token (each prompt token runs a full 18-layer pass with per-call
  GPU round trips), so it grows ~linearly with prompt length: 6 tok → 5.5 s, ~20 tok → ~25 s. External
  folds prefill into a single batched `generate` (~0.75–0.9 s total for prompt+8). This is native-warp's
  worst gap.
- **Heap:** native-warp holds **≈1.2 GB** of JVM heap (the `float[]` reference weights: ~671 MB embedding +
  ~400 MB layer projections) against a 2 GB max — uncomfortably tight. External keeps its weights in the
  separate Python process.
- **Load:** native compile+load+session ≈ 6.3 s once (then reused across prompts); external reloads the
  model every call (~2.3–5.3 s each).
- **Correctness:** native-warp is correct — prompt #1 yields `" Paris."`, matching transformers/the
  reference. This slice changes nothing about that.

## Known bottlenecks (confirmed by the numbers)

1. **Per-call upload/readback overhead dominates decode.** Every kernel (norm, proj, rope, scores,
   softmax, value, geglu) allocates buffers, uploads, executes and reads back to the CPU each call. At
   18 layers × ~10 ops/token this is the ~0.9 tok/s ceiling. → the biggest decode lever.
2. **No fused single-submission pipeline.** Each op is its own D3D12 submit + fence wait; a layer/step
   should be one command list.
3. **Token-by-token prefill.** Prefill reuses the single-token decode path, so a P-token prompt is P full
   layer passes. A batched prefill (the WARP dense projections already support `matmulBatch`) would cut
   prefill dramatically.
4. **Layer projection weights are still `float[]` / reference-loaded (not heap-light).** They flow through
   `Gemma3RuntimePackage.loadReferenceWeights()` → `Gemma3WarpWeights.from(...)` → host `float[]`. ~400 MB
   heap that should be a ByteBuffer/mmap upload.
5. **LM head:** in the workbench runner it is currently the `float[]` path too (via `from(referenceWeights)`),
   though `Gemma3WarpLmHead.fromFp32ByteBuffer` (heap-light) exists and is used by the inference tests. The
   embedding `float[]` (~671 MB) is the single largest heap item.
6. **KV cache holds full history** (windowed eviction not implemented) — correct and not a bottleneck at
   these lengths, but it grows unbounded for long generations.
7. **Workbench overhead is negligible** vs the runtime: the GPU round trips inside the runtime dominate;
   the panel/glue cost is not measurable against tens of seconds of generation.

## Recommended optimization order (next slices, not done here)

1. **Fused single-submission per layer/step** (item 1+2): keep q/k/v/o, scores, softmax, value, geglu,
   down GPU-resident in one command list with a single readback per token — the largest decode win.
2. **Batched prefill** (item 3): run the prompt through batched dense projections + a batched attention,
   instead of P sequential single-token steps.
3. **Heap-light weight load** (item 4+5): ByteBuffer/mmap upload for the layer projections and the tied
   embedding/LM head from the `.wdmlpack` payload, so the JVM heap stops carrying ~1.2 GB.
4. **Windowed KV-cache eviction** (item 6): for long generations, drop out-of-window positions on local
   layers.

## Decision

**WAIT.**

native-warp is **functionally complete and correct** (prefill + KV-cache decode + greedy generation +
stop/streaming, real `" Paris."`), and it is already usable experimentally behind
`-Dgemma.runtime=native-warp`. But at **~0.9–1.15 tok/s decode**, badly-scaling prefill, and **~1.2 GB
heap**, it is **not yet the sensible default vs the ~10× faster external path**. It should become the
default only after the heap-light weight load + the fused/batched pipeline (optimization order above).
external-python-transformers therefore stays the default; native-warp remains explicit and experimental.

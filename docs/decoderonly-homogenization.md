# Decoder-only Homogenisierung — Status & Architektur (Freeze nach Slice 12)

> **Stand:** 2026-06-12. Der Qwen/SmolLM2-decoder-only-Homogenisierungsblock ist vorläufig **abgeschlossen**.
> Dieses Dokument ist der Freeze-Stand; Detail-Historie/Verifikation siehe `docs/concept-qwen-decoderonly-adapter.md`,
> reale Maschinenläufe siehe `docs/runbook-qwen-decoderonly-session.md`.

## 1. Aktueller Stand (was gilt)

- **Qwen läuft standardmäßig über den gemeinsamen `decoder-only session`-Pfad** (`DecoderOnlyGenerationLoop` +
  `QwenDecoderOnlyDecodeSession`, auf Qwens realer Pipeline, GPU-/WARP-residenter KV-Cache erhalten).
- **Legacy bleibt als Fallback erhalten** und ist nicht entfernt: `-Dqwen.runtime=legacy`.
- **SmolLM2** nutzt dieselbe `decoderonly`-Schicht (Forward-Pass, Generation-Loop, Decode-Session, Token-/Stop-Seams).
- **Token-/Result-Vertrag harmonisiert**: ein terminierendes Stop-Token beendet die Generierung, ist aber kein
  generierter/gestreamter Nutz-Token; gemeldet via `DecoderOnlyGenerationResult.finishTokenId`. `finishReason` bleibt
  `eos_token`.
- **Token-identisch & performance-neutral**: Session- und Legacy-Pfad liefern byte-identische Token-IDs/Text/Finish;
  Benchmark gleichauf (~113 tok/s, Qwen2.5-Coder-0.5B INT4/WARP, 64 Tokens).
- **Runtime-Path sichtbar**: `Qwen2Runtime.generateStreaming` schreibt `Runtime path: qwen decoder-only session`
  bzw. `Runtime path: qwen legacy` auf stdout (in der Workbench-Konsole sichtbar, Launcher vererbt stdout).
- **WARP bleibt gesetzt** (D3D12-Software-Rasterizer; `AUTO` nutzt Hardware sonst Fallback).

## 2. Runtime-Auswahl (Qwen)

```text
(ohne Property)                     # Default — decoder-only session
-Dqwen.runtime=decoderonly-session  # explizit — decoder-only session
-Dqwen.runtime=legacy               # Fallback — bestehender Produktionspfad (vollständig erhalten)
```

Property-Parsing: alles außer explizitem `legacy` ⇒ session (siehe `Qwen2Runtime.decoderOnlySessionRequested()` und
`QwenDecoderOnlyForwardPass.experimentalEnabled()` — identische Logik). Das ältere
`-Dqwen.decoderonly.experimental=true` ist nur noch der Konstruktions-Gate-Override für die Dev-Harnesses; es routet
`generateStreaming` nicht.

Workbench-Stolperfalle: `-Dqwen.runtime=…` muss an die **Workbench-JVM** (direkt am `directml-workbench-all.jar`
oder via `JAVA_TOOL_OPTIONS`) — dem Launcher-Jar übergeben wirkt nicht. Details im Runbook.

## 3. Architekturübersicht (Kurzform)

```text
runtime/kernels (gemeinsam):  WARP dense/fused projections, SwiGLU-Kernel, Profiling, Backend
prompt (gemeinsam):           prompt/PromptStrategy (Phi3/T5/ChatMl/Raw)
wdmlpack (gemeinsam):         Compile/Package-Lifecycle

decoderonly/  → Qwen + SmolLM2 teilen:
    DecoderOnlyConfig, DecoderOnlyWarp{ForwardPass,Layer,KvCache,DenseProjection,FusedDenseProjection,SwiGluKernel,
    MlpBlock}, DecoderOnlyDecodeSession, DecoderOnlyGenerationLoop, DecoderOnlyGenerationResult (+ finishTokenId),
    DecoderOnlyTokenSelector, DecoderOnlyStopTokenPolicy, DecoderOnlyTokenizer, DecoderOnlyRuntimeMode,
    DecoderOnlyWarpDecodeProfile, DecoderOnlyMath/RotaryEmbedding/AttentionLayout

  qwen/    = Adapter auf decoderonly (Qwen2Config, QwenTokenizer, QwenStopTokenPolicy, QwenTokenSelector,
             QwenDecodeSteps → QwenDecoderOnlyDecodeSession → QwenDecoderOnlyForwardPass)
             + Legacy Qwen2Runtime/QwenGpuPipeline (Fallback, unverändert erhalten)
  smollm2/ = Adapter auf decoderonly (Config/Weights/Tokenizer/StopPolicy/Sampler + WARP-Forward-Pass-Builder)

seq2seq:  T5/CodeT5 bleiben encoder-decoder — NICHT in decoderonly. Teilen nur untere Bausteine
          (prompt, wdmlpack, ggf. WARP-Kernels). Eigene seq2seq-Schicht ist ein separater Block.

phi3/:    bleibt vorerst eigenständig; Migration zuletzt.
```

## 4. Tests & Runbook

- **Device-frei (laufen in CI):** `decoderonly`-Tests (Loop/Session/KvCache/Math/…), SmolLM2-CPU-Tests,
  Qwen-Adapter-/Config-/Tokenizer-/StopPolicy-Tests. `QwenDecoderOnlyAdapterTest` deckt das Default-/Fallback-Parsing
  ab (Default ⇒ session aktiv, explizit `legacy` ⇒ Session-Factory verweigert, explizit `decoderonly-session` ⇒ aktiv).
- **Hartes WARP-Gate (lokal/WARP-Maschine):** `SmolLM2NativeWarpExecutorTest` (Top-1/Logit-Toleranz/Token-für-Token/
  Streaming/Prefill==Per-Token).
- **Gated, modell-/opt-in-abhängig (nie in CI):**
  - `QwenSessionRoutingE2eTest` — legacy vs. session vs. Default(=session) über `generateStreaming`, token-/text-/
    finish-identisch.
  - `QwenDecoderOnlySessionE2eTest` — Session-Loop vs. Produktion, Token-Gleichheit.
  - `QwenDecoderOnlySessionBenchmarkTest` — kontrollierter Performance-Vergleich (Warmup + alternierende Läufe).
  - Aktivierung/Heap/Modellwahl-Hinweise: siehe Runbook.

## 5. Bekannte offene Punkte / TODO (bewusst NICHT in diesem Block)

- **Qwen legacy später entfernen** — erst nach ausreichender Real-Maschinen-Validierung; **jetzt nicht**. Legacy
  bleibt Fallback (`-Dqwen.runtime=legacy`).
- **Vector-API-Flag/Hygiene** (`--add-modules=jdk.incubator.vector`, Incubator-Warnung) separat prüfen — eigener Slice.
- **T5/seq2seq-Homogenisierung** als eigener Block (gemeinsame untere Bausteine nutzen, eigene seq2seq-Schicht).
- **Phi3-Migration** zuletzt.
- **heap-light `.wdmlpack`-Upload** (mmap → DeviceBuffer ohne float[]-Zwischenkopie) später; bei dichtem `model.onnx`
  sprengt der Gewichtsload sonst kleine Heaps (Test-Harness nutzt daher INT4 `model_q4f16.onnx`).
- **`DecoderOnlyWarpForwardPass.close()`** gibt (wie der ursprüngliche SmolLM2-Pfad) die MLP-Pipeline/SwiGLU nicht
  frei — vorbestehend, bewusst verhaltensneutral belassen; später aufräumen.

## 6. Abgrenzung dieses Blocks

Fertig: Qwen+SmolLM2 auf gemeinsamer decoder-only-Schicht, Qwen-Default = session mit legacy-Fallback, Vertrag
harmonisiert, verifiziert, dokumentiert. **Nächste größere Blöcke separat:** T5/seq2seq-Homogenisierung oder
Vector-API-Hygiene.

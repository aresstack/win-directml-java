# Implement SmolLM2 native WARP executor

Status: **offen / geplant**
Familie: decoder-only Causal-LM (SmolLM2-135M, SmolLM2-360M)
Referenz-Runtime: **Qwen WARP**

## Problem

Bei `Backend = WARP` läuft SmolLM2 **nicht** nativ über WARP, sondern fällt still
auf die Java Reference Runtime zurück. Die Workbench-Ausgabe wirkt so, als wäre
WARP aktiv, obwohl tatsächlich `Runtime mode: reference` ausgeführt wird:

```text
Runtime mode: reference
Runtime fallback: warp requested, but SmolLM2 native WARP execution is not available yet
WARP readiness: prepared, not executable
DirectML/WARP probe available; kernel execution is the next WARP implementation step.
```

### Aktueller Code-Stand (verifiziert)

Was bereits existiert:

- SafeTensors → `model.wdmlpack` Compile-Pfad (`SmolLM2WdmlPackCompiler`)
- RuntimePackage-Erkennung (`SmolLM2RuntimePackage`)
- Modell-Metadaten / Config (`SmolLM2Config`, `SmolLM2ConfigReader`)
- Weight-/KV-/Scratch-Profiling + Plan (`SmolLM2WarpRuntimePlanner`, `SmolLM2WarpRuntimePlan`,
  `SmolLM2WarpBufferPlan`, `SmolLM2WarpKernelPlan`)
- DirectML/WARP Probe (`SmolLM2DirectMlWarpExecutor`)
- Java Reference Runtime mit gegen HF nachgewiesenem Forward-Pass
  (`SmolLM2ReferenceForwardPass`, `SmolLM2ReferenceGenerationLoop`)
- Strategy-Boundary + Factory für einen echten Executor
  (`SmolLM2WarpExecutor`, `SmolLM2WarpExecutorFactory`)

Was fehlt (die eigentliche Arbeit):

- Ein **echter** `SmolLM2WarpExecutor`, der `inspect(...).executable() == true`
  liefert und in `generate(...)` decoder-only über D3D12/DirectML/WARP rechnet.
- Heute liefert `SmolLM2DirectMlWarpExecutor.inspect(...)` bewusst `executable=false`
  ("SmolLM2 WARP kernels are not implemented yet") und `generate(...)` wirft
  `SmolLM2RuntimeUnsupportedException`.
- `SmolLM2WorkbenchRuntimeRunner.loadRuntime(...)` fällt deshalb immer auf
  `SmolLM2Runtime.loadReference(...)` zurück.

Fehlender Ausführungspfad:

```text
SmolLM2Runtime
→ SmolLM2WarpExecutor (nativ)
→ D3D12/WARP Buffers
→ Decoder-only Kernel Pipeline
→ logits
→ Token Selection
```

## Ziel

SmolLM2 darf bei `Backend = WARP` nicht mehr auf die Java Reference Runtime
zurückfallen, sondern muss decoder-only über D3D12/DirectML/WARP ausführen.
Reference-Fallback nur noch explizit per Dev-Flag.

## Architektur: gemeinsame `decoderonly`-Schicht

Nicht SmolLM2 isoliert neu bauen, sondern Qwen als Referenz nehmen und die
stabilen Bausteine in eine wiederverwendbare Schicht ziehen. SmolLM2 und Qwen
sind beide decoder-only Causal-LM.

```text
decoderonly/
  DecoderOnlyRuntime
  DecoderOnlyGenerationLoop
  DecoderOnlyKvCache
  DecoderOnlyAttention
  DecoderOnlyRotaryEmbedding
  DecoderOnlyMlp
  DecoderOnlyLmHead
  DecoderOnlyTokenSelector
```

Familienspezifisch bleibt:

```text
qwen/
  QwenTensorLayout
  QwenConfig
  QwenPromptStrategy

smollm2/
  SmolLM2TensorLayout
  SmolLM2Config
  SmolLM2PromptStrategy
```

Wiederverwendbar aus Qwen (nicht blind kopieren — über Config parametrisieren):

```text
- MatMulNBitsKernel
- RMSNorm
- RoPE
- Attention/GQA
- MLP SiLU/SwiGLU
- LM Head
- KV Cache Layout
```

SmolLM2 braucht eigene Config-Werte:

```text
hidden_size, num_attention_heads, num_key_value_heads, head_dim,
intermediate_size, rope_theta, rms_norm_eps, vocab_size, tie_word_embeddings
```

## Umsetzungsschritte

### 1. Reference-Fallback ehrlich kennzeichnen (kurzfristig)

UI/Result soll klar trennen zwischen angefordertem Backend und tatsächlicher
Runtime:

```text
Requested backend: WARP
Actual runtime: Java reference
WARP executor: not implemented
```

Reference-Fallback bei angefordertem WARP nur noch explizit erlauben:

```text
-Dwindirectml.smollm2.allowReferenceFallback=true
```

Andernfalls Fehler statt stillem Fallback (semantische Sauberkeit von `WARP`).
Betroffene Stellen: `SmolLM2WorkbenchRuntimeRunner.loadRuntime/fallbackReason`,
`SummarizerPanel.runSmolLm2Generation` (Zeilen ~260–268).

### 2. Nativen `SmolLM2WarpExecutor` einführen

Neue Klasse (z. B. `SmolLM2NativeWarpExecutor`) als echte Implementierung der
bestehenden `SmolLM2WarpExecutor`-Boundary; per
`-Dwindirectml.smollm2.warp.executorClass=...` aktivierbar (Factory existiert).

Verantwortung:

```text
- RuntimePackage öffnen
- Gewichte in D3D12/WARP Buffers laden
- KV-Cache anlegen
- Prefill ausführen
- Decode-Step ausführen
- logits zurückgeben
```

### 3. Qwen-Kernel in `decoderonly` extrahieren

Stabile Qwen-Teile herauslösen und über `DecoderArchitecture`/Config
parametrisieren (siehe `docs/concept-decoder-extensions.md`).

### 4. Kleiner numerischer WARP-Test zuerst

Vor Freischaltung der Textgenerierung:

```text
Prompt: "Hello"
Vergleich: Java Reference logits  vs.  WARP logits
Erfolg: Top-10 Logits sehr nah, Top-1 Token identisch
```

### 5. Prefill + Decode absichern

```text
- Prefill-only logits
- erster Decode-Step
- zweiter Decode-Step mit KV Cache
- langer Prompt
- 135M und 360M
```

Erst wenn das stabil ist, darf `Runtime mode: warp` erscheinen.

## Erfolgskriterium

Bei SmolLM2-135M und SmolLM2-360M steht im Log:

```text
Runtime mode: warp
```

und die ersten Top-10-Logits stimmen gegen Java Reference / HF ausreichend
überein.

## Status der WARP-Pfade (Kontext)

```text
Qwen:    WARP produktiv brauchbar
T5:      WARP-Pfad vorhanden, MatMulNBitsKernel-Device-Bug gefixt
SmolLM2: wdmlpack + Reference Runtime korrekt, WARP Probe vorhanden,
         WARP Executor fehlt
```

## Verweise

- `directml-inference/src/main/java/com/aresstack/windirectml/inference/smollm2/SmolLM2WarpExecutor.java`
- `.../smollm2/SmolLM2WarpExecutorFactory.java`
- `.../smollm2/SmolLM2DirectMlWarpExecutor.java` (aktuelle Probe)
- `.../smollm2/SmolLM2UnsupportedWarpExecutor.java`
- `directml-workbench/src/main/java/com/aresstack/windirectml/workbench/runtime/SmolLM2WorkbenchRuntimeRunner.java`
- `docs/concept-decoder-extensions.md` (gemeinsame Decoder-Schicht)

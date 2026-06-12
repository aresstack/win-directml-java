# Konzept: Qwen ↔ decoderonly Adapterfähigkeit (Slice 4)

> **Status:** Adapter-Slice umgesetzt (2026-06-12). Qwen-Produktionsruntime unverändert und weiterhin Default.
> **Scope:** `directml-inference` / `qwen` ↔ `decoderonly`. Keine Produktionsmigration.

## 1. Ziel

Qwen auf die gemeinsamen `decoderonly`-Nahtstellen vorbereiten (Architekturhomogenisierung),
**ohne** den bestehenden `Qwen2Runtime`/`QwenGpuPipeline`-Pfad anzufassen oder umzuschalten.
Qwen bleibt die Produktionsreferenz.

## 2. Was in diesem Slice adaptiert wurde (produktionsneutral)

| Naht | Status | Artefakt |
|------|--------|----------|
| `DecoderOnlyConfig` | ✅ bereits erfüllt | `Qwen2Config implements DecoderOnlyConfig` (int/float-Komponenten + `headDim()`) |
| `DecoderOnlyStopTokenPolicy` | ✅ bereits erfüllt | `QwenStopTokenPolicy.asDecoderOnlyPolicy()` |
| `DecoderOnlyTokenizer` | ✅ neu angebunden | `QwenTokenizer implements DecoderOnlyTokenizer` (+ `isSpecialToken`) |
| `DecoderOnlyTokenSelector` | ✅ neu angebunden | `QwenTokenSelector` (kapselt Repetition-Penalty, delegiert an `DecoderOnlyGreedyTokenSelector`) |

Alle vier sind reine Adapter/Bindings. `Qwen2Runtime` und `QwenGpuPipeline` wurden **nicht** verändert;
der bestehende Decode-Loop ruft den Greedy-Selector weiterhin direkt auf. Kein Performance-Risiko.

## 3. Forward-Pass: warum Qwen (noch) nicht hinter `DecoderOnlyForwardPass` passt

Die Naht sieht heute so aus:

```java
public interface DecoderOnlyForwardPass {
    DecoderOnlyConfig config();
    DecoderOnlyWarpDecodeProfile decodeProfile();
    long lastCallLmHeadNanos();
    float[] logitsForLastToken(List<Integer> tokenIds, DecoderOnlyWarpKvCache kvCache); // ← der Knackpunkt
}
```

Sie wurde aus dem SmolLM2-Pfad extrahiert und trägt dessen **Cache-Besitzmodell**:

- Der KV-Cache (`DecoderOnlyWarpKvCache`, CPU-`float[]`) gehört der **Generation-Loop** und wird
  pro Schritt hineingereicht. Der Forward-Pass ist damit *stateless gegeben den Cache*.
- SmolLM2 hält Projektionen auf der GPU, **aber den KV-Cache auf der CPU** — deshalb passt das externe Cache-Modell.

Qwens Produktionsruntime ist fundamental anders aufgebaut:

- `Qwen2Runtime` ist **stateful**: `generateStreaming()` macht intern `prefill(int[])` → `decodeSingleToken(int)`
  mit **eigenem** KV-Cache (`kvCacheK/V`) und optional **GPU-resident** über
  `gpuPipeline.uploadKvCacheFromCpu(...)` / `isAttnGpuResidentEnabled()`.
- Der KV-Cache ist Teil der WARP-residenten Pipeline und INT4-/GPU-nah.

Daraus folgt der Konflikt:

```text
DecoderOnlyForwardPass.logitsForLastToken(tokens, DecoderOnlyWarpKvCache)
  → erzwingt einen loop-eigenen CPU-KV-Cache

Qwen besitzt seinen KV-Cache intern (CPU + optional GPU-resident)
```

Eine Kapselung in der heutigen Form hätte nur zwei Auswege, beide verboten/unsauber:

1. **Qwen auf den externen CPU-`DecoderOnlyWarpKvCache` umbiegen** → verliert die GPU-residente KV-Pipeline
   = Performance-Regression des Produktionspfads. ❌ (in Slice 4 ausdrücklich nicht erlaubt)
2. **Adapter ignoriert den hereingereichten Cache** und treibt Qwens internen Zustand → der `kvCache`-Parameter
   der Naht wird bedeutungslos, das Cache-Management/Sizing der Loop läuft leer, und zwei `generate()`-Läufe
   auf derselben Instanz bräuchten ein explizites Reset-Signal, das die Naht nicht hat. ❌ (Vertragsbruch)

**Ergebnis: „falls nein".** Der experimentelle Qwen-DecoderOnly-Forward-Pfad wird in Slice 4 **nicht** aktiviert.
Das Flag `-Dqwen.decoderonly.experimental=true` ist reserviert, aber bewusst noch nicht verdrahtet — es gäbe
nichts Sauberes, das es einschalten könnte, ohne (1) oder (2).

## 4. Welche Schnittstelle in `decoderonly` noch fehlt

Damit Qwen (und jede Familie mit GPU-residentem KV-Cache) sauber andocken kann, muss die Naht so evolvieren,
dass **der Forward-Pass seinen KV-Cache selbst besitzt**, statt ihn von der Loop zu bekommen. Skizze:

```java
// Vorschlag: Decode-Session, deren Zustand dem Forward-Pass gehört.
public interface DecoderOnlyDecodeSession extends AutoCloseable {
    float[] prefill(List<Integer> promptTokenIds); // logits für die letzte Prompt-Position
    float[] decode(int nextTokenId);               // logits für den nächsten Schritt
}

public interface DecoderOnlyForwardPass {
    DecoderOnlyConfig config();
    DecoderOnlyWarpDecodeProfile decodeProfile();
    long lastCallLmHeadNanos();
    DecoderOnlyDecodeSession newDecodeSession(int maxTokens); // Cache-Besitz beim Pass, nicht bei der Loop
}
```

- SmolLM2 implementiert das trivial (Session kapselt einen `DecoderOnlyWarpKvCache`).
- Qwen implementiert es ohne Verbiegen (Session = `resetCache()` + `prefill`/`decodeSingleToken`), die GPU-residente
  KV-Pipeline bleibt erhalten.
- Die `DecoderOnlyGenerationLoop` würde dann eine Session statt eines selbst erzeugten Caches treiben.

Das ist ein eigener, größerer Naht-Umbau (berührt SmolLM2 + die Loop) und gehört **nicht** in diesen Adapter-Slice.
Es ist die Voraussetzung für eine echte Qwen-Migration.

## 5. Nächster Schritt (separat zu entscheiden)

1. Naht-Evolution auf das Session-Modell (Abschnitt 4), SmolLM2 zuerst, verhaltensneutral verifiziert.
2. Danach experimenteller Qwen-`DecoderOnlyForwardPass` hinter `-Dqwen.decoderonly.experimental=true`,
   alte `Qwen2Runtime` bleibt Default und parallel, bis Tests **und** Performance passen.
3. Erst dann echte Qwen-Migration.

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

1. Naht-Evolution auf das Session-Modell (Abschnitt 4), SmolLM2 zuerst, verhaltensneutral verifiziert. ✅ (Slice 5)
2. Danach experimenteller Qwen-`DecoderOnlyForwardPass` hinter `-Dqwen.decoderonly.experimental=true`,
   alte `Qwen2Runtime` bleibt Default und parallel, bis Tests **und** Performance passen. ✅ (Slice 6 Adapter, Slice 7 E2E)
3. Erst dann echte Qwen-Migration. ⏳ (noch nicht freigegeben)

## 6. Slice 7 — E2E-Verifikationsergebnis

Harness: `QwenDecoderOnlySessionE2eTest` (gated, opt-in). Fährt den experimentellen Session-Pfad
(`QwenDecoderOnlyForwardPass` + `DecoderOnlyGenerationLoop` + `QwenTokenSelector`) gegen die produktive
`Qwen2Runtime` auf **demselben** geladenen Modell, mit identischem Prompt, `maxTokens`, Greedy + Penalty.

Lauf am 2026-06-12, lokal: Qwen2.5-Coder-0.5B (`model_q4f16.onnx`, INT4 MatMulNBits),
backend `auto` → DirectML/WARP, **24/24 Layer auf GPU**, GPU-residente Attention aktiv, `maxTokens=12`, `penalty=1.0`.

| Prompt | Produktion (Gold) | Experimentell (Session) | Ergebnis |
|--------|-------------------|-------------------------|----------|
| „Write a one-line Python function…" | 10 Tokens, `eos_token`, ids `[750,912,2877,11,293,1648,470,264,488,293]` | 10 streamed (gen=11), `eos_token`, gen-ids = dieselben 10 **+ `151645`** | ✅ token-für-token identisch |
| „Say hello." | 9 Tokens, `eos_token`, ids `[9707,0,2585,646,358,7789,498,3351,30]` | 9 streamed (gen=10), `eos_token`, **+ `151645`** | ✅ token-für-token identisch |

**Befund:**
- **Akzeptierte (gestreamte) Token-IDs sind byte-identisch** zur Produktion; **Finish-Reason identisch** (`eos_token`).
- Der **einzige** Unterschied ist die in Abschnitt „Known loop-contract difference" beschriebene Stop-Token-Behandlung:
  die geteilte `DecoderOnlyGenerationLoop` nimmt das terminierende Stop-Token (`151645` = `<|im_end|>`) in
  `generatedTokenIds` auf, die Produktions-`Qwen2Runtime` bricht davor ab. Das ist **erklärt und erwartet**, keine
  Numerik-Abweichung.
- Qwens GPU-residenter KV-Cache bleibt im Session-Pfad erhalten (die Session treibt die echte Pipeline über
  `QwenDecodeSteps`); kein CPU-Cache wird materialisiert.

**Laufzeit (nur grob gemessen, NICHT optimiert):** Produktion 617/198 ms, experimentell 219/184 ms für die beiden
Läufe. **Achtung Reihenfolge-Bias:** die Produktion lief jeweils zuerst (kalter JIT-/GPU-Warmup, erster Prefill),
die Session danach auf der bereits warmen Runtime — die Zahlen sind daher **kein** sauberer Performance-Vergleich,
nur eine Plausibilitätsprüfung, dass der Session-Pfad nicht pathologisch langsamer ist. Ein kontrollierter
Benchmark (Warmup, getrennte Läufe, mehr Tokens) ist bewusst einem späteren Slice vorbehalten.

**Heap-Hinweis:** Das dichte `model.onnx` (FP16→FP32 host) sprengt den Test-Heap (`maxHeapSize=2g`); der Lauf
nutzte daher `model_q4f16.onnx` (INT4). Für `model.onnx` muss der Test-Heap erhöht werden (`-Xmx`), sonst OOM beim
Gewichtsladen — reine Test-Infrastruktur, kein Adapter-Problem.

**Fazit:** Der experimentelle Qwen-Session-Pfad ist **token-äquivalent** zur Produktion. Eine echte Migration /
Default-Umschaltung ist damit *technisch* plausibel, bleibt aber bis zu einem kontrollierten Performance-Vergleich
und einer ausdrücklichen Freigabe **offen**.

## 7. Slice 8 — Result-Vertrag harmonisiert

Der in §6 dokumentierte Unterschied (geteilte Loop nahm das Stop-Token in `generatedTokenIds` auf) ist beseitigt.
Einheitlicher Vertrag jetzt überall (`DecoderOnlyGenerationLoop` **und** `SmolLM2ReferenceGenerationLoop`,
deckungsgleich mit der produktiven `Qwen2Runtime`):

- Ein terminierendes Stop-Token **beendet** die Generierung, wird aber **nicht** als generierter/gestreamter
  Nutz-Token geführt.
- `finishReason` bleibt `eos_token`.
- Das konkrete Stop-Token wird separat als `DecoderOnlyGenerationResult.finishTokenId` gemeldet (`-1`, wenn keines;
  d. h. bei `finishReason="length"`).

Folge: Im Qwen-Session-E2E stimmen `generatedTokenIds`, gestreamte Tokens **und** Finish-Reason ohne Sonderbeziehung
direkt mit der Produktion überein. SmolLM2 erhält denselben Vertrag (Stop-Token nicht mehr im Output) — reine
Result-Bookkeeping-Angleichung, keine Numerik-/Laufzeitänderung.

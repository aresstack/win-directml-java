# Runbook: Qwen legacy vs. decoder-only session — echte Läufe

> **Zweck:** Qwen-Läufe auf realen Maschinen erfassen und legacy gegen den decoder-only Session-Pfad vergleichen.
> **Status (ab Slice 11a):** **Default ist `decoder-only session`**; `legacy` ist opt-out via `-Dqwen.runtime=legacy`
> und bleibt vollständig erhalten. Dieser Runbook ändert nichts — nur Messen/Beobachten.
> **Verwandt:** `docs/concept-qwen-decoderonly-adapter.md` §10 (Default-Flip), §9 (Aktivierung), §8 (Benchmark).

## 0. Was verglichen wird

Derselbe Generierungseinstieg (`Qwen2Runtime.generateStreaming`), einmal über den bestehenden Produktionspfad,
einmal über den gemeinsamen `DecoderOnlyGenerationLoop` + `QwenDecoderOnlyDecodeSession`:

```text
(ohne Property)                     # Default — decoder-only session
-Dqwen.runtime=decoderonly-session  # explizit — decoder-only session
-Dqwen.runtime=legacy               # opt-out — bestehender Produktionspfad
```

Erwartung (bereits in CI/Harness verifiziert, hier auf echter Hardware bestätigen): **identische generated token IDs,
identischer Text, identischer Finish-Reason**, Performance neutral.

## 1. Voraussetzungen

- Qwen2.5-Coder-0.5B INT4-Modellverzeichnis (mit `config.json`, `tokenizer.json`, `model_q4f16.onnx` / `model.onnx`).
- Java 21+ (Workbench-Launcher findet es über `JAVA_HOME_21_X64` / `JAVA_HOME` / `PATH`).

## 2. Runtime-Pfad wählen — Stolperfalle Launcher

Der Workbench-**Launcher** (`directml-workbench-launcher-all.jar`) hardcodet die JVM-Flags der Workbench und reicht
nur Programm-Argumente nach `-jar` weiter. **Ein `-Dqwen.runtime=...`, das man dem Launcher gibt, erreicht die
Workbench-JVM NICHT.** Zwei funktionierende Wege:

**A) Workbench-Jar direkt starten (empfohlen, eindeutig):**

```powershell
java --enable-preview --enable-native-access=ALL-UNNAMED `
  -Dqwen.runtime=decoderonly-session `
  -jar directml-workbench-all.jar
```

(Ohne Property läuft seit Slice 11a der **Session-Pfad** als Default; für den alten Pfad explizit
`-Dqwen.runtime=legacy` setzen.)

> **Vector-Modul (seit Slice V3b optional):** Der Produktstart braucht `--add-modules=jdk.incubator.vector`
> **nicht** mehr (CPU-Math fällt sauber auf Skalar zurück). Für die FMA-SIMD-Beschleunigung der CPU-Pfade das Flag
> optional ergänzen: `java --enable-preview --enable-native-access=ALL-UNNAMED --add-modules=jdk.incubator.vector
> -Dqwen.runtime=decoderonly-session -jar directml-workbench-all.jar`. Compile/Test verwenden das Modul weiterhin.

**B) Über den Launcher mit `JAVA_TOOL_OPTIONS` (wird an die Kind-JVM vererbt):**

```powershell
$env:JAVA_TOOL_OPTIONS = "-Dqwen.runtime=decoderonly-session"
java -jar directml-workbench-launcher-all.jar
# danach: Remove-Item Env:\JAVA_TOOL_OPTIONS
```

Kontrolle im Konsolen-Log — der Einstieg loggt ehrlich:

```text
Runtime path: qwen legacy
Runtime path: qwen decoder-only session
```

> Hinweis Backend: `WARP` = D3D12-Software-Rasterizer; `AUTO` = bestе verfügbare Hardware, sonst Fallback. Der
> Runtime-**Pfad** (legacy/session) ist orthogonal zum Backend — beide Pfade nutzen dieselbe Qwen-Pipeline und deren
> (ggf. GPU-residenten) KV-Cache.

## 3. Vorgehen je Maschine

Für jede Maschine **beide** Pfade nacheinander mit **identischem Prompt und maxTokens** ausführen:

1. Workbench starten (Abschnitt 2), Qwen2.5-Coder-0.5B wählen, Backend wie geplant (WARP bzw. AUTO).
2. Prompt eingeben, generieren, Konsole + Workbench-Ausgabe ablesen (Runtime path, Output tokens, Finish reason,
   total ms, avg/token bzw. tok/s, Profilzeilen).
3. Werte in die Tabelle (Abschnitt 4) eintragen; generated token IDs / Textausgabe zwischen legacy und session
   vergleichen.
4. Pfad wechseln (Property umstellen, Workbench neu starten), wiederholen.

Zielmaschinen: **Hauptmaschine WARP**, **gehärtetes Intel-Notebook WARP**, optional **AUTO/GPU**.

## 4. Erfassungstabelle (pro Maschine ausfüllen)

| Feld | legacy | decoder-only session |
|------|--------|----------------------|
| Modell | | |
| Backend (WARP/AUTO) | | |
| Runtime path (aus Log) | qwen legacy | qwen decoder-only session |
| Prompt | | |
| maxTokens | | |
| generated token count | | |
| finish reason | | |
| total ms | | |
| tok/s | | |
| auffällige Logs | | |
| Textausgabe plausibel (ja/nein) | | |
| generated token IDs == legacy? | — | ja / nein |

(Block je Maschine kopieren: Hauptmaschine WARP / Intel-Notebook WARP / AUTO-GPU.)

## 5. Reproduzierbare Quervergleiche per Gradle-Harness (optional, ohne Workbench)

Diese gated Tests laufen nie in normaler CI und brauchen ein lokales Modell. Bevorzugt `model_q4f16.onnx` (INT4) —
das dichte `model.onnx` (FP16→FP32 host) sprengt den Test-Heap (`maxHeapSize=2g`); dafür müsste der Test-Heap erhöht
werden.

**Token-/Text-/Finish-Gleichheit der Routing-Umschaltung** (`generateStreaming` legacy vs. session):

```bash
./gradlew :directml-inference:test --tests "*.qwen.QwenSessionRoutingE2eTest" \
  -Dqwen.enable.experimental.runtime=true \
  -Dqwen.model.dir=<dir>/qwen2.5-coder-0.5b-directml-int4 \
  -Dqwen.harness.modelfile=model_q4f16.onnx --info
```

**Kontrollierter Performance-Vergleich** (Warmup + alternierende Messläufe, tok/s, prefill/decode):

```bash
./gradlew :directml-inference:test --tests "*.qwen.QwenDecoderOnlySessionBenchmarkTest" \
  -Dqwen.decoderonly.benchmark=true -Dqwen.decoderonly.experimental=true \
  -Dqwen.enable.experimental.runtime=true \
  -Dqwen.model.dir=<dir>/qwen2.5-coder-0.5b-directml-int4 \
  -Dqwen.harness.modelfile=model_q4f16.onnx -Dqwen.harness.maxtokens=64 --info
```

Lokale Referenzmessung (Hauptmaschine WARP, INT4, 64 Tokens, siehe Konzept-Doku §8): legacy ≈ 113.5 tok/s,
session ≈ 112.9 tok/s — gleichauf.

## 6. Bewertung & nächste Entscheidung

„Sauber" heißt: pro Maschine **session == legacy** bei generated token IDs **und** Finish-Reason, Textausgabe
plausibel, kein pathologischer tok/s-Einbruch beim Session-Pfad, keine auffälligen Fehlerlogs.

Erst danach entscheiden wir den nächsten Code-Slice:

```text
Slice 11a: Qwen decoder-only session wird Default
Slice 11b: legacy bleibt Default, session bleibt opt-in
Slice 11c: zuerst weitere Homogenisierung / Runtime-Diagnostik
```

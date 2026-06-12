# Vector-API-Hygiene — Slice V1: Bestandsaufnahme + Produktpfad-Abgrenzung

> **Status:** reine Analyse/Doku. Nichts entfernt, kein Runtime-/WARP-/AUTO-Verhalten geändert,
> `--add-modules=jdk.incubator.vector` bleibt überall gesetzt.
> **Frage:** Braucht der produktive WARP/AUTO-Start `jdk.incubator.vector` noch?

## 1. Ist-Stand: wer nutzt die Vector-API?

**Genau drei Klassen importieren `jdk.incubator.vector`** (FloatVector/VectorOperators/VectorSpecies):

| Klasse | Rolle | Vector-API |
|--------|-------|-----------|
| `qwen/SimdOps` | SIMD dot/axpy-Helper (Qwen) | direkt |
| `phi3/SimdOps` | SIMD dot/axpy-Helper (Phi3) — identisch zu qwen/SimdOps außer Paket | direkt |
| `decoderonly/DecoderOnlyReferenceDenseOps` | Reference-Dense-Math (dot/multiplyRows) | direkt |

**Alle drei sind graceful-fallback-sicher:** im statischen Initialisierer
`try { FloatVector.SPECIES_PREFERRED } catch (Throwable) { ENABLED=false }`, und **jede** Vector-Methode hat einen
Skalar-Pfad (`if (!ENABLED) return scalar…`). D. h. die Klassen **laden und funktionieren auch ohne** das Modul
(dann skalar). `Qwen2Runtime` importiert die Vector-API **nicht** direkt; es nutzt nur `SimdOps` und loggt explizit
„SIMD path DISABLED — jdk.incubator.vector module not loaded … continue" — der Code ist also bewusst für den
Betrieb ohne Modul ausgelegt.

## 2. Produktpfad (WARP/AUTO) vs. Reference/Diagnostic/Test

Die rechenintensiven Matmuls des Produktpfads laufen **nativ** über `MatMulNBitsKernel`/`WarpDenseProjection`
(D3D12) — **keine Vector-API**. Tatsächliche Vector-API-**Ausführung** im Produktpfad:

| Pfad | Vector-API im Produktbetrieb? | Detail |
|------|------------------------------|--------|
| **Qwen** (Default `decoder-only session` **und** `legacy`) | **ja** | CPU-Attention `SimdOps.dot/axpy` in `Qwen2Runtime.decodeSingleToken`/`prefill` |
| **SmolLM2 WARP** | **nein** (nur Klassenladen) | Prefill-SwiGLU = **skalares** `DecoderOnlyReferenceDenseOps.gatedSiluMultiply`; Attention = eigener skalarer `dotOffset`; LM-Head auf WARP. Die Vector-Methoden (`dot`/`multiplyRows`) laufen nur im Reference-/Dev-LM-Head-Modus |
| **T5 WARP** | **nein** | nutzt `T5WarpLinearProjection` (nativ) + `T5ReferenceMath` (kein Vector-Import) |
| Reference/Diagnostic | ja | `DecoderOnlyReferenceDenseProjection`, `SmolLM2ReferenceForwardPass`/`ReferenceDenseOps`, `SmolLM2DenseTensor` (Reference-LM-Head), Phi3-CPU |

**Fazit:** Der einzige produktive WARP/AUTO-Verbraucher von ausgeführtem Vector-API-Code ist **Qwens CPU-Attention**
(degradierbar zu skalar). SmolLM2-WARP und T5-WARP führen im Produktpfad **keinen** Vector-API-Code aus.

## 3. Könnte der Produktstart ohne das Modul laufen?

- **Qwen Default `decoder-only session`:** ja — `SimdOps.enabled()=false` → skalare Attention (langsamer, aber
  funktional; Projektionen liegen ohnehin auf WARP).
- **SmolLM2 WARP:** ja — kein ausgeführter Vector-Code; `DecoderOnlyReferenceDenseOps` lädt via Fallback.
- **T5 WARP:** ja — kein Vector-Import im T5-WARP-Pfad.

> ⚠️ **KORREKTUR durch Slice V2 (empirisch):** Diese V1-Annahme ist **falsch**. Siehe §9 — ohne das Modul wirft der
> Start `NoClassDefFoundError: jdk/incubator/vector/Vector`. Der `try/catch`-Fallback greift nicht, weil bereits das
> **Linken** der Klasse die Vector-Typen auflöst (Felder/Signaturen), bevor der Static-Init läuft. Der Produktstart
> **kann derzeit NICHT** ohne das Modul laufen, solange die Klassen Vector-Typen direkt referenzieren.

~~**Es gibt keine harte Modulabhängigkeit zum Starten.**~~ (durch V2 widerlegt — es ist eine harte Lade-Abhängigkeit.)
Das Modul ist Performance-Enabler für CPU-Math **und** — wie V2 zeigt — eine harte Klassen-Lade-Abhängigkeit der drei
Vector-Klassen.

Wichtige Unterscheidung:
- **Runtime ohne Modul = möglich** (Fallback existiert). Betrifft Launcher/`JavaExec`/Test-JVM-Flags.
- **Compile ohne Modul = NICHT möglich**, solange die 3 Klassen `import jdk.incubator.vector` haben → `--add-modules`
  muss in `compileJava`/`compileTestJava` bleiben. „Modul ganz weg" bräuchte Isolierung der Vector-Nutzung hinter
  einen optional/reflektiv geladenen Baustein (größerer Umbau, eigener späterer Slice).

## 4. Wo wird das Flag heute gesetzt?

| Datei | Kontext | Art |
|-------|---------|-----|
| `build.gradle` (compileJava/compileTestJava `compilerArgs`) | **Compile** | erforderlich (Imports) |
| `build.gradle` (`tasks.withType(Test)` `jvmArgs`) | Test-JVM | runtime |
| `build.gradle` (`tasks.withType(JavaExec)` `jvmArgs`) | dev run | runtime |
| `directml-workbench-launcher/.../WorkbenchLauncher.java` | Kind-JVM-Kommando der Workbench | runtime (Produkt) |
| `directml-workbench/packaging/launcher.ps1` | Packaging-Start | runtime (Produkt) |
| `docs/runbook-qwen-decoderonly-session.md`, `docs/qwen-hybrid-backend.md`, `docs/decoderonly-homogenization.md` | Doku/Befehle | Doku |

## 5. Welche Tests bräche ein Entfernen?

- **CI/Tests behalten das Flag** (build.gradle Test-jvmArgs) → kein Test bricht durch ein späteres Entfernen aus dem
  **Produkt-Launcher**.
- Würde man das Flag aus der **Test-JVM** entfernen: die Klassen laden weiter (Fallback), Ergebnisse wären
  skalar statt FMA-SIMD → minimale Float-Rundungsunterschiede. Toleranz-Tests (z. B. WARP-vs-Reference) blieben grün;
  exakte-Gleichheits-Tests auf reinen CPU-Vector-Pfaden könnten abweichen. Empfehlung: Test-Flag vorerst behalten.

## 6. Risikoanalyse

| Maßnahme | Risiko | Effekt |
|----------|--------|--------|
| Flag aus Produkt-Launcher entfernen | niedrig | Qwen-CPU-Attention → skalar (modest; Attention ist nicht der Flaschenhals, Projektionen auf WARP). SmolLM2/T5 WARP unberührt. Incubator-Warnung verschwindet. |
| Flag aus Compile entfernen | hoch / nicht möglich ohne Umbau | Compile bricht (Imports). Bräuchte Vector-Isolierung. |
| Flag aus Test-JVM entfernen | mittel | mögliche Rundungs-/Exakt-Test-Abweichungen; CPU-Tests langsamer. |
| Status quo (behalten) | keins | nur die kosmetische Incubator-Warnung bleibt. |

## 7. Empfehlung & Slice-V2-Vorschlag

**Entscheidungsvorschlag: Vector-Modul im Produktstart OPTIONAL machen** (nicht hart entfernen, nicht ganz behalten):
Der Produkt-Launcher soll ohne das Modul lauffähig sein (Fallback ist da), das Flag aber als optionalen
Performance-Schalter anbieten. Compile + Test behalten das Flag.

**Slice V2 (Vorschlag):**
1. **Empirischer No-Modul-Smoke-Test zuerst** (Gate): Qwen-`decoder-only session` + SmolLM2-WARP einmal **ohne**
   `--add-modules=jdk.incubator.vector` starten/generieren und Token-Identität + `SimdOps.enabled()==false`
   bestätigen (auf der WARP-Maschine; analog Runbook). Beweist die Lauffähigkeit vor jeder Entfernung.
2. Danach das Flag aus den **Produkt-Launchern** entfernen/optional machen: `WorkbenchLauncher.java`,
   `directml-workbench/packaging/launcher.ps1` (und ggf. `JavaExec`-jvmArgs in `build.gradle`).
3. `compileJava`/`compileTestJava` + Test-JVM-Flag **behalten**.
4. Doku/Runbooks aktualisieren (Flag als optional kennzeichnen).

**Dateien, die in V2 angepasst würden:**
- `directml-workbench-launcher/src/main/java/.../WorkbenchLauncher.java` (Kind-JVM-Kommando)
- `directml-workbench/packaging/launcher.ps1`
- ggf. `build.gradle` (nur `JavaExec`-jvmArgs; **nicht** compile/test)
- `docs/runbook-qwen-decoderonly-session.md`, `docs/qwen-hybrid-backend.md`, `docs/decoderonly-homogenization.md`

**Späterer optionaler Slice V3 (groß):** Vector-API hinter einen optional/reflektiv geladenen Baustein isolieren,
sodass auch der **Compile** das Modul nicht mehr braucht und die Vector-API wirklich nur Reference/Diagnostic ist.

## 8. Antworten auf die V1-Leitfragen (kompakt)

- **Welche Klassen nutzen die Vector-API wirklich?** `qwen/SimdOps`, `phi3/SimdOps`, `decoderonly/DecoderOnlyReferenceDenseOps`.
- **Ist der produktive WARP/AUTO-Pfad betroffen?** Nur Qwens CPU-Attention (degradierbar). SmolLM2/T5 WARP: nein.
- **Workbench ohne `--add-modules` grundsätzlich möglich?** Ja (graceful Fallback); Compile braucht das Flag weiter.
- **Risiken?** Modest Qwen-CPU-Attention-Perf; CPU-Test-Rundung; Compile bleibt modulabhängig bis zur Isolierung.
- **Sinnvoller V2?** Empirischer No-Modul-Smoke-Test → Flag in Produkt-Launchern optional/entfernen → Compile/Test behalten.
  → **Durch V2 revidiert:** Der Smoke-Test ist **rot** (NoClassDefFoundError). Flag bleibt überall gesetzt; das Ziel
  „Produktstart ohne Modul" braucht zuerst die Vector-Isolierung (ehemals V3, jetzt zwingende Voraussetzung). Siehe §9.

## 9. Slice V2 — empirisches Ergebnis: Produktstart OHNE Modul ist (noch) NICHT möglich

**Smoke-Test (auf der WARP-Maschine):** Test-JVM **ohne** `--add-modules=jdk.incubator.vector` (Compile-Flag blieb).
Ausgeführt: `SmolLM2NativeWarpExecutorTest` (synthetische Gewichte, echtes WARP) und `QwenDecoderOnlySessionE2eTest`
(Qwen2.5-Coder-0.5B INT4, WARP).

**Ergebnis: ROT.** Alle 4 SmolLM2-Native-Fälle + der Qwen-E2E scheitern mit:

```text
java.lang.NoClassDefFoundError: jdk/incubator/vector/Vector
```

**Ursache:** Der graceful `try { FloatVector.SPECIES_PREFERRED } catch (Throwable)`-Fallback greift **nicht** für den
Fall „Modul fehlt". Schon das **Laden/Linken** von `SimdOps` / `DecoderOnlyReferenceDenseOps` muss die Vector-Typen
(Feld `VectorSpecies<Float>`, Methoden-Signaturen mit `FloatVector`) auflösen → `NoClassDefFoundError` **vor** dem
Static-Init → nicht abfangbar durch den Static-Init der Klasse. Der `catch` schützt nur den Laufzeit-Fall
„Modul da, aber CPU/SPECIES nicht verfügbar".

**Betroffen (laden eine Vector-Klasse, daher harter Fehler ohne Modul):**
- **SmolLM2 WARP** — Prefill ruft `DecoderOnlyReferenceDenseOps.gatedSiluMultiply` → lädt `DecoderOnlyReferenceDenseOps`.
- **Qwen** (Default session **und** legacy) — `Qwen2Runtime` → `SimdOps`.
- alle Reference-/Diagnostic-Pfade.
- **T5 WARP** importiert keine Vector-API (V1) → würde vermutlich ohne Modul starten, ist aber hier nicht der relevante
  Pfad und wurde **nicht** separat getestet (kein automatischer T5-WARP-Test; bräuchte Modell + GUI).

**Konsequenz / Entscheidung:**
- **Flag bleibt überall gesetzt** (Produkt-Launcher `WorkbenchLauncher.java` / `packaging/launcher.ps1`, Compile, Test) —
  **nichts entfernt**, weil ein Entfernen das Produkt mit `NoClassDefFoundError` brechen würde.
- Das V2-Ziel „Produktstart ohne Modul" ist im erlaubten V2-Rahmen **nicht erreichbar** (es bräuchte das Entfernen der
  direkten Vector-Referenzen aus immer-geladenen Klassen — d. h. die V3-Isolierung, die in V2 ausgeschlossen war).
- **V3 ist damit von „optional" zu „erforderlich" hochgestuft**, falls der modullose Produktstart wirklich gewünscht
  ist.

### Korrigierte Empfehlung (für V3)
Vector-API-Nutzung hinter eine **isolierte, optional/reflektiv geladene** Implementierung legen, sodass die
immer-geladenen Klassen (`SimdOps`-Aufrufstellen, `DecoderOnlyReferenceDenseOps`) **keine** `jdk.incubator.vector`-Typen
mehr in Feldern/Signaturen referenzieren:
- z. B. ein Interface `DenseSimd` mit zwei Impls: `ScalarDenseSimd` (immer ladbar) und `VectorDenseSimd`
  (lädt Vector-API), per `try { Class.forName(...) } catch` ausgewählt — so scheitert nur das Laden der **Vector-Impl**
  (abfangbar), nicht der Aufrufer.
- Danach erst Flag aus Produkt-Launchern entfernen + erneuter Smoke-Test (jetzt grün erwartet).
- Compile/Test behalten das Flag, solange die Vector-Impl existiert.

**Status V2:** abgeschlossen als **Befund/Korrektur** — keine Launcher-/Flag-Änderung (bewusst), Doku korrigiert.
Compile/Test laufen unverändert mit Modul; alle Suiten + `SmolLM2NativeWarpExecutorTest` grün (mit Flag).

## 10. Slice V3a — Vector-API isoliert: immer-geladene Klassen sind jetzt ohne Modul ladbar

**Ziel umgesetzt:** Die in V2 als hart modulabhängig identifizierten Klassen referenzieren keine
`jdk.incubator.vector`-Typen mehr in Feldern/Signaturen. Die Vector-Nutzung liegt in **einer** isolierten Klasse,
die **reflektiv** geladen wird; fehlt das Modul, greift sauber der Skalar-Fallback.

**Neue, neutrale SIMD-Schicht (`inference/simd`):**

| Klasse | Vector-Import? | Rolle |
|--------|----------------|-------|
| `simd/FloatMathOps` | nein | neutrales Interface: `enabled()`, `dot()`, `axpy()` |
| `simd/ScalarFloatMathOps` | nein | immer ladbarer Skalar-Impl (`enabled()==false`) |
| `simd/SimdMath` | nein | reflektiver Loader: `Class.forName(VectorFloatMathOps)` + `catch (Throwable)` → sonst Skalar |
| `simd/vector/VectorFloatMathOps` | **ja (einzige)** | FMA-Dot/Broadcast-AXPY; public no-arg ctor; `enabled = lanes>=4` |

**Umgebaute Fassaden (jetzt ohne Vector-Import, delegieren an `SimdMath.provider()`):**
- `qwen/SimdOps` — package-private Fassade, stabil; delegiert `enabled/dot/axpy`.
- `phi3/SimdOps` — identische Fassade (Phi3 nutzt nur `dot`).
- `decoderonly/DecoderOnlyReferenceDenseOps` — `OPS = SimdMath.provider()`; Parallel-/IntStream-Logik,
  `multiplyRows`, `gatedSiluMultiply`, `silu` unverändert.

**Verifikation:** `grep "import jdk.incubator.vector"` über `src/main` → **nur** `VectorFloatMathOps.java`.

**Empirischer No-Modul-Lauf (WARP-Maschine, Test-JVM ohne `--add-modules`, Compile-Flag blieb):**

| Test | V2 (vor Isolierung) | V3a (nach Isolierung) |
|------|---------------------|------------------------|
| `simd/NoVectorModuleLoadabilityTest` (neu) | — | **grün** (1/1; Fassaden ladbar, `provider().enabled()==false`, Skalar korrekt) |
| `SmolLM2NativeWarpExecutorTest` | **rot** (4/4 `NoClassDefFoundError`) | **grün** (4/4) |
| `QwenDecoderOnlySessionE2eTest` | **rot** (`NoClassDefFoundError`) | **grün** (1/1) |

→ Der `NoClassDefFoundError: jdk/incubator/vector/Vector` beim Laden der immer-geladenen Klassen ist **weg**.
Qwen-`decoder-only session` (Default) und SmolLM2-WARP starten jetzt **ohne** das Modul (skalarer CPU-Pfad).

**Mit Modul (CI-Default) unverändert grün:** generation/model/decoderonly/smollm2/qwen/t5/phi3/simd +
`SmolLM2NativeWarpExecutorTest`. Numerik bei vorhandenem Modul identisch (gleicher FMA-/Broadcast-Code in
`VectorFloatMathOps`).

**Neue Test-Schalter:** `-Dwindirectml.test.noVectorModule=true` aktiviert `NoVectorModuleLoadabilityTest`
(sonst `assumeTrue`-Skip). `FloatMathOpsTest` läuft immer (geräte-frei, mit/ohne Modul).

**Bewusst NICHT geändert in V3a:** Produkt-Launcher-Flags (`WorkbenchLauncher.java`, `packaging/launcher.ps1`),
Compile-Flag, Test-jvmArgs-Flag — alle bleiben gesetzt. Das Entfernen der **Produkt-Launcher**-Flags ist Slice V3b.

**Bereit für V3b:** Da die immer-geladenen Klassen jetzt modullos ladbar sind und der modullose Start empirisch grün
ist, kann V3b die `--add-modules`-Flags aus den Produkt-Launchern entfernen/optional machen (Compile/Test behalten sie,
solange `VectorFloatMathOps` existiert).

**Status V3a:** abgeschlossen — Vector-API isoliert, immer-geladene Klassen modullos ladbar, No-Modul-Smoke grün,
mit-Modul-Suiten grün, Produkt-Launcher unverändert.

## 11. Slice V3b — Produkt-Launcher ohne Vector-Modul

**Ziel umgesetzt:** Der produktive Workbench-/Launcher-Start läuft **ohne** `--add-modules=jdk.incubator.vector`.
Compile und Test behalten das Modul.

**Geänderte produktive Launcher/Skripte (Flag entfernt):**
- `directml-workbench-launcher/.../WorkbenchLauncher.java` — Kind-JVM-Kommando jetzt über die testbare
  `buildCommand(javaExe, jarPath, args)`, **ohne** `--add-modules=jdk.incubator.vector`.
- `directml-workbench/packaging/launcher.ps1` — `--add-modules=jdk.incubator.vector` aus der `java`-Zeile entfernt
  (mit Kommentar, wie man SIMD optional wieder aktiviert).

**Bewusst behalten (mit Modul):**
- `build.gradle` `compileJava`/`compileTestJava` `compilerArgs` (Compile braucht das Modul, solange
  `VectorFloatMathOps` existiert).
- `build.gradle` `tasks.withType(Test)` `jvmArgs` (CI/Tests laufen mit Modul → SIMD-Pfad bleibt getestet).
- `build.gradle` `tasks.withType(JavaExec)` `jvmArgs` (Dev-Run via `gradle run`, kein ausgeliefertes Startskript) —
  bewusst unverändert gelassen.

**Begleitende Meldungs-Korrektur:** `Qwen2Runtime`'s „SIMD DISABLED"-Warnung sagte fälschlich, der Launcher füge das
Flag automatisch hinzu. Sie sagt jetzt wahrheitsgemäß: scalar fallback ist der Default des Produkt-Launchers; SIMD ist
optional via `--add-modules=jdk.incubator.vector` (JAVA_TOOL_OPTIONS oder direktes `java -jar`).

**Opt-in für SIMD im Produkt:** `--add-modules=jdk.incubator.vector` über `JAVA_TOOL_OPTIONS` setzen oder direkt an
`java -jar directml-workbench-all.jar` übergeben. Dann ist `SimdOps.enabled()==true` und die CPU-Math nutzt FMA-SIMD.

**No-Modul-Smoke (Test-JVM ohne `--add-modules`, Compile-Flag blieb) — alle grün:**

| Pfad | Test | Ergebnis |
|------|------|----------|
| Loadability | `simd/NoVectorModuleLoadabilityTest` | 1/1 |
| SmolLM2 WARP | `SmolLM2NativeWarpExecutorTest` | 4/4 |
| Qwen decoder-only session WARP | `QwenDecoderOnlySessionE2eTest` | 1/1 |
| MiniLM Embedding WARP | `DirectMlMiniLmEmbeddingReferenceTest` | 5/5 |
| Reranker WARP (inkl. batch=5) | `RerankerRealModelReferenceTest` | 6/6 |

T5-WARP hat keinen automatischen Smoke-Test (Modell + GUI nötig), importiert die Vector-API aber nicht (V1/§2) und
startet daher ebenfalls ohne Modul. Die GUI selbst (`java -jar directml-workbench-all.jar`) konnte in der
Test-/Sandbox-Umgebung nicht headless gefahren werden; die obigen programmatischen WARP-Smokes decken dieselben
Lade-/Generierungs-/Embedding-/Reranking-Pfade ab.

**Regressionstest:** `WorkbenchLauncherTest.buildCommand_doesNotRequireIncubatorVectorModule` schützt davor, dass das
Flag wieder in den Produkt-Launcher zurückwandert.

**V4 (optional, später):** Vollständiges Entfernen von `--add-modules=jdk.incubator.vector` aus **Compile** (und damit
aus Test). Das erfordert, `VectorFloatMathOps` aus dem normalen `compileJava` herauszuhalten (z. B. eigener
Source-Set/Modul-Build mit dem Incubator-Flag, separat kompiliert und reflektiv geladen). **Nur nice-to-have** — der
produktive Start ist mit V3b bereits modullos; V4 würde nur die Incubator-Warnung beim Bauen beseitigen.

**Status V3b:** abgeschlossen — Produkt-Launcher modullos, Compile/Test behalten das Modul, No-Modul-Smokes grün,
mit-Modul-Suiten grün.

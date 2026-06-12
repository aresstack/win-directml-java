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

**Es gibt keine harte Modulabhängigkeit zum Starten.** Das Modul ist ein **Performance-Enabler** für CPU-Math
(Qwen-Attention, Reference-Pfade), kein Start-Erfordernis.

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

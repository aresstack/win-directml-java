# Gemma 3 270M-it — Evaluations-Ergebnis (MODEL-GEMMA-4)

> Echte Ausführung der externen Probe aus
> [`model-candidate-gemma3-270m-external-probe.md`](model-candidate-gemma3-270m-external-probe.md)
> via `scripts/probe-gemma3-270m-it.py`. **Keine Runtime-/Produktcode-Änderung**, nur Probe + Doku.
> **Keine erfundenen Zahlen** — alle Werte aus dem realen Lauf (gemessen, HF-Login-Token vorhanden).
>
> Vorgeschichte: MODEL-GEMMA-3 lief mit HTTP 401 (gated, ohne Token). Mit akzeptierter Lizenz +
> gespeichertem HF-Login lud `google/gemma-3-270m-it` jetzt erfolgreich und generierte.

## Umgebung (gemessen)

| Feld | Wert |
|------|------|
| Datum | 2026-06-14 |
| OS / Python | Windows 11 Pro N · Python 3.12.10 |
| transformers / torch | 5.11.0 / 2.12.0+cpu (CPU) |
| HF-Zugang | Login-Token aus HF-Cache (Gemma-Lizenz akzeptiert) — Gemma lädt |
| Lauf | `--limit 28 --max-new-tokens 160`, Vergleich `--compare smollm2` |
| Prompts | **28** (Gruppe 1:7 · 2:5 · 3:6 · 4:5 · 5:5) |

## Prompt-Anzahl geklärt (15 vs 28)

Im früheren Lauf waren nur ~15 Prompts sichtbar, weil das **Skript** out-of-the-box nur **15**
eingebettete Prompts hatte (kein Parser der `.md`; `--limit` war korrekt, aber durch
`len(PROMPTS)=15` gedeckelt). Die `.md` enthält real **28** Prompts. Das Skript wurde auf die vollen
**28 maschinenlesbaren** Prompts erweitert (1:1 aus der `.md`, Snippets inline) — **nicht künstlich
aufgefüllt**, es sind dieselben 28. Dieser Lauf hat alle 28 ausgeführt.

Außerdem behoben: `--out` legt das Zielverzeichnis jetzt an (`Path(out).parent.mkdir(...)`), der
frühere `FileNotFoundError: build/gemma-probe/...jsonl` tritt nicht mehr auf; `.jsonl` schreibt JSONL.

## Latenz (gemessen, CPU)

| Modell | Prompts | ø tok/s |
|--------|--------:|--------:|
| gemma-3-270m-it | 28 | **22.0** |
| SmolLM2-360M-Instruct | 28 | 14.2 |

Gemma ist auf dieser CPU **schneller** als SmolLM2-360M (~22 vs ~14 tok/s; deckt sich mit der
Beobachtung ~20–26 tok/s) und liefert meist kurze Antworten.

## Qualität pro Gruppe (gemma-3-270m-it, real)

| Gruppe | Urteil | Beobachtung |
|--------|:------:|------|
| 1 Code erklären | **0** | EN/DE-Erklärungen flüssig & kurz, aber Natural/ADABAS nur **oberflächlich** (z. B. „reads from a local file", DISPLAY-Felder ungenau) |
| 2 Pseudocode/Java-Skizze | **-** | P2.1 **echo't** den Input statt Pseudocode; Java-Skizze ist ein **Platzhalter** mit „replace this"-Kommentaren (halluziniert); SQL `WHERE NAME = 'EMPLOYEES'` **fachlich falsch** |
| 3 JSON-Extraktion | **0/-** | JSON ist **syntaktisch immer valide**, aber die **Werte sind Platzhalter** (`"view":"..."`, `"fields":["..."]`) statt extrahiert — strukturtreu, aber inhaltsleer |
| 4 Tech-Zusammenfassung | **+** | **deutsche** Zusammenfassung sauber & korrekt; 2-Bullet ok |
| 5 DE/EN-Umformulierung | **+** | einfache Übersetzungen DE↔EN brauchbar (kleine Holprigkeiten) |

**JSON valide?** Syntaktisch **ja** (alle 6 Group-3-Antworten parsen), inhaltlich **nein** (Platzhalter).

## Vergleich zur SmolLM2-360M-Baseline

| Aspekt | gemma-3-270m-it | SmolLM2-360M |
|--------|------------------|--------------|
| Latenz (CPU) | **~22 tok/s** | ~14 tok/s |
| EN-Erklärung | flüssig, oberflächlich | flüssig, oberflächlich |
| Deutsch (Summary/Translate) | **brauchbar** | schwächer (echo't dt. Prompts öfter) |
| JSON-Struktur | **immer valide**, aber Platzhalter | mal echte Werte (P3.1 korrekt extrahiert!), mal halluziniert/Markdown-Fence |
| Pseudocode/SQL/Java-Skizze | echo/halluziniert/falsch | ebenfalls schwach |
| Antwortlänge | meist kurz | meist kurz |

Kurz: Gemma ist **schneller** und bei **Deutsch + valider JSON-Struktur** disziplinierter; SmolLM2
trifft bei **JSON-Werte-Extraktion** vereinzelt besser (P3.1), ist dafür inkonsistenter. Beide sind
out-of-the-box **schwach** bei Code-Transformation (Pseudocode/SQL/Java-Skizze) und
Natural/ADABAS-Tiefe.

## Entscheidung

**Verdikt: WAIT — aber positiv (Fine-Tuning plausibel).**

**Begründung:** `gemma-3-270m-it` ist auf CPU **schnell genug** (~22 tok/s) und zeigt out-of-the-box
brauchbare **deutsche Zusammenfassung/Übersetzung** und eine **konsequent valide JSON-Struktur** —
genau die Eigenschaften, die Google als „Fine-Tune-Basis / task-specific" bewirbt. Es ist aber
**kein** brauchbarer Spezialassistent von der Stange: Pseudocode-Aufgaben echoen den Input,
Java-Skizzen halluzinieren, SQL ist fachlich falsch, JSON-Extraktion liefert Platzhalter statt Werte,
Natural/ADABAS wird nur oberflächlich verstanden. Gegenüber SmolLM2-360M ist es **nicht klar besser
in der Aufgaben-Qualität**, aber **schneller** und **strukturierter** — was es als **Fine-Tuning-Basis
für strukturierte Extraktion / kurze Erklärungen** interessant macht.

**Folge:** Noch **kein** GO für eine eigene `gemma/`-Runtime-Familie auf Basis der Out-of-the-box-
Qualität. Wenn Gemma weiterverfolgt wird, dann als **Fine-Tuning-Kandidat** (LoRA/SFT auf unsere
engen Aufgaben: Code-Erklärung, Natural/ADABAS, JSON-Extraktion), gefolgt von einer erneuten Probe.
Der Runtime-Aufwand (eigene `GemmaWarpForwardPass`, siehe `model-candidate-gemma3-270m.md` §2) bleibt
unverändert groß und sollte erst nach einer überzeugenden (ggf. fine-getunten) Qualitätsprobe
investiert werden.

## Reproduzieren

```bash
# Gemma-Lizenz akzeptieren + HF-Login (huggingface-cli login oder HF_TOKEN), dann:
python scripts/probe-gemma3-270m-it.py --compare smollm2 \
    --out build/gemma-probe/compare.jsonl --limit 28 --max-new-tokens 160
```

(JSONL-Rohausgaben liegen unter `build/` und sind nicht versioniert.)

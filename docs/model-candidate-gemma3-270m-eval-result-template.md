# Gemma 3 270M-it — Evaluations-Ergebnis (Template, MODEL-GEMMA-2)

> **Status:** leere Bewertungsvorlage. Nach der externen Probe (siehe
> [`model-candidate-gemma3-270m-external-probe.md`](model-candidate-gemma3-270m-external-probe.md))
> manuell ausfüllen. **Keine Zahlen erfinden** — leer lassen, bis tatsächlich gesampelt.

## Umgebung

| Feld | Wert |
|------|------|
| Datum / Tester | |
| Maschine / CPU | |
| RAM | |
| Python / transformers / torch | |
| HF-Token gesetzt + Lizenz akzeptiert | ☐ ja ☐ nein |
| `--max-new-tokens` | |
| Vergleichsmodelle geladen | ☐ SmolLM2-360M ☐ Qwen2.5-Coder-0.5B ☐ keine |

## Bewertungsskala

`++` sehr gut · `+` ok · `0` grenzwertig · `-` schwach · `n/a` nicht anwendbar.
Halluzination: `ja`/`nein` (hier ist `nein` gut). JSON valide: `ja`/`nein`/`n/a`.

## Matrix pro Prompt-Gruppe (gemma-3-270m-it)

| Gruppe | folgt Instruktion | fachlich grob richtig | halluziniert | JSON valide | kurze Antwort möglich | Deutsch brauchbar | Code-/Natural-Semantik erkennbar | besser als Smol360 | CPU-Speed plausibel | Notiz |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|------|
| 1 Code erklären | | | | n/a | | | | | | |
| 2 Pseudocode/Java-Skizze | | | | n/a | | | | | | |
| 3 JSON-Extraktion | | | | | | n/a | | | | |
| 4 Technische Zusammenfassung | | | | n/a | | | n/a | | | |
| 5 DE/EN-Umformulierung | | | | n/a | | | n/a | | | |

## Vergleich (optional, falls Modelle geladen)

| Aufgabe | gemma-3-270m-it | SmolLM2-360M | Qwen2.5-Coder-0.5B |
|---------|:-:|:-:|:-:|
| 1 Code erklären | | | |
| 2 Pseudocode/Skizze | | | |
| 3 JSON-Extraktion | | | |
| 4 Tech-Zusammenfassung | | | |
| 5 DE/EN-Umformulierung | | | |
| grobe Tokens/s (CPU) | | | |

## Auffälligkeiten (frei)

- _(z. B. JSON mit Markdown-Fences statt purem JSON, abgeschnittene Antworten, deutsche Grammatikfehler,
  erfundene Feld-/Methodennamen, Wiederholungen, Latenz-Eindruck.)_

## Entscheidung: GO / WAIT / NO-GO

Entscheidungsregeln (aus `model-candidate-gemma3-270m.md` §7):

- **GO** → eigene `gemma/`-Familie implementieren (Geschwister von qwen/smollm2, gemeinsame
  GPU-Primitives + `model/`-Ladepfad, eigene Gemma-Forward-Logik) **nur**, wenn
  `gemma-3-270m-it` für unsere Zielaufgaben **spürbar besser als SmolLM2-360M** ist **und**
  mindestens Qwen-nahe Latenz plausibel scheint.
- **WAIT** → wenn das Bild gemischt ist (z. B. gut bei Extraktion/Umformulierung, schwach bei Code)
  oder die Probe-Datenlage zu dünn ist.
- **NO-GO** → wenn die Qualität ungetuned nicht reicht (wahrscheinlich für Code) oder der
  256k-FP32-LM-Head die Latenz auf unserem WARP-Profil unattraktiv macht.

**Verdikt:** ☐ GO ☐ WAIT ☐ NO-GO

**Begründung (3–5 Sätze):**

> _…_

**Nächster Schritt bei GO:** gated-fähiger Downloader-Slice (HF-Token-Header + 401/403-Lizenzhinweis),
dann Tokenizer-Kompatibilitätsprüfung, dann `GemmaWarpForwardPass`-Design (jeweils eigene Slices).

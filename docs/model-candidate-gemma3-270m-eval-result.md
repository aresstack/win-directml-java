# Gemma 3 270M-it — Evaluations-Ergebnis (MODEL-GEMMA-3)

> Tatsächliche Ausführung der externen Probe aus
> [`model-candidate-gemma3-270m-external-probe.md`](model-candidate-gemma3-270m-external-probe.md)
> via `scripts/probe-gemma3-270m-it.py`. **Keine Runtime-/Produktcode-Änderung** in
> `win-directml-java`. **Keine erfundenen Zahlen** — Gemma konnte nicht gesampelt werden (siehe unten),
> daher keine Gemma-Scores.

## Umgebung (gemessen)

| Feld | Wert |
|------|------|
| Datum / Tester | 2026-06-14 / automated probe run |
| OS | Windows 11 Pro N (10.0.22631) |
| Python | 3.12.10 |
| transformers / torch | 5.11.0 / 2.12.0+cpu (CPU) |
| HF erreichbar | ja |
| HF-Token gesetzt + Gemma-Lizenz akzeptiert | **nein** (kein `HF_TOKEN`, Lizenz nicht akzeptiert) |
| `--max-new-tokens` | 64 |
| Vergleichsmodelle geladen | SmolLM2-360M-Instruct ✓ (heruntergeladen) · Qwen2.5-Coder-0.5B ✗ (nicht ausgeführt, siehe unten) |

## Hauptergebnis: Gemma ist gated → nicht sampelbar

`google/gemma-3-270m-it` ließ sich in dieser Umgebung **nicht** laden:

```
SKIPPED: could not load google/gemma-3-270m-it: You are trying to access a gated repo.
401 Client Error. Cannot access gated repo for url
https://huggingface.co/google/gemma-3-270m-it/resolve/main/config.json
Access to model google/gemma-3-270m-it is restricted. ... Please log in.
```

**Ursache (sauber dokumentiert, wie gefordert):**
- Gemma-**Lizenz** auf der HF-Modellseite **nicht akzeptiert**.
- **`HF_TOKEN` fehlt** (unauthenticated request).
- HF antwortet daher mit **401** (gated).

→ Es gibt **keine** Gemma-Antworten zum Bewerten. Es wurde **nicht** am Produkt-Downloader gearbeitet
(wie vorgegeben). Der Skript-Pfad funktioniert korrekt (sauberes Skip + Hinweis statt Crash).

**Entsperren (manuell, außerhalb dieses Repos):** Lizenz auf
<https://huggingface.co/google/gemma-3-270m-it> akzeptieren, `HF_TOKEN` setzen, dann
`python scripts/probe-gemma3-270m-it.py --compare smollm2 qwen --out result.md` erneut laufen lassen und
die SmolLM2-Spalte unten um die Gemma-Spalte ergänzen.

## Baseline (real gemessen): SmolLM2-360M-Instruct

Der Probe-Harness wurde so an einem **echten** Modell validiert; gleichzeitig ist das die Messlatte,
die Gemma „spürbar" überbieten müsste. Latenz: **~9–16 tok/s** auf dieser CPU. Qualität gemischt:

| Gruppe | folgt Instruktion | fachlich grob richtig | halluziniert | JSON valide | kurz | Deutsch brauchbar | Code-/Natural-Semantik | Beobachtung (SmolLM2-360M) |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|------|
| 1 Code erklären | + | + | nein | n/a | + | **-** | + | EN-Erklärung von Natural **und** Java korrekt & knapp; **deutscher** Prompt echo'te das Snippet statt zu antworten |
| 2 Pseudocode/Skizze | 0 | 0 | teils | n/a | 0 | n/a | 0 | Java-Skelett brauchbar (echte Methode + SQL); Natural→Pseudocode echo'te Input; SQL-Rewrite lieferte `CREATE TABLE` statt `SELECT ... ORDER BY` |
| 3 JSON-Extraktion | 0 | 0 | teils | **teils** | + | n/a | + | `{view,accessType,fields}` **valide & korrekt**; „calls" lieferte falschen Inhalt + Markdown-Fence; query/documents lieferte gar kein JSON |
| 4 Tech-Zusammenfassung | **-** | + | nein | n/a | **-** | **-** | n/a | EN-„Summary" gab den Satz unverändert zurück (nicht gekürzt); DE echo'te Prompt; 2-Bullet teils ok |
| 5 DE/EN-Umformulierung | 0 | 0 | nein | n/a | + | 0 | n/a | gemischt (kurze Umformulierungen teils ok, deutsche Prompts schwächer) |

Stärkste Disziplin von SmolLM2-360M out-of-the-box: **knappe englische Erklärung** + **eine** saubere
JSON-Extraktion. Schwächen: **deutsche** Instruktionen und „nur-X"-Transformationen (Pseudocode/SQL,
JSON-only) werden oft mit Echo des Inputs beantwortet.

## Qwen2.5-Coder-0.5B-Instruct

**Nicht ausgeführt.** Optionaler Vergleich; ein ~1 GB-Download, der ohne Gemma-Baseline die
Gemma-Entscheidung nicht beeinflusst. Bei Bedarf:
`python scripts/probe-gemma3-270m-it.py --compare qwen`.

## Bewertung der Leitfragen

- **Folgt Gemma der Instruktion?** unbekannt (nicht sampelbar — gated).
- **Bleibt die Antwort kurz?** unbekannt.
- **Versteht es Natural/ADABAS grob?** unbekannt.
- **Liefert es gültiges JSON?** unbekannt.
- **Besser als Smol360?** unbekannt — die Smol360-Messlatte ist dokumentiert (knappe EN-Erklärung +
  1× saubere JSON-Extraktion; schwach bei DE + reinen Transformationen).
- **Viel schlechter als Qwen?** unbekannt (Qwen nicht ausgeführt).
- **CPU-Latenz realistisch?** für die Klasse plausibel (Smol360 ~9–16 tok/s CPU gemessen); Gemmas
  256k-FP32-LM-Head bleibt laut Feasibility-Doc der dominante Posten — nicht hier verifiziert.

## Entscheidung

**Verdikt: ☑ WAIT (access-blocked).**

**Begründung:** Die externe Probe wurde real ausgeführt und der Harness an SmolLM2-360M validiert, aber
`gemma-3-270m-it` konnte mangels akzeptierter Lizenz + `HF_TOKEN` **nicht geladen** werden (401 gated).
Ohne echte Gemma-Antworten ist **kein GO/NO-GO** seriös begründbar (keine erfundenen Zahlen). Die
Empfehlung aus MODEL-GEMMA-1 (WAIT, erst billige externe Probe) bleibt bestehen; sie ist jetzt **nur
noch durch HF-Zugang blockiert**, nicht durch fehlende Vorbereitung.

**Nächster Schritt (außerhalb dieses Repos, kein Produkt-Downloader-Umbau):** HF-Lizenz akzeptieren +
`HF_TOKEN` setzen, Probe erneut mit `--compare smollm2 qwen` laufen lassen, diese Datei um die
Gemma-Spalte ergänzen und dann GO/WAIT/NO-GO final setzen.

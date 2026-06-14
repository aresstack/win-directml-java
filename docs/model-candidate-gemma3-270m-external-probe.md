# Gemma 3 270M-it — Externe Qualitätsprobe (MODEL-GEMMA-2)

> **Status:** reine Probe-Anleitung. Läuft **außerhalb** unserer Java/WARP-Runtime über
> HF/`transformers`. Keine `.wdmlpack`-Erzeugung, keine Workbench, keine Downloader-Änderung,
> keine Kernel. Zweck: günstig prüfen, ob `google/gemma-3-270m-it` für unsere engen CPU-Zielaufgaben
> taugt, **bevor** Runtime-Aufwand investiert wird (siehe `model-candidate-gemma3-270m.md`, WAIT).

## Was geprüft wird

- Modell: `google/gemma-3-270m-it`
- Prompts: [`model-candidate-gemma3-270m-eval-prompts.md`](model-candidate-gemma3-270m-eval-prompts.md)
  (28 Prompts, 5 Gruppen, aus vorhandenen Projektdaten).
- Optionaler Vergleich, **falls lokal vorhanden**: `HuggingFaceTB/SmolLM2-360M-Instruct`,
  `Qwen/Qwen2.5-Coder-0.5B-Instruct`.
- Ergebnis in [`model-candidate-gemma3-270m-eval-result-template.md`](model-candidate-gemma3-270m-eval-result-template.md).

## Gated-Repo: HF-Lizenz + Token (wichtig)

`google/gemma-3-270m-it` ist **gated**. Ohne akzeptierte Lizenz + Token liefert HF **401/403**.

1. Auf <https://huggingface.co/google/gemma-3-270m-it> einloggen und die **Gemma-Lizenz akzeptieren**.
2. Access-Token erzeugen (<https://huggingface.co/settings/tokens>, Rolle „read").
3. Token setzen:
   ```bash
   export HF_TOKEN=hf_xxx            # Linux/macOS
   $env:HF_TOKEN = "hf_xxx"          # Windows PowerShell
   ```
   (oder `huggingface-cli login`).

> Dies betrifft **nur** die externe Probe. Der Produkt-Downloader wird in diesem Slice **nicht**
> umgebaut; ein gated-fähiger Downloader (Authorization-Header + 401/403-Lizenzhinweis) wäre ein
> späterer, separater Slice (vgl. `model-candidate-gemma3-270m.md` §1/§7).

## Voraussetzungen

```bash
python -m venv .venv && . .venv/bin/activate     # optional
pip install "transformers>=4.45" torch --upgrade
# CPU genügt; 270M passt locker in RAM. Keine GPU nötig.
```

## Ausführen

```bash
# Nur Gemma:
python scripts/probe-gemma3-270m-it.py

# Mit Vergleich gegen lokal vorhandene Modelle (werden übersprungen, wenn nicht ladbar):
python scripts/probe-gemma3-270m-it.py --compare smollm2 qwen

# Eigene Token-/Längen-Limits:
python scripts/probe-gemma3-270m-it.py --max-new-tokens 128 --limit 12
```

Das Skript:
- lädt `google/gemma-3-270m-it` via `transformers` (CPU),
- wendet das **Chat-Template** an (`<start_of_turn>user … <start_of_turn>model`),
- sampelt eine repräsentative Teilmenge der Prompts (alle 5 Gruppen) und gibt Antworten + grobe
  Latenz (Tokens/s) aus,
- schreibt optional eine Markdown-Tabelle (`--out gemma-probe-output.md`).

Die **vollständigen** 28 Prompts stehen in der Eval-Prompts-Datei; das Skript fährt eine kompakte,
repräsentative Auswahl, damit die Probe schnell bleibt. Für eine vollständige Durchsicht die Prompts
manuell aus der `.md` übernehmen.

## Bewertung

Antworten manuell gegen die Kriterien in
[`model-candidate-gemma3-270m-eval-result-template.md`](model-candidate-gemma3-270m-eval-result-template.md)
einordnen und am Ende **GO / WAIT / NO-GO** ableiten:

- **GO** nur, wenn `gemma-3-270m-it` für unsere Aufgaben **spürbar besser** als SmolLM2-360M ist und
  Qwen-nahe Latenz plausibel scheint → dann separater `gemma/`-Familien-Implementierungs-Slice.
- **WAIT/NO-GO**, wenn Qualität ungetuned nicht reicht oder der 256k-FP32-LM-Head die Latenz auf
  unserem WARP-Profil unattraktiv macht.

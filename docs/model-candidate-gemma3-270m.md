# Model candidate — Gemma 3 270M (MODEL-GEMMA-1 feasibility)

> **Status:** reine Analyse/Doku. Keine Runtime-/Downloader-/UI-Änderung, kein Code. Prüft, ob
> `google/gemma-3-270m-it` als Qualitätskandidat zwischen SmolLM2-135M/360M und Qwen2.5-Coder-0.5B taugt und ob er in
> unsere `decoderonly`-Schicht passt.
>
> **Hinweis zur Quelle:** Diese Analyse wurde ohne Netzzugriff erstellt. Die *Architektur-Deltas* (Norm-Stil,
> Aktivierung, Attention-Muster, Tokenizer-Familie) sind aus der bekannten Gemma-3-Architektur sicher; die *exakten
> Zahlen* (Layer-Zahl, intermediate_size, sliding_window etc.) sind als „aus `config.json` bestätigen" markiert und
> sollten beim ersten Download gegen das echte Repo verifiziert werden.

## 1. Modellvarianten

| Repo | Rolle | Gated | Erste Wahl |
|------|-------|-------|-----------|
| `google/gemma-3-270m-it` | **Instruction-tuned** (chat_template, `<start_of_turn>`) | **ja** (Gemma-Lizenz + HF-Token) | **ja — Priorität** |
| `google/gemma-3-270m` | Pretrained/Base | ja | nur Architekturvergleich, nicht als Workbench-Default |
| `google/gemma-3-270m-it-qat-q4_0-unquantized` | QAT (INT4-aware, FP-Gewichte) | ja | **später** analysieren, nicht jetzt |
| `google/gemma-3-270m-qat-q4_0-unquantized` | QAT base | ja | später |
| `*-gguf` Varianten | für llama.cpp/Ollama | — | **nicht** für unsere Java/WARP-Runtime |

Dateien (laut Repo-Vorschau) für `gemma-3-270m-it`: `config.json`, `model.safetensors` (~536 MB),
`tokenizer.json` (~33 MB), `tokenizer.model` (~4.7 MB SentencePiece), `tokenizer_config.json`,
`special_tokens_map.json`, `chat_template.jinja`, `generation_config.json`, `added_tokens.json`, `README.md`.
→ SafeTensors ✓, tokenizer.json ✓, config.json ✓, chat_template ✓.

**Gated-Repo-Folge für einen späteren Downloader-Slice (NICHT Teil dieser Analyse):** HF liefert für gated Repos
ohne akzeptierte Lizenz/Token **401/403**. Unser `ModelDownloader` macht heute ein nacktes `GET` ohne
`Authorization`-Header → ein Pflicht-Datei-Download bräche mit einem opaken „HTTP 401/403". Ein Gemma-Download-Slice
müsste (a) einen optionalen `Authorization: Bearer <HF_TOKEN>`-Header unterstützen und (b) bei 401/403 klar auf
„Gemma-Lizenz auf HF akzeptieren + Token setzen" hinweisen. Required-Set für den ersten Workbench-Eintrag (wenn es so
weit kommt): `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
`chat_template.jinja`.

## 2. Architektur — Gemma 3 vs. unsere `decoderonly`-Schicht

Decoder-only: **ja**. Aber Gemma 3 weicht in mehreren *mathematisch relevanten* Punkten vom Llama/Qwen-Schema ab,
auf das unsere `decoderonly`-Schicht heute hart festgelegt ist.

Bekannte/zu bestätigende Dimensionen (270M):

| Feld | Wert | Sicherheit |
|------|------|-----------|
| vocab_size | **262.144 (256k)** | sicher |
| hidden_size | **640** | sicher |
| head_dim | **256** (entkoppelt von hidden/heads!) | sicher |
| num_attention_heads | **4** | hoch |
| num_key_value_heads | **1** (GQA) | hoch |
| num_hidden_layers | ~? | **aus config.json bestätigen** |
| intermediate_size | ~2048 (GeGLU) | aus config.json bestätigen |
| max_position_embeddings / Kontext | **32k** | sicher |
| rms_norm_eps | 1e-6 | hoch |

**Gemma-spezifische Abweichungen (das sind die Show-Stopper für direkte Wiederverwendung):**

1. **Zero-centered RMSNorm:** Gemma wendet `x * (1 + weight)` an. Unsere `DecoderOnlyMath.rmsNorm` macht
   `x * rms * weight` (Standard). → andere Norm-Berechnung.
2. **GeGLU mit GELU-tanh:** Gemma-MLP = `down(gelu_tanh(gate) * up)`. Unsere decoderonly-Schicht ist **hart SwiGLU/SiLU**
   (`DecoderOnlyWarpSwiGluKernel` HLSL `silu(gate)*up`, `DecoderOnlyReferenceDenseOps.silu`, `DecoderOnlyMath` SiLU).
   → andere Aktivierung, eigener Kernel/Reference-Pfad.
3. **Sandwich-Norms (4 pro Layer):** `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`,
   `post_feedforward_layernorm`. Llama/Qwen haben 2. → mehr Norm-Tensoren + andere Residual-Struktur.
4. **QK-Norm:** RMSNorm auf Q und K (`q_norm`/`k_norm`) vor RoPE. In decoderonly **nicht vorhanden**.
5. **Embedding-Scaling:** Eingabe-Embeddings × `sqrt(hidden_size)`. In decoderonly **nicht vorhanden**.
6. **Alternierende local/global Attention:** Muster ~5 local : 1 global; local-Layer mit **Sliding-Window** (≈512,
   bestätigen). Unsere Schicht macht **überall volle causale Attention**, kein Window.
7. **Duale RoPE-Theta:** local-Layer (~10.000) vs global-Layer (~1.000.000). `DecoderOnlyConfig` hat nur **eine**
   `ropeTheta`.
8. **head_dim ≠ hidden/heads:** 4×256=1024 ≠ 640 → q_proj liefert 1024, dann o_proj 1024→640. Unsere Schicht nimmt
   meist `headDim = hidden/heads` an (prüfen, ob `headDim()` schon frei wählbar ist — Interface hat `headDim()`, gut,
   aber die Pfade müssen es konsequent nutzen).
9. **Tied embeddings:** lm_head = embed_tokens (kein separater `lm_head.weight`). Das **unterstützt** decoderonly bereits
   (`DecoderOnlyEmbeddingTable` tied/untied, Forward-Pass „tied/untied LM head").
10. **Kein Logit-Softcapping** (Gemma 3 hat Gemma-2's attn/final softcap entfernt; via QK-Norm ersetzt) — neutral für uns.

**Verdikt Architektur:** Der bestehende `DecoderOnlyWarpForwardPass` ist **nicht direkt** nutzbar und ein *kleiner*
Adapter reicht **nicht**. Gemma braucht eine **eigene `GemmaWarpForwardPass`** (bzw. eine deutlich erweiterte,
konfigurierbare decoderonly-Schicht), weil Norm-Stil, Aktivierung, Norm-Anzahl/Residual-Struktur, QK-Norm,
Embedding-Scaling, local/global+Sliding-Window und duale RoPE-Theta **mathematische** Unterschiede sind, nicht nur
andere Tensor-Namen. **Wiederverwendbar** sind die *unteren Bausteine*: `WarpDenseProjection`/`MatMulNBitsKernel`
(GEMM/GEMV inkl. der neuen ByteBuffer-Naht), KV-Cache-Grundstruktur (aber Window-Logik fehlt), Token-Selektion/Greedy,
Generation-Loop, Stop-Token-Policy, `RuntimeModelPackage`/`RuntimeTensorCatalog`-Ladepfad. Realistischer Schnitt:
**neue `gemma/`-Familie als Geschwister von `qwen`/`smollm2`, die dieselben GPU-Primitives + den `model/`-Ladepfad
nutzt, aber eigene Layer-Forward-Logik mitbringt** (analog dazu, dass T5 eine eigene seq2seq-Schicht hat statt in
decoderonly gezwängt zu werden).

## 3. Tensorlayout (HF-`transformers` Gemma3-Namen, zu bestätigen)

```
model.embed_tokens.weight                                  # [vocab, hidden], tied -> lm_head
model.layers.{i}.input_layernorm.weight                    # RMSNorm (zero-centered)
model.layers.{i}.self_attn.q_proj.weight                   # [heads*head_dim, hidden] = [1024, 640]
model.layers.{i}.self_attn.k_proj.weight                   # [kv_heads*head_dim, hidden]
model.layers.{i}.self_attn.v_proj.weight
model.layers.{i}.self_attn.o_proj.weight                   # [hidden, heads*head_dim]
model.layers.{i}.self_attn.q_norm.weight                   # QK-Norm (Gemma-spezifisch)
model.layers.{i}.self_attn.k_norm.weight
model.layers.{i}.post_attention_layernorm.weight
model.layers.{i}.pre_feedforward_layernorm.weight          # Sandwich-Norm (Gemma-spezifisch)
model.layers.{i}.mlp.gate_proj.weight
model.layers.{i}.mlp.up_proj.weight
model.layers.{i}.mlp.down_proj.weight
model.layers.{i}.post_feedforward_layernorm.weight         # Sandwich-Norm (Gemma-spezifisch)
model.norm.weight                                          # final RMSNorm
# kein lm_head.weight (tied)
```

Mapping-Einschätzung: q/k/v/o und gate/up/down passen strukturell auf unsere fused/non-fused Dense-Projektionen.
**Neu/zusätzlich:** `q_norm`/`k_norm`, `pre/post_feedforward_layernorm`, das zero-centered Norm-Detail und das
Aktivierungs-Detail. → **eigene `GemmaWarpForwardPass`** nötig; ein reiner `GemmaAdapter` über
`DecoderOnlyWarpForwardPass` reicht nicht.

## 4. Tokenizer

- **SentencePiece-BPE, 256k** (Gemma-Tokenizer). Repo liefert `tokenizer.json` (HF-Fast-Format) **und**
  `tokenizer.model` (SentencePiece-Proto).
- **Special tokens:** `<bos>`, `<eos>`, `<pad>`, sowie Chat-Tokens `<start_of_turn>` / `<end_of_turn>`.
- **Chat-Template (it):** `<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n`.
- **Java-Tokenizer-Pfad:** Risiko **mittel**. SmolLM2/Qwen nutzen bereits `tokenizer.json`-BPE in Java. Gemma's
  `tokenizer.json` ist eine andere BPE-Konfiguration (Metaspace `▁`, Byte-Fallback, großes Merge-Set, Gemma-Normalizer).
  Zu prüfen, ob unser bestehender `tokenizer.json`-Parser Gemma's Normalizer/Pre-Tokenizer + Byte-Fallback korrekt
  abbildet. **Kein neuer nativer Dependency nötig**, *wenn* der vorhandene tokenizer.json-Pfad Gemma versteht — sonst
  eine begrenzte Tokenizer-Erweiterung (kein SentencePiece-Native-Binding, da tokenizer.json vorhanden).

## 5. Performance-Schätzung Intel-WARP

Parameter-Aufteilung (270M): ~**170M Embedding/Vocab** (256k×640) + ~**100M Transformer-Body**.

- **Body** (hidden 640, wenige Layer): deutlich **leichter** als Qwen-0.5B (hidden 896, ~24 Layer, ~360M Nicht-Embedding)
  und auch leichter als SmolLM2-360M.
- **LM-Head/Vocab-Projektion:** 256k×640 ≈ **164M MAC/Token** — der dominante Posten. Vergleich:
  - Qwen-0.5B LM-Head ≈ 151k×896 ≈ 135M MAC/Token, aber **INT4** auf GPU.
  - SmolLM2-360M LM-Head ≈ 49k×960 ≈ 47M MAC/Token (viel kleiner — 49k-Vokabular!).
- **Folge:** Gemma's 256k-LM-Head ist auf **FP32-WARP** der teure Teil und liegt grob in Qwen-Größenordnung, während
  Gemma's Body viel günstiger ist.

| Vergleich | grobe Erwartung pro Token auf Intel-WARP |
|-----------|------------------------------------------|
| Gemma-270M vs SmolLM2-135M | Gemma **langsamer** (135M hat 49k-Vocab + winzigen Body) |
| Gemma-270M vs SmolLM2-360M | **unklar/eher langsamer** — Gemma's 256k-LM-Head (~164M) übersteigt SmolLM2's gesamten Token-Aufwand deutlich; Gemma-Body kleiner, aber LM-Head dominiert |
| Gemma-270M vs Qwen-0.5B INT4 | **knapp/eventuell schneller** — viel leichterer Body, aber FP32-LM-Head (kein INT4-Pfad in unserer Runtime) frisst den Vorteil teilweise auf |

**Wichtige Dämpfer:** (a) Unsere Runtime hat **keinen INT4-Pfad für Gemma** — der QAT-q4_0-unquantized ist trotz des
Namens FP, d. h. wir würden **FP32** fahren → 256k×640 FP32-Gewicht = ~656 MB allein für embed/lm_head; die neue
heap-light ByteBuffer-Upload-Naht (H2) wäre hier besonders wertvoll. (b) WARP profitiert **nicht** automatisch von
INT4/QAT. (c) Sliding-Window/local-global spart bei langem Kontext etwas Attention-Aufwand, ist bei kurzen
Workbench-Prompts aber kaum relevant. **Fazit Speed:** Gemma-270M ist **nicht zuverlässig „deutlich schneller als
Qwen-0.5B"** — der 256k-LM-Head nivelliert den Body-Vorteil; ein klarer Speed-Gewinn ist **nicht garantiert**.

## 6. Qualitätsrisiko

`gemma-3-270m-it` ist von Google explizit als **Fine-Tune-Basis / task-specific** positioniert (instruction following,
strukturierte Ausgaben, Klassifikation/Extraktion), **nicht** als starker genereller Chat-Assistent. Erwartung
out-of-the-box:

| Aufgabe | Erwartung 270M-it (ungetuned) |
|---------|-------------------------------|
| kurze Zusammenfassung | mäßig–ok |
| technische Erklärung | mäßig |
| Code-Erklärung kleiner Snippets | **schwach–mäßig** (kein Code-Tuning wie Qwen-Coder) |
| Übersetzung DE/EN | mäßig |
| strukturierte Textausgabe (JSON/Listen) | **relativ stärkste** Disziplin |

Einordnung: vermutlich **besser instruktionsfähig als SmolLM2-135M**, aber **nicht klar besser als SmolLM2-360M** für
generisches Chatten/Erklären — und **deutlich unter Qwen-0.5B** bei Code. Es ist primär eine **Fine-Tune-Basis**, kein
starker Allzweck-Workbench-Assistent.

## 7. Empfehlung

**WAIT.**

Begründung:
- **Architektur:** implementierbar, aber **nicht trivial** — eine eigene `GemmaWarpForwardPass` mit zero-centered
  RMSNorm, GeGLU/GELU-tanh, Sandwich-Norms, QK-Norm, Embedding-Scaling, local/global+Sliding-Window und dualer
  RoPE-Theta. Nennenswerter Implementierungs- und Test-Aufwand (mehrere Slices).
- **Speed:** kein garantierter Gewinn gegenüber Qwen-0.5B (256k-FP32-LM-Head dominiert); ggf. langsamer als
  SmolLM2-360M.
- **Qualität:** unsicher und laut Positionierung eher Fine-Tune-Basis.

Bevor wir den Runtime-Aufwand investieren, zuerst eine **billige externe Qualitätsprobe**: `gemma-3-270m-it` via
HF/`transformers` (oder Ollama-GGUF) **außerhalb** unserer Runtime mit echten Workbench-Prompts (DE/EN-Summary,
Code-Erklärung, strukturierte Ausgabe) sampeln und gegen SmolLM2-360M und Qwen-0.5B vergleichen.

- **GO** (→ neue `gemma/`-Familie implementieren, Geschwister von qwen/smollm2, gemeinsame GPU-Primitives + `model/`-
  Ladepfad, eigene Gemma-Forward-Logik) **nur**, wenn die externe Probe zeigt, dass `gemma-3-270m-it` für unsere
  Aufgaben **spürbar besser** als SmolLM2-360M ist und mindestens Qwen-nahe Latenz erreichbar scheint.
- **NO-GO**, wenn die Probe zeigt, dass die Qualität ungetuned nicht reicht (wahrscheinlich für Code) oder die
  256k-LM-Head-Latenz Gemma auf WARP unattraktiv macht.

**Konkrete nächste Schritte (separate Slices, falls gewünscht):**
1. Externe Prompt-Probe `gemma-3-270m-it` (kein Code in diesem Repo).
2. Falls vielversprechend: gated-fähiger Downloader-Slice (HF-Token-Header + 401/403-Lizenzhinweis) — **erst** dann.
3. Tokenizer-Kompatibilitätsprüfung `tokenizer.json` gegen unseren Java-BPE-Pfad.
4. `GemmaWarpForwardPass`-Design (eigene Slice-Serie).

## 8. Antworten auf die Leitfragen (kompakt)

- **Passt Gemma in die bestehende decoderonly-Schicht?** Nein, nicht direkt; eigene `GemmaWarpForwardPass` nötig
  (zero-centered RMSNorm, GeGLU/GELU-tanh, 4 Sandwich-Norms, QK-Norm, Embedding-Scaling, local/global+Sliding-Window,
  duale RoPE-Theta, head_dim≠hidden/heads). Untere GPU-/Lade-Bausteine sind wiederverwendbar; tied embeddings werden
  bereits unterstützt.
- **Auf Intel-WARP realistisch schneller als Qwen-0.5B?** Nicht zuverlässig — leichterer Body, aber 256k-FP32-LM-Head
  dominiert; bestenfalls knapp, kein sicherer Gewinn. Kein INT4-Pfad für Gemma in unserer Runtime.
- **Empfehlung:** WAIT — erst externe Qualitätsprobe, dann GO/NO-GO.

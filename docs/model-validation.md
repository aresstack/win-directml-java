# Model validation

The model validator is a CPU-only pre-flight check for local model directories. It does not load weights into DirectML
and does not require a GPU.

It checks:

- required files such as `config.json` and `tokenizer.json`
- one supported weight file: `model.safetensors` or `pytorch_model.bin`
- selected `config.json` shape fields: `hidden_size`, `num_hidden_layers`, `num_attention_heads`
- tokenizer family via `tokenizer.json` `model.type`
- support status for known model families

## Sidecar CLI

MiniLM:

```powershell
java --enable-preview --enable-native-access=ALL-UNNAMED `
  -Dembed.model=minilm `
  -Dminilm.modelDir=model/all-MiniLM-L6-v2 `
  -jar directml-sidecar-all.jar `
  --validate-models
```

E5 base-v2:

```powershell
java --enable-preview --enable-native-access=ALL-UNNAMED `
  -Dembed.model=e5 `
  -De5.model=base-v2 `
  -De5.modelDir=model/e5-base-v2 `
  -jar directml-sidecar-all.jar `
  --validate-models
```

Optional reranker validation:

```powershell
java --enable-preview --enable-native-access=ALL-UNNAMED `
  -Dembed.model=minilm `
  -Dminilm.modelDir=model/all-MiniLM-L6-v2 `
  -Drerank.modelDir=model/cross-encoder-ms-marco-MiniLM-L-6-v2 `
  -jar directml-sidecar-all.jar `
  --validate-models
```

Exit codes:

| Code | Meaning                                      |
|-----:|----------------------------------------------|
|  `0` | All validation reports are OK.               |
|  `4` | At least one validation report has an error. |

## Workbench

The Config & Control tab has a **Validate Models** button. It runs the same Java-8 validator directly in the Workbench
process, so it works without starting the sidecar and without DirectML.

## E5 status

The WordPiece E5 variants currently validated as runnable are:

- `small-v2`
- `base-v2`
- `large-v2`

`base-sts-en-de` is intentionally reported as not ready. The upstream checkpoint is XLM-R/SentencePiece-shaped, while
the current runtime supports WordPiece E5 variants. The validator treats this as an error so users do not accidentally
mount an incompatible model and debug it as a DirectML problem.

## Example output

```text
== MiniLM all-MiniLM-L6-v2 ==  (model/all-MiniLM-L6-v2)
[OK] file present: config.json
[OK] file present: tokenizer.json
[OK] one weight file present: [model.safetensors, pytorch_model.bin]
[OK] hidden_size = 384
[OK] num_hidden_layers = 6
[OK] num_attention_heads = 12
[OK] tokenizer type = WordPiece
Result: OK (0 error(s), 0 warning(s))

model validation reports: 1, failed: 0
```

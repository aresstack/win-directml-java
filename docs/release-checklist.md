# Release checklist

Manual pre-release verification. **No tag, publish, Maven Central upload or GitHub Release happens without an
explicit human go-ahead.** Two distinct artifact tracks:

- **Published Maven library** (embeddings / reranking): `directml-config`, `directml-windows-bindings`,
  `directml-encoder`, `directml-runtime`. This is the only thing published to Maven Central.
- **Experimental Workbench app** (text generation): `directml-workbench` + `directml-workbench-launcher` fat jars.
  Built artifacts, **not** Maven-published.

## 1. Standard regression (must be green)

```bash
JAVA_HOME=".../zulu-21" GRADLE_USER_HOME=.chatgpt/gradle-home ./gradlew --no-daemon --offline \
  :directml-config:test :directml-inference:test :directml-encoder:test :directml-workbench:test
```

## 2. Gated real-model smokes (optional, opt-in — need local models under `model/`)

```bash
# T5 real-model correctness certs (the four curated models)
./gradlew :directml-inference:test --tests '*T5MixedRuntimeCorrectnessCertTest' \
  -Dt5.correctness.cert=true -Dt5.realModel=true
# Phi-3-mini ONNX->wdmlpack compile + heap-light decode smoke
./gradlew :directml-inference:test --tests '*Phi3WdmlPackCompilerTest' \
  -Dphi3.compile.realModel=true -Dphi3.realModel=true
```

These stay opt-in; the standard run must stay light. Downloaded `model/...` artifacts are git-ignored.

## 3. Workbench app build + manual smoke

```bash
./gradlew :directml-workbench:shadowJar :directml-workbench-launcher:shadowJar
# -> directml-workbench/build/libs/directml-workbench-all.jar
# -> directml-workbench-launcher/build/libs/directml-workbench-launcher-all.jar
./gradlew :directml-workbench-launcher:run     # launches the Workbench
```

Manual Download → Convert → Summarize smoke (per `SUPPORTED_MODELS.md` §3.2): download a model (e.g. Phi-3-mini or a
T5 model), Convert it to its `.wdmlpack` in the Download tab, then run it in the Summarizer with Backend WARP/AUTO/CPU.

## 4. Artifact hygiene (fat jar must NOT contain model data)

```bash
unzip -l directml-workbench/build/libs/directml-workbench-all.jar | \
  grep -iE '\.wdmlpack|\.safetensors|\.onnx|model/.*(phi|t5|flan|codet5|gemma|qwen|smollm)'   # expect: no matches
```

The fat jar must contain only code + dependencies — no model weights, no local downloads, no temp Phi/T5 artifacts.
(`model/...` is git-ignored, so it is never committed and never bundled.)

## 5. Maven artifact boundary

- Confirm `release.ps1` / the publish config still publishes only the four core library modules (§"Releases / Maven
  Central" in `README.md`), **not** `directml-inference` / `directml-workbench*`.
- Confirm the README + `docs/release-notes-next.md` do not present text generation as a Maven-Central stability
  promise, and do not suggest an ONNX Runtime / Python product path.

## 6. Docs

- `README.md`, `SUPPORTED_MODELS.md`, `docs/workbench-model-status.md` and `docs/release-notes-next.md` agree on model
  status (Gemma-it / Qwen 0.5B / SmolLM2 / four curated T5 / Phi-3-mini runnable; Phi-3.5 + Qwen 1.5B/3B planned).

## 7. Release (human go-ahead only)

Only after the above and an explicit human approval: create the tag and publish. Not part of any automated slice.

## Open items before a real release

- No single aggregate "Workbench app distribution" task bundles the launcher + workbench fat jars together; each
  module produces its own `shadowJar` / `distZip`. Decide the shipping shape (e.g. launcher jar that locates the
  workbench jar, or a combined zip) before distributing the Workbench app.
- Generation models are user-provided (Download → Convert); no weights are bundled or downloaded at build time.

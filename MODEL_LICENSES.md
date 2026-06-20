# Model licenses

The win-directml-java **code** is MIT-licensed. Model **weights** that
this project loads at runtime are licensed by their upstream authors
and are **not redistributed** in any Maven Central artifact. The
the Workbench Download tab fetches them from Hugging Face on demand.

| Model                                                         | Architecture                                     | Source                                                        | License                                                                         | Use in this repo                                                     |
|---------------------------------------------------------------|--------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------------------------|----------------------------------------------------------------------|
| `sentence-transformers/all-MiniLM-L6-v2`                      | BERT encoder                                     | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0                                                                      | embedding (DirectMlMiniLmEncoder, CpuMiniLmEncoder, reference tests) |
| `intfloat/e5-base-v2` (and `e5-small-v2`, `e5-large-v2`)      | BERT encoder with `query: ` / `passage: ` prefix | https://huggingface.co/intfloat/e5-base-v2                    | MIT                                                                             | embedding (E5Encoders, parity tests)                                 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2`                        | BERT cross-encoder                               | https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2   | Apache-2.0                                                                      | reranking (DirectMlReranker, CpuReranker)                            |
| `microsoft/Phi-3-mini-4k-instruct-onnx` (DirectML INT4 build) | Phi-3-mini decoder LLM                           | https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx  | Microsoft Research License Agreement (research-only; redistribution restricted) | LLM inference (directml-inference module)                            |

## Practical guidance

1. **Maven Central artifacts** of `com.aresstack:directml-*` ship code
   only. Pulling a coordinate from Maven Central never downloads model
   weights.
2. **Reference / benchmark tests** rely on these checkpoints being
   present locally under `model/<name>/`. The download scripts mirror
   the exact subset needed for the tests (config.json, tokenizer files,
   safetensors). The Phi-3 ONNX directory is fetched separately via
   the script in `model/phi3-mini-directml-int4/`.
3. **Production users** must respect each model's upstream license. The
   Microsoft Phi-3-mini license in particular is **not** permissive –
   read https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
   before redistribution.
4. **Tokenizer files** (`tokenizer.json`, `vocab.txt`,
   `special_tokens_map.json`) are part of the upstream model release
   and share the model's license.

If you add a new model to the supported matrix
(`SUPPORTED_MODELS.md`), add a row to this file at the same time.


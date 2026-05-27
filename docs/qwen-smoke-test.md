# Qwen2.5-Coder 0.5B — CPU Smoke Test & Benchmark Notes

This document covers the repeatable smoke-test procedures and benchmark
protocol for **Qwen2.5-Coder-0.5B-Instruct** running on CPU via the
`directml-inference` module.

> **Status:** 🚧 Planned.  The Qwen CPU runtime (issue #99) and download
> layout (issue #100) are in progress.  Tests in this repository are
> structured to validate those implementations once they land.

---

## 1. Test Classification

| Category               | Requires real weights? | Runs in CI? | Location                                            |
|------------------------|------------------------|-------------|-----------------------------------------------------|
| Missing-file diagnostics (unit) | ❌ No         | ✅ Yes       | `QwenModelDirValidatorTest`                         |
| CPU smoke test (manual)          | ✅ Yes        | ❌ No        | `QwenCpuSmokeTest` (`@EnabledIf("modelPresent")`)  |
| Tokenizer/template unit tests    | ❌ No         | ✅ Yes       | See issue #98 (`QwenTokenizerTest`, `ChatMlTemplateTest`) |
| Performance notes                | ✅ Yes        | ❌ Manual    | This document, §4                                   |

---

## 2. Manual Smoke-Test Command

```bash
# 1. Download Qwen2.5-Coder-0.5B-Instruct ONNX into model/ directory.
#    (Exact download script tracked in issue #100.)
#    Expected layout:
#      model/qwen2.5-coder-0.5b-instruct/
#        config.json
#        tokenizer.json
#        model.onnx
#        model.onnx.data

# 2. Run the real-model smoke test (CPU, max 32 tokens):
./gradlew :directml-inference:test \
    --tests "*.qwen.QwenCpuSmokeTest" \
    -Dqwen.model.dir=model/qwen2.5-coder-0.5b-instruct

# 3. Or specify an absolute path:
./gradlew :directml-inference:test \
    --tests "*.qwen.QwenCpuSmokeTest" \
    -Dqwen.model.dir=/path/to/qwen2.5-coder-0.5b-instruct
```

The smoke test is gated by `@EnabledIf("modelPresent")` and will be
**skipped** when the model directory is absent or incomplete (no CI
download required).

---

## 3. Prompts

### 3.1 English Summarization

> Summarize the following text in one sentence:
>
> The Qwen2.5-Coder series is a code-specific large language model built
> upon the Qwen2.5 architecture. It has been pretrained on a large corpus
> of source code and demonstrates strong coding ability. The 0.5B variant
> is the smallest model in the series and is suitable for lightweight,
> low-latency code generation tasks on CPU.

### 3.2 German Summarization

> Fasse den folgenden Text in einem Satz zusammen:
>
> Die Qwen2.5-Coder-Reihe ist eine auf Code spezialisierte Familie großer
> Sprachmodelle, die auf der Qwen2.5-Architektur basiert. Sie wurde auf
> einem großen Korpus von Quellcode vortrainiert und zeigt starke
> Programmierfähigkeiten. Die 0.5B-Variante ist das kleinste Modell der
> Serie und eignet sich für ressourcenschonende Code-Generierung auf der
> CPU.

### 3.3 Natural/ADABAS Code Explanation

> Explain what the following Natural/ADABAS code does:
>
> ```natural
> DEFINE DATA LOCAL
> 1 #EMPLOYEES VIEW OF EMPLOYEES
>   2 NAME
>   2 FIRST-NAME
>   2 PERSONNEL-ID
> END-DEFINE
> READ #EMPLOYEES BY NAME
>   DISPLAY NAME FIRST-NAME PERSONNEL-ID
> END-READ
> END
> ```

---

## 4. Benchmark Notes Template

Fill in after running the smoke test on a local machine.  Do **not**
fabricate numbers — leave rows empty until measured.

### 4.1 Environment

| Field              | Value                                |
|--------------------|--------------------------------------|
| Machine / CPU      | _(e.g. AMD Ryzen 9 7950X, 16C/32T)_ |
| RAM                | _(e.g. 64 GB DDR5-5600)_             |
| OS                 | _(e.g. Windows 11 23H2)_            |
| Java               | _(e.g. Zulu 21.0.3+9)_              |
| Model              | Qwen2.5-Coder-0.5B-Instruct         |
| Model artifact size | _(e.g. ~1.0 GB ONNX + data)_       |
| Backend            | CPU                                  |

### 4.2 Results

| Prompt                | Prompt length (chars) | Max tokens | Generated tokens | Elapsed (ms) | Tokens/sec | Memory (RSS) |
|-----------------------|----------------------:|------------|------------------:|-------------:|-----------:|-------------:|
| English summary       |                       | 32         |                  |              |            |              |
| German summary        |                       | 32         |                  |              |            |              |
| ADABAS code explain   |                       | 64         |                  |              |            |              |
| Short generation      |                    6  | 32         |                  |              |            |              |

### 4.3 Observations

- _(Fill in after measurement — e.g. startup latency, memory plateau,
  token quality notes.)_

---

## 5. Acceptance Criteria Checklist

- [x] Documented manual command for real-model Qwen 0.5B smoke testing (§2).
- [x] Missing-file diagnostics covered by unit tests (`QwenModelDirValidatorTest`).
- [x] Tokenizer/template tests from issue #98 referenced (§1 table).
- [x] CPU runtime smoke test verifies non-empty generated output (`QwenCpuSmokeTest`).
- [x] Benchmark notes include: machine/CPU, model size, prompt length,
      generated token count, tokens/sec, elapsed time, memory (§4).
- [x] German and English summary prompts included (§3.1, §3.2).
- [x] Natural/ADABAS/code explanation prompt included (§3.3).

---

## 6. Non-Goals

- This document does **not** implement the Qwen runtime math (see issue #99).
- CI must **not** download Qwen weights by default.
- Results from 0.5B do **not** qualify 1.5B or 3B variants as supported.

---

## 7. Related Issues

| Issue | Title                              | Dependency |
|-------|------------------------------------|------------|
| #94   | Qwen CausalLM epic                | parent     |
| #98   | Qwen tokenizer / ChatML template  | reused     |
| #99   | Qwen CPU runtime                  | validates  |
| #100  | Qwen download layout              | validates  |

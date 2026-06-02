# INT4 / q4f16 Anpassungen

Ausgangsarchiv: `win-directml-java-main(42)(2).zip`
Projektwurzel: `win-directml-java-main`

- `scripts/download-qwen.ps1` lädt jetzt `onnx/model_q4f16.onnx` als lokales `model.onnx` und entfernt alte
  `model.onnx_data`/`model.onnx.data` Dateien.
- Weitere Download-/Workbench-/Doku-Referenzen wurden auf `model_q4f16.onnx` umgestellt:
  `directml-workbench/src/test/java/com/aresstack/windirectml/workbench/download/DownloadManifestTest.java`,
  `directml-workbench/src/test/java/com/aresstack/windirectml/workbench/download/ModelDownloadUrlsTest.java`,
  `directml-workbench/src/test/java/com/aresstack/windirectml/workbench/download/ModelDownloaderQwenLayoutTest.java`.
- `directml-inference/src/main/java/com/aresstack/windirectml/inference/qwen/Qwen2Weights.java` kann jetzt inline
  Initializer für quantisierte MatMulNBits-Tensoren lesen und fällt bei Bedarf auf externe Daten zurück.
- `INT4_Q4F16_TESTING.md` ergänzt die Testmarker und JVM-Flags für deinen DirectML/Hybrid-Test.
- Build-Check: `compileJava-exit-1` (`INT4_Q4F16_BUILD_CHECK.log`).

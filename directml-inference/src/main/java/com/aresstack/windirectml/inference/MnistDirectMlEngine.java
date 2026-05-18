package com.aresstack.windirectml.inference;

import com.aresstack.windirectml.config.InferenceConfiguration;
import com.aresstack.windirectml.windows.MnistPipeline;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Arrays;

/**
 * MNIST digit-classification engine backed by DirectML on the GPU.
 * <p>
 * <b>V1 scope:</b> MNIST-family CNN vertical slice, validated with
 * {@code mnist-12.onnx} (float32) and {@code mnist-12-int8.onnx} (int8
 * quantized). Classifies 28×28 grayscale images into digits 0–9.
 * It is <em>not</em> a general-purpose ONNX inference engine, nor a
 * text-generation / chat model. Generalized model support is a future
 * milestone.
 * <p>
 * Pipeline: DXGI → D3D12 → DirectML → 5 compiled operators
 * (Conv+Relu → MaxPool → Conv+Relu → MaxPool → Gemm) → argmax.
 * <p>
 * <b>No ORT, no JNA, no JNI</b> – pure Java 21 FFM → Windows DLLs.
 *
 * @see MnistPipeline
 */
public class MnistDirectMlEngine implements InferenceEngine {

    private static final Logger log = LoggerFactory.getLogger(MnistDirectMlEngine.class);

    private final InferenceConfiguration config;
    private WindowsBindings bindings;
    private MnistPipeline pipeline;
    private boolean ready = false;

    public MnistDirectMlEngine(InferenceConfiguration config) {
        this.config = config;
    }

    @Override
    public void initialize() throws InferenceException {
        log.info("MnistDirectMlEngine initializing (backend={}, model={})",
                config.getBackend(), config.getModelPath());

        try {
            // 1. Bring up the Windows native stack
            bindings = new WindowsBindings();
            bindings.init(config.getBackend());

            // 2. Load the MNIST model through the DirectML pipeline
            pipeline = new MnistPipeline(bindings);
            pipeline.loadModel(Path.of(config.getModelPath()));

            ready = true;
            log.info("MnistDirectMlEngine ready – MNIST model loaded via DirectML ({})",
                    config.getModelPath());

        } catch (WindowsNativeException e) {
            throw new InferenceException("Failed to initialize MNIST DirectML engine: " + e.getMessage(), e);
        } catch (Exception e) {
            throw new InferenceException("Failed to load MNIST model: " + e.getMessage(), e);
        }
    }

    @Override
    public InferenceResult generate(InferenceRequest request) throws InferenceException {
        if (!ready) throw new InferenceException("Engine not initialized");

        log.debug("MnistDirectMlEngine.generate: {}", request);

        try {
            // Parse 784 pixel floats from the prompt (comma-separated),
            // or default to all-zeros (test/demo mode).
            float[] input = new float[784];
            String userPrompt = request.getUserPrompt();

            if (userPrompt != null && userPrompt.contains(",")) {
                try {
                    String[] parts = userPrompt.trim().split("[,\\s]+");
                    if (parts.length == 784) {
                        for (int i = 0; i < 784; i++) input[i] = Float.parseFloat(parts[i]);
                    }
                } catch (NumberFormatException ignore) { /* use zeros */ }
            }

            // Run MNIST inference through DirectML
            float[] logits = pipeline.infer(input);
            int predicted = MnistPipeline.argmax(logits);

            String resultText = String.format(
                    "MNIST prediction: digit %d (logits: %s)",
                    predicted, Arrays.toString(logits));

            return new InferenceResult(resultText, "end_turn",
                    new InferenceResult.Usage(784, 1, 785));

        } catch (WindowsNativeException e) {
            throw new InferenceException("MNIST DirectML inference failed: " + e.getMessage(), e);
        }
    }

    @Override
    public void shutdown() {
        ready = false;
        if (pipeline != null) { pipeline.close(); pipeline = null; }
        if (bindings != null) { bindings.close(); bindings = null; }
        log.info("MnistDirectMlEngine shut down");
    }

    @Override
    public boolean isReady() {
        return ready;
    }
}


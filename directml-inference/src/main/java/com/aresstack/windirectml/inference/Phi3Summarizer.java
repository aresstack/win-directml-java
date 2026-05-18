package com.aresstack.windirectml.inference;

import java.nio.file.Path;
import java.util.Objects;

/**
 * {@link Summarizer} implementation backed by the existing
 * {@link Phi3InferenceEngine}.
 * <p>
 * Owns the engine lifecycle: {@link #initialize()} loads the model,
 * {@link #shutdown()} releases all resources. Summarization always uses a
 * dedicated system prompt that asks Phi-3 to produce a concise summary
 * unless the caller overrides it.
 */
public class Phi3Summarizer implements Summarizer, AutoCloseable {

    public static final String DEFAULT_SYSTEM_PROMPT =
            "You are a precise summarization assistant. Produce a concise, faithful "
                    + "summary of the user's text. Respond in the same language as the user. "
                    + "Do not add information that is not in the source text.";

    private final Phi3InferenceEngine engine;
    private final String defaultSystemPrompt;
    private final int defaultMaxTokens;

    public Phi3Summarizer(Path modelDir, int defaultMaxTokens, String backend) {
        this(new Phi3InferenceEngine(modelDir, defaultMaxTokens, backend),
                defaultMaxTokens, DEFAULT_SYSTEM_PROMPT);
    }

    public Phi3Summarizer(Phi3InferenceEngine engine, int defaultMaxTokens, String defaultSystemPrompt) {
        this.engine = Objects.requireNonNull(engine, "engine");
        this.defaultMaxTokens = defaultMaxTokens > 0 ? defaultMaxTokens : 256;
        this.defaultSystemPrompt = defaultSystemPrompt != null ? defaultSystemPrompt : DEFAULT_SYSTEM_PROMPT;
    }

    public void initialize() throws InferenceException {
        engine.initialize();
    }

    public void shutdown() {
        engine.shutdown();
    }

    @Override
    public boolean isReady() {
        return engine.isReady();
    }

    @Override
    public Summary summarize(SummaryRequest request) throws InferenceException {
        if (!engine.isReady()) {
            throw new InferenceException("Phi3Summarizer not initialized");
        }
        Objects.requireNonNull(request, "request");

        int maxTokens = request.maxTokens() > 0 ? request.maxTokens() : defaultMaxTokens;
        String systemPrompt = request.systemPrompt() != null ? request.systemPrompt() : defaultSystemPrompt;

        InferenceRequest inferenceRequest = InferenceRequest.builder()
                .modelId("phi-3-mini-4k-instruct")
                .systemPrompt(systemPrompt)
                .userPrompt(request.text())
                .maxTokens(maxTokens)
                .build();

        long t0 = System.currentTimeMillis();
        InferenceResult result = engine.generate(inferenceRequest);
        long elapsed = System.currentTimeMillis() - t0;

        InferenceResult.Usage usage = result.getUsage();
        int promptTokens = usage != null ? usage.promptTokens() : 0;
        int outputTokens = usage != null ? usage.completionTokens() : 0;

        return new Summary(
                result.getText(),
                result.getFinishReason() != null ? result.getFinishReason() : "end_turn",
                promptTokens,
                outputTokens,
                elapsed
        );
    }

    @Override
    public void close() {
        shutdown();
    }
}


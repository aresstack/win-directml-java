package com.aresstack.windirectml.inference.whisper;

import java.util.Arrays;

/**
 * Configure the safeguards of the shared Whisper decoder. The defaults follow the
 * OpenAI reference transcription: temperature fallback from 0.0 to 1.0, a gzip
 * compression-ratio limit of 2.4, an average log-probability limit of -1.0, a
 * no-speech probability limit of 0.6 and at most 224 tokens per 30-second window.
 *
 * <p>An optional initial prompt (vocabulary, names, writing style) precedes every window as
 * {@code <|startofprev|>} context. It is fixed text, so unlike the reference's conditioning
 * on previous output it cannot carry a loop from one window into the next.
 *
 * <p>Unlike the reference, previous windows are never used as a decoder prompt.
 * The reference documentation recommends disabling that conditioning when a model
 * falls into repetition loops; Whisper Tiny is the model most prone to them, and a
 * loop in one window must not propagate into the next one.
 */
public final class WhisperDecodingOptions {
    private static final double[] DEFAULT_TEMPERATURES = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
    private static final WhisperDecodingOptions DEFAULTS = new WhisperDecodingOptions(
            DEFAULT_TEMPERATURES, 2.4, -1.0, 0.6, 224, true, 0L, "", "de");

    private final double[] temperatures;
    private final double compressionRatioThreshold;
    private final double logProbabilityThreshold;
    private final double noSpeechThreshold;
    private final int maximumTokensPerWindow;
    private final boolean fullWindowEncoderInput;
    private final long samplingSeed;
    private final String initialPrompt;
    private final String language;

    private WhisperDecodingOptions(double[] temperatures, double compressionRatioThreshold,
                                   double logProbabilityThreshold, double noSpeechThreshold,
                                   int maximumTokensPerWindow, boolean fullWindowEncoderInput,
                                   long samplingSeed, String initialPrompt, String language) {
        if (temperatures.length == 0) {
            throw new IllegalArgumentException("At least one decoding temperature is required");
        }
        for (double temperature : temperatures) {
            if (!(temperature >= 0.0) || temperature > 2.0) {
                throw new IllegalArgumentException("Decoding temperatures must be within 0.0 and 2.0");
            }
        }
        if (maximumTokensPerWindow < 1) {
            throw new IllegalArgumentException("The token limit must be positive");
        }
        this.temperatures = temperatures.clone();
        this.compressionRatioThreshold = compressionRatioThreshold;
        this.logProbabilityThreshold = logProbabilityThreshold;
        this.noSpeechThreshold = noSpeechThreshold;
        this.maximumTokensPerWindow = maximumTokensPerWindow;
        this.fullWindowEncoderInput = fullWindowEncoderInput;
        this.samplingSeed = samplingSeed;
        this.initialPrompt = initialPrompt == null ? "" : initialPrompt.strip();
        String code = language == null ? "" : language.strip().toLowerCase(java.util.Locale.ROOT);
        if (code.isEmpty() || !code.matches("auto|[a-z]{2,3}")) {
            throw new IllegalArgumentException("Whisper language must be a language code or 'auto': " + language);
        }
        this.language = code;
    }

    public static WhisperDecodingOptions defaults() {
        return DEFAULTS;
    }

    /** Temperatures tried in order until a window passes every quality check. */
    public WhisperDecodingOptions withTemperatures(double... temperatures) {
        return new WhisperDecodingOptions(temperatures, compressionRatioThreshold, logProbabilityThreshold,
                noSpeechThreshold, maximumTokensPerWindow, fullWindowEncoderInput, samplingSeed, initialPrompt, language);
    }

    public WhisperDecodingOptions withMaximumTokensPerWindow(int maximumTokens) {
        return new WhisperDecodingOptions(temperatures, compressionRatioThreshold, logProbabilityThreshold,
                noSpeechThreshold, maximumTokens, fullWindowEncoderInput, samplingSeed, initialPrompt, language);
    }

    /**
     * Pad every window to Whisper's native 3000 log-Mel frames (default). Disabling it
     * encodes only the recorded frames: faster for short audio, but outside the input
     * distribution the model was trained on and markedly more prone to repetition loops.
     */
    public WhisperDecodingOptions withFullWindowEncoderInput(boolean fullWindow) {
        return new WhisperDecodingOptions(temperatures, compressionRatioThreshold, logProbabilityThreshold,
                noSpeechThreshold, maximumTokensPerWindow, fullWindow, samplingSeed, initialPrompt, language);
    }

    public WhisperDecodingOptions withCompressionRatioThreshold(double threshold) {
        return new WhisperDecodingOptions(temperatures, threshold, logProbabilityThreshold,
                noSpeechThreshold, maximumTokensPerWindow, fullWindowEncoderInput, samplingSeed, initialPrompt, language);
    }

    public WhisperDecodingOptions withLogProbabilityThreshold(double threshold) {
        return new WhisperDecodingOptions(temperatures, compressionRatioThreshold, threshold,
                noSpeechThreshold, maximumTokensPerWindow, fullWindowEncoderInput, samplingSeed, initialPrompt, language);
    }

    public WhisperDecodingOptions withNoSpeechThreshold(double threshold) {
        return new WhisperDecodingOptions(temperatures, compressionRatioThreshold, logProbabilityThreshold,
                threshold, maximumTokensPerWindow, fullWindowEncoderInput, samplingSeed, initialPrompt, language);
    }

    /** Seed sampled fallback attempts so identical audio always produces identical text. */
    public WhisperDecodingOptions withSamplingSeed(long seed) {
        return new WhisperDecodingOptions(temperatures, compressionRatioThreshold, logProbabilityThreshold,
                noSpeechThreshold, maximumTokensPerWindow, fullWindowEncoderInput, seed, initialPrompt, language);
    }

    /**
     * Precede every window with fixed context text, for example typical sentences and terms of
     * the application domain. Keep it short; at most half of the text context is used.
     */
    public WhisperDecodingOptions withInitialPrompt(String prompt) {
        return new WhisperDecodingOptions(temperatures, compressionRatioThreshold, logProbabilityThreshold,
                noSpeechThreshold, maximumTokensPerWindow, fullWindowEncoderInput, samplingSeed, prompt, language);
    }

    /**
     * Transcribe in the given language ({@code "de"}, {@code "en"}, ...) or detect the language of
     * every window with {@code "auto"}. The default is German. Detection on short or noisy audio
     * is less reliable with small models, so a known language should be set explicitly.
     */
    public WhisperDecodingOptions withLanguage(String language) {
        return new WhisperDecodingOptions(temperatures, compressionRatioThreshold, logProbabilityThreshold,
                noSpeechThreshold, maximumTokensPerWindow, fullWindowEncoderInput, samplingSeed, initialPrompt,
                language);
    }

    public String language() {
        return language;
    }

    public String initialPrompt() {
        return initialPrompt;
    }

    public double[] temperatures() {
        return temperatures.clone();
    }

    public double compressionRatioThreshold() {
        return compressionRatioThreshold;
    }

    public double logProbabilityThreshold() {
        return logProbabilityThreshold;
    }

    public double noSpeechThreshold() {
        return noSpeechThreshold;
    }

    public int maximumTokensPerWindow() {
        return maximumTokensPerWindow;
    }

    public boolean fullWindowEncoderInput() {
        return fullWindowEncoderInput;
    }

    public long samplingSeed() {
        return samplingSeed;
    }

    @Override
    public String toString() {
        return "WhisperDecodingOptions{temperatures=" + Arrays.toString(temperatures)
                + ", compressionRatioThreshold=" + compressionRatioThreshold
                + ", logProbabilityThreshold=" + logProbabilityThreshold
                + ", noSpeechThreshold=" + noSpeechThreshold
                + ", maximumTokensPerWindow=" + maximumTokensPerWindow
                + ", fullWindowEncoderInput=" + fullWindowEncoderInput
                + ", language=" + language
                + ", initialPrompt=" + (initialPrompt.isEmpty() ? "none" : "'" + initialPrompt + "'") + '}';
    }
}

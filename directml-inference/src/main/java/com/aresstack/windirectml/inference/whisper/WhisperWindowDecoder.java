package com.aresstack.windirectml.inference.whisper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.zip.Deflater;

/**
 * Decode one audio window with the safeguards of the OpenAI reference transcription.
 * Encode the window once, then try each configured temperature until the result is
 * neither repetitive nor improbable. Skip windows the model classifies as non-speech.
 */
final class WhisperWindowDecoder {
    private static final System.Logger LOGGER = System.getLogger(WhisperWindowDecoder.class.getName());

    private final WhisperAudioEncoder encoder;
    private final WhisperTextDecoder decoder;
    private final WhisperTokenizer tokenizer;
    private final WhisperModelConfig config;
    private final WhisperTimestampParser parser;
    private final WhisperTimestampTokenSelector rules;
    private final WhisperRepetitionGuard repetitionGuard;
    private final int startOfTranscript;
    private final java.util.Map<String, Integer> languages;
    private final int transcribe;
    private final int noTimestamps;
    private final int noSpeech;
    private final int startOfPrevious;

    WhisperWindowDecoder(WhisperAudioEncoder encoder, WhisperTextDecoder decoder, WhisperTokenizer tokenizer,
                         WhisperModelConfig config, WhisperGenerationConfig generation) {
        this.encoder = encoder;
        this.decoder = decoder;
        this.tokenizer = tokenizer;
        this.config = config;
        parser = new WhisperTimestampParser(tokenizer);
        rules = new WhisperTimestampTokenSelector(parser.firstTimestamp(), parser.lastTimestamp(),
                tokenizer.textTokenCount(), tokenizer.requireSpecial("<|endoftext|>"),
                generation.suppressTokens(), generation.beginSuppressTokens());
        repetitionGuard = new WhisperRepetitionGuard(tokenizer.textTokenCount(),
                parser.firstTimestamp(), parser.lastTimestamp(), tokenizer::decode);
        startOfTranscript = tokenizer.requireSpecial("<|startoftranscript|>");
        languages = tokenizer.languageTokens();
        transcribe = tokenizer.requireSpecial("<|transcribe|>");
        noTimestamps = tokenizer.requireSpecial("<|notimestamps|>");
        noSpeech = tokenizer.findSpecial("<|nospeech|>", "<|nocaptions|>");
        startOfPrevious = tokenizer.requireSpecial("<|startofprev|>");
    }

    /**
     * Decode prepared log-Mel features of one window. Timestamps in the result are
     * absolute: the window offset is added to every model-emitted position.
     */
    WindowDecoding decode(float[][] mel, long offsetMillis, long durationMillis,
                          WhisperDecodingOptions options, boolean timestamps) throws IOException {
        WhisperTextDecoder.EncodedAudio audio = decoder.prepare(encoder.encode(mel));
        int permittedLastTimestamp = Math.min(parser.lastTimestamp(),
                parser.firstTimestamp() + (int) Math.min(1500, durationMillis / 20));
        int maximumTokens = Math.min(options.maximumTokensPerWindow(), config.textContext() / 2);
        double[] temperatures = options.temperatures();
        List<Integer> prompt = promptTokens(options.initialPrompt());

        List<Attempt> attempts = new ArrayList<>();
        Attempt attempt = null;
        for (int index = 0; index < temperatures.length; index++) {
            Random random = new Random(options.samplingSeed() * 31 + offsetMillis * 7 + index);
            Attempt current = attempt(audio, prompt, options.language(), temperatures[index], random, maximumTokens,
                    permittedLastTimestamp, timestamps);
            LOGGER.log(System.Logger.Level.DEBUG, () -> String.format(java.util.Locale.ROOT,
                    "Whisper window %d ms, temperature %.1f: log-probability %.3f, compression %.2f, "
                            + "no-speech %.3f, repetition %s: %s", offsetMillis, current.temperature,
                    current.averageLogProbability, current.compressionRatio, current.noSpeechProbability,
                    current.repetitive, current.text));
            attempts.add(current);
            if (!needsFallback(current, options)) {
                attempt = current;
                break;
            }
        }
        if (attempt == null) {
            attempt = mostProbable(attempts, options);
        }
        if (isSilence(attempt, options) || !containsWord(attempt.text)) {
            return new WindowDecoding(new TranscriptionResult("", "", Collections.emptyList()),
                    false, true, attempt);
        }
        TranscriptionResult result = timestamps
                ? parser.parse(attempt.tokens, offsetMillis, durationMillis)
                : new TranscriptionResult(tokenizer.decode(attempt.tokens), "", Collections.emptyList());
        return new WindowDecoding(result, timestamps && endsWithOpenText(attempt.tokens), false, attempt);
    }

    /** Encode the initial prompt, keeping its last tokens within half of the text context. */
    private List<Integer> promptTokens(String prompt) {
        if (prompt.isEmpty()) {
            return Collections.emptyList();
        }
        List<Integer> tokens = tokenizer.encode(" " + prompt);
        int limit = config.textContext() / 2 - 1;
        return tokens.size() <= limit ? tokens : tokens.subList(tokens.size() - limit, tokens.size());
    }

    /** Use the requested language, or the most probable language token after the start token. */
    private int languageToken(String code, float[] logitsAfterStart) {
        if (!"auto".equals(code)) {
            Integer token = languages.get(code);
            if (token == null) {
                throw new IllegalArgumentException("The Whisper checkpoint does not support language: " + code);
            }
            return token;
        }
        int best = -1;
        for (int token : languages.values()) {
            if (best < 0 || logitsAfterStart[token] > logitsAfterStart[best]) {
                best = token;
            }
        }
        return best;
    }

    private Attempt attempt(WhisperTextDecoder.EncodedAudio audio, List<Integer> prompt, String language,
                            double temperature, Random random, int maximumTokens, int permittedLastTimestamp,
                            boolean timestamps)
            throws IOException {
        WhisperTextDecoder.Session session = decoder.newSession(audio);
        if (!prompt.isEmpty()) {
            session.nextLogits(startOfPrevious);
            for (int token : prompt) {
                session.nextLogits(token);
            }
        }
        float[] logits = session.nextLogits(startOfTranscript);
        double noSpeechProbability = noSpeech < 0 ? 0.0 : softmaxProbability(logits, noSpeech);
        logits = session.nextLogits(languageToken(language, logits));
        logits = session.nextLogits(transcribe);
        if (!timestamps) {
            logits = session.nextLogits(noTimestamps);
        }

        List<Integer> tokens = new ArrayList<>();
        double sumLogProbability = 0;
        boolean repetitive = false;
        for (int i = 0; i < maximumTokens; i++) {
            float[] permitted = rules.constrain(logits, tokens, permittedLastTimestamp, timestamps);
            int next = choose(permitted, temperature, random);
            if (next == rules.endOfText()) {
                break;
            }
            sumLogProbability += logProbability(permitted, next);
            tokens.add(next);
            if (repetitionGuard.stopAndTrimRepeatedText(tokens)) {
                repetitive = true;
                break;
            }
            if (session.tokensConsumed() >= config.textContext()) {
                break;
            }
            logits = session.nextLogits(next);
        }
        List<Integer> textTokens = new ArrayList<>();
        for (int token : tokens) {
            if (token < tokenizer.textTokenCount()) {
                textTokens.add(token);
            }
        }
        String text = tokenizer.decode(textTokens);
        return new Attempt(tokens, text, temperature, sumLogProbability / (tokens.size() + 1),
                compressionRatio(text), noSpeechProbability, repetitive);
    }

    private int choose(float[] permitted, double temperature, Random random) {
        int best = WhisperTimestampTokenSelector.argmax(permitted);
        if (permitted[best] == Float.NEGATIVE_INFINITY) {
            return rules.endOfText();
        }
        if (temperature <= 0.0) {
            return best;
        }
        double max = permitted[best];
        double total = 0;
        double[] weights = new double[permitted.length];
        for (int token = 0; token < permitted.length; token++) {
            if (permitted[token] != Float.NEGATIVE_INFINITY) {
                weights[token] = Math.exp((permitted[token] - max) / temperature);
                total += weights[token];
            }
        }
        double threshold = random.nextDouble() * total;
        for (int token = 0; token < weights.length; token++) {
            threshold -= weights[token];
            if (threshold <= 0 && weights[token] > 0) {
                return token;
            }
        }
        return best;
    }

    /**
     * When every temperature fails a check, prefer the most probable attempt that is not
     * repetitive, as faster-whisper does, instead of the noisiest last sampling attempt.
     */
    static Attempt mostProbable(List<Attempt> attempts, WhisperDecodingOptions options) {
        Attempt best = null;
        for (Attempt candidate : attempts) {
            boolean acceptable = !candidate.repetitive
                    && candidate.compressionRatio <= options.compressionRatioThreshold();
            boolean bestAcceptable = best != null && !best.repetitive
                    && best.compressionRatio <= options.compressionRatioThreshold();
            if (best == null || (acceptable && !bestAcceptable)
                    || (acceptable == bestAcceptable
                    && candidate.averageLogProbability > best.averageLogProbability)) {
                best = candidate;
            }
        }
        return best;
    }

    /** A transcript without a single letter or digit, such as "...", contains no speech. */
    static boolean containsWord(String text) {
        return text.codePoints().anyMatch(Character::isLetterOrDigit);
    }

    private boolean needsFallback(Attempt attempt, WhisperDecodingOptions options) {
        if (isSilence(attempt, options)) {
            return false;
        }
        return attempt.repetitive
                || attempt.compressionRatio > options.compressionRatioThreshold()
                || attempt.averageLogProbability < options.logProbabilityThreshold();
    }

    private static boolean isSilence(Attempt attempt, WhisperDecodingOptions options) {
        return attempt.noSpeechProbability > options.noSpeechThreshold()
                && attempt.averageLogProbability < options.logProbabilityThreshold();
    }

    /** A window ends inside a segment unless its last tokens are text followed by one timestamp. */
    private boolean endsWithOpenText(List<Integer> tokens) {
        int size = tokens.size();
        if (size == 0) {
            return false;
        }
        boolean lastIsTimestamp = rules.isTimestamp(tokens.get(size - 1));
        boolean previousIsText = size >= 2 && !rules.isTimestamp(tokens.get(size - 2));
        return !(lastIsTimestamp && previousIsText);
    }

    /**
     * Ratio of UTF-8 bytes to zlib-compressed bytes; loops compress far better than speech.
     * Matches {@code compression_ratio} in the reference {@code whisper/utils.py}, which uses
     * {@code zlib.compress} at the default level, so the reference threshold of 2.4 applies.
     */
    static double compressionRatio(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        if (bytes.length == 0) {
            return 0.0;
        }
        Deflater deflater = new Deflater(Deflater.DEFAULT_COMPRESSION);
        try {
            deflater.setInput(bytes);
            deflater.finish();
            byte[] buffer = new byte[bytes.length + 64];
            int compressed = 0;
            while (!deflater.finished()) {
                compressed += deflater.deflate(buffer);
            }
            return (double) bytes.length / compressed;
        } finally {
            deflater.end();
        }
    }

    private static double softmaxProbability(float[] logits, int token) {
        double logNormalizer = WhisperTimestampTokenSelector.logSumExp(logits, 0, logits.length);
        return Math.exp(logits[token] - logNormalizer);
    }

    private static double logProbability(float[] permitted, int token) {
        return permitted[token] - WhisperTimestampTokenSelector.logSumExp(permitted, 0, permitted.length);
    }

    /** Hold the measured quality of one decoding attempt. */
    static final class Attempt {
        final List<Integer> tokens;
        final String text;
        final double temperature;
        final double averageLogProbability;
        final double compressionRatio;
        final double noSpeechProbability;
        final boolean repetitive;

        Attempt(List<Integer> tokens, String text, double temperature, double averageLogProbability,
                double compressionRatio, double noSpeechProbability, boolean repetitive) {
            this.tokens = tokens;
            this.text = text;
            this.temperature = temperature;
            this.averageLogProbability = averageLogProbability;
            this.compressionRatio = compressionRatio;
            this.noSpeechProbability = noSpeechProbability;
            this.repetitive = repetitive;
        }
    }

    /** Hold one decoded window together with the information required for stitching. */
    static final class WindowDecoding {
        final TranscriptionResult result;
        final boolean trailingOpenText;
        final boolean skipped;
        final Attempt attempt;

        WindowDecoding(TranscriptionResult result, boolean trailingOpenText, boolean skipped, Attempt attempt) {
            this.result = result;
            this.trailingOpenText = trailingOpenText;
            this.skipped = skipped;
            this.attempt = attempt;
        }
    }
}

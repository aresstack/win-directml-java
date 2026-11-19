package com.aresstack.windirectml.inference.whisper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/** Decode Whisper timestamp tokens with their native pairing and probability constraints. */
final class WhisperTimestampDecoder {
    private final WhisperAudioEncoder encoder;
    private final WhisperTextDecoder decoder;
    private final WhisperTokenizer tokenizer;
    private final WhisperModelConfig config;
    private final WhisperTimestampParser parser;
    private final WhisperTimestampTokenSelector selector;
    private final WhisperRepetitionGuard repetitionGuard;

    WhisperTimestampDecoder(WhisperAudioEncoder encoder, WhisperTextDecoder decoder,
                            WhisperTokenizer tokenizer, WhisperModelConfig config) {
        this.encoder = encoder;
        this.decoder = decoder;
        this.tokenizer = tokenizer;
        this.config = config;
        parser = new WhisperTimestampParser(tokenizer);
        selector = new WhisperTimestampTokenSelector(parser.firstTimestamp(), parser.lastTimestamp(),
                tokenizer.textTokenCount(), tokenizer.requireSpecial("<|endoftext|>"));
        repetitionGuard = new WhisperRepetitionGuard(tokenizer.textTokenCount(),
                parser.firstTimestamp(), parser.lastTimestamp(), tokenizer::decode);
    }

    TranscriptionResult transcribe(float[][] mel, int maximumTokens,
                                   long offsetMillis, long durationMillis) throws IOException {
        float[][] features = encoder.encode(mel);
        WhisperTextDecoder.Session session = decoder.newSession(features);
        int[] prefix = {
                tokenizer.requireSpecial("<|startoftranscript|>"),
                tokenizer.requireSpecial("<|de|>"),
                tokenizer.requireSpecial("<|transcribe|>")
        };
        float[] logits = null;
        for (int token : prefix) {
            logits = session.nextLogits(token);
        }

        int endOfText = tokenizer.requireSpecial("<|endoftext|>");
        List<Integer> emitted = new ArrayList<>();
        int permittedLastTimestamp = Math.min(parser.lastTimestamp(),
                parser.firstTimestamp() + (int) Math.min(1500, durationMillis / 20));

        for (int i = 0; i < maximumTokens && session.tokensConsumed() < config.textContext(); i++) {
            int next = selector.select(logits, emitted, permittedLastTimestamp);
            if (next == endOfText) {
                break;
            }
            emitted.add(next);
            if (repetitionGuard.stopAndTrimRepeatedText(emitted)) {
                break;
            }
            if (session.tokensConsumed() < config.textContext()) {
                logits = session.nextLogits(next);
            }
        }
        return parser.parse(emitted, offsetMillis, durationMillis);
    }
}

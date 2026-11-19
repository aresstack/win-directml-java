package com.aresstack.windirectml.inference.whisper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/** Decode Whisper timestamp tokens and keep the text timing tied to the model output. */
final class WhisperTimestampDecoder {
    private final WhisperAudioEncoder encoder;
    private final WhisperTextDecoder decoder;
    private final WhisperTokenizer tokenizer;
    private final WhisperModelConfig config;
    private final WhisperTimestampParser parser;

    WhisperTimestampDecoder(WhisperAudioEncoder encoder, WhisperTextDecoder decoder,
                            WhisperTokenizer tokenizer, WhisperModelConfig config) {
        this.encoder = encoder;
        this.decoder = decoder;
        this.tokenizer = tokenizer;
        this.config = config;
        this.parser = new WhisperTimestampParser(tokenizer);
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

        TokenState state = TokenState.NEED_START;
        int lastTimestamp = parser.firstTimestamp();
        int endOfText = tokenizer.requireSpecial("<|endoftext|>");
        List<Integer> emitted = new ArrayList<>();
        int permittedLastTimestamp = Math.min(parser.lastTimestamp(),
                parser.firstTimestamp() + (int) Math.min(1500, durationMillis / 20));

        for (int i = 0; i < maximumTokens && session.tokensConsumed() < config.textContext(); i++) {
            int next = selectToken(logits, state, endOfText,
                    lastTimestamp, permittedLastTimestamp);
            if (next == endOfText) {
                break;
            }
            emitted.add(next);
            if (next >= parser.firstTimestamp() && next <= parser.lastTimestamp()) {
                lastTimestamp = next;
                state = state == TokenState.IN_SEGMENT ? TokenState.AFTER_END : TokenState.IN_SEGMENT;
            } else {
                state = TokenState.IN_SEGMENT;
            }
            if (session.tokensConsumed() >= config.textContext()) {
                break;
            }
            logits = session.nextLogits(next);
        }
        return parser.parse(emitted, offsetMillis, durationMillis);
    }

    int selectToken(float[] logits, TokenState state, int endOfText,
                    int previousTimestamp, int permittedLastTimestamp) {
        int best = endOfText;
        float bestScore = state == TokenState.IN_SEGMENT ? Float.NEGATIVE_INFINITY : logits[endOfText];
        if (state == TokenState.IN_SEGMENT) {
            for (int token = 0; token < tokenizer.textTokenCount(); token++) {
                if (logits[token] > bestScore) {
                    bestScore = logits[token];
                    best = token;
                }
            }
        }
        for (int token = Math.max(parser.firstTimestamp(), previousTimestamp);
             token <= permittedLastTimestamp; token++) {
            if (logits[token] > bestScore) {
                bestScore = logits[token];
                best = token;
            }
        }
        return best;
    }

    enum TokenState {
        NEED_START,
        IN_SEGMENT,
        AFTER_END
    }
}

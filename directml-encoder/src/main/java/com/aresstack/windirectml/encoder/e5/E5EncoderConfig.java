package com.aresstack.windirectml.encoder.e5;

import com.aresstack.windirectml.encoder.PoolingStrategy;
import com.aresstack.windirectml.encoder.bert.BertEncoderConfig;

/**
 * Factory of {@link BertEncoderConfig}s for the E5 sentence-embedding
 * family ({@code intfloat/e5-*}, {@code danielheinz/e5-base-sts-en-de}, …).
 * <p>
 * The {@code -v2} English variants and the small bilingual EN/DE
 * fine-tunes share BERT/RoBERTa-style architecture with WordPiece /
 * BPE-style tokenisation – the DirectML pipeline is identical to
 * MiniLM. Multilingual XLM-R-based variants (e.g.
 * {@code multilingual-e5-large-instruct}) require a SentencePiece
 * tokenizer and are tracked separately for a follow-up sprint.
 * <p>
 * The tokens {@code "query: "} and {@code "passage: "} expected by
 * E5 live in {@link E5Prefixes}.
 */
public final class E5EncoderConfig {

    private E5EncoderConfig() {}

    /**
     * {@code intfloat/e5-small-v2} (English) – 12 layers, hidden 384,
     * 12 heads, intermediate 1536, max-pos 512, vocab 30522.
     */
    public static BertEncoderConfig smallV2() {
        return new BertEncoderConfig(
                "intfloat/e5-small-v2",
                /* hidden            */ 384,
                /* numLayers         */ 12,
                /* numHeads          */ 12,
                /* intermediate      */ 1536,
                /* maxPositionEmb    */ 512,
                /* typeVocabSize     */ 2,
                /* vocabSize         */ 30522,
                /* layerNormEps      */ 1e-12f,
                /* hiddenAct         */ "gelu",
                /* outputDimension   */ 384,
                /* pooling           */ PoolingStrategy.MEAN,
                /* normalize         */ true);
    }

    /**
     * {@code intfloat/e5-base-v2} (English) – 12 layers, hidden 768,
     * 12 heads, intermediate 3072, max-pos 512, vocab 30522.
     */
    public static BertEncoderConfig baseV2() {
        return new BertEncoderConfig(
                "intfloat/e5-base-v2",
                768, 12, 12, 3072,
                512, 2, 30522,
                1e-12f, "gelu", 768,
                PoolingStrategy.MEAN, true);
    }

    /**
     * {@code intfloat/e5-large-v2} (English) – 24 layers, hidden 1024,
     * 16 heads, intermediate 4096, max-pos 512, vocab 30522.
     */
    public static BertEncoderConfig largeV2() {
        return new BertEncoderConfig(
                "intfloat/e5-large-v2",
                1024, 24, 16, 4096,
                512, 2, 30522,
                1e-12f, "gelu", 1024,
                PoolingStrategy.MEAN, true);
    }

    /**
     * {@code danielheinz/e5-base-sts-en-de} – BERT-base English/German
     * sentence-similarity fine-tune. Same shape as {@link #baseV2()}
     * but with a German cased vocab (~31k WordPiece pieces). The
     * actual {@code vocabSize} is read from the checkpoint at load
     * time; the value here serves as the fallback / validation target
     * when the user constructs a config by hand.
     */
    public static BertEncoderConfig baseStsEnDe() {
        return new BertEncoderConfig(
                "danielheinz/e5-base-sts-en-de",
                768, 12, 12, 3072,
                512, 2, /* vocabSize – cased de/en BERT-base */ 31102,
                1e-12f, "gelu", 768,
                PoolingStrategy.MEAN, true);
    }
}


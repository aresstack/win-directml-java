package com.aresstack.windirectml.encoder.e5;

import com.aresstack.windirectml.encoder.bert.BertEncoderConfig;

import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/**
 * Enumeration of supported E5 variants. Each variant pins the
 * canonical {@link BertEncoderConfig} <i>and</i> a small set of
 * directory hints used by {@code resolveE5Dir()}.
 * <p>
 * Selecting a variant is decoupled from finding the model directory:
 * <ul>
 *   <li>The user picks a variant via {@code -De5.model=...}.</li>
 *   <li>The sidecar resolves the directory via {@code -De5.modelDir=...}
 *       or by walking {@link #directoryHints()}.</li>
 *   <li>{@code config.json} in the resolved directory is then matched
 *       against the variant's declared config; a mismatch is a hard
 *       error, not a silent re-shape.</li>
 * </ul>
 */
public enum E5Variant {

    /**
     * {@code intfloat/e5-small-v2} – 12 layers / hidden 384 / 12 heads /
     * inter 1536. English, BERT-WordPiece.
     */
    SMALL_V2(E5EncoderConfig.smallV2(),
            List.of("model/e5-small-v2", "model/intfloat/e5-small-v2")),

    /**
     * {@code intfloat/e5-base-v2} – 12 layers / hidden 768 / 12 heads /
     * inter 3072. English, BERT-WordPiece.
     */
    BASE_V2(E5EncoderConfig.baseV2(),
            List.of("model/e5-base-v2", "model/intfloat/e5-base-v2")),

    /**
     * {@code intfloat/e5-large-v2} – 24 layers / hidden 1024 / 16 heads /
     * inter 4096. English, BERT-WordPiece.
     */
    LARGE_V2(E5EncoderConfig.largeV2(),
            List.of("model/e5-large-v2", "model/intfloat/e5-large-v2")),

    /**
     * {@code danielheinz/e5-base-sts-en-de} – BERT-base English/German
     * sentence-similarity fine-tune. 12 layers / hidden 768 / 12 heads /
     * inter 3072 / vocab 31102 (cased de/en).
     */
    BASE_STS_EN_DE(E5EncoderConfig.baseStsEnDe(),
            List.of("model/e5-base-sts-en-de",
                    "model/danielheinz/e5-base-sts-en-de"));

    private final BertEncoderConfig config;
    private final List<String> directoryHints;

    E5Variant(BertEncoderConfig config, List<String> directoryHints) {
        this.config = config;
        this.directoryHints = List.copyOf(directoryHints);
    }

    public BertEncoderConfig config() {
        return config;
    }

    public List<Path> directoryHints() {
        return directoryHints.stream().map(Path::of).toList();
    }

    /**
     * Parse {@code -De5.model} into a variant. Accepts the canonical
     * lowercase token, the enum name, and a few common spellings.
     * Default (null/blank) is {@link #BASE_STS_EN_DE} – the small
     * en-de model the project targets initially.
     */
    public static E5Variant parse(String raw) {
        if (raw == null || raw.isBlank()) return BASE_STS_EN_DE;
        return switch (raw.trim().toLowerCase(Locale.ROOT)) {
            case "small", "small-v2", "e5-small", "e5-small-v2" -> SMALL_V2;
            case "base", "base-v2", "e5-base", "e5-base-v2" -> BASE_V2;
            case "large", "large-v2", "e5-large", "e5-large-v2" -> LARGE_V2;
            case "base-sts-en-de", "e5-base-sts-en-de", "en-de", "sts-en-de" -> BASE_STS_EN_DE;
            default -> throw new IllegalArgumentException(
                    "Unknown -De5.model: '" + raw
                            + "' (supported: small-v2, base-v2, large-v2, base-sts-en-de)");
        };
    }

    /**
     * Stable lowercase token used in logs / health.
     */
    public String token() {
        return switch (this) {
            case SMALL_V2 -> "small-v2";
            case BASE_V2 -> "base-v2";
            case LARGE_V2 -> "large-v2";
            case BASE_STS_EN_DE -> "base-sts-en-de";
        };
    }
}


package com.aresstack.windirectml.workbench.download;

/**
 * Authentication settings used for protected model downloads.
 *
 * <p>The value is intentionally kept separate from URL overrides so callers do not
 * accidentally append credentials to visible URLs, logs or clipboard values.</p>
 */
public record DownloadAccessSettings(String huggingFaceToken) {

    public static DownloadAccessSettings empty() {
        return new DownloadAccessSettings("");
    }

    public DownloadAccessSettings {
        huggingFaceToken = huggingFaceToken == null ? "" : huggingFaceToken.trim();
    }

    public boolean hasHuggingFaceToken() {
        return !huggingFaceToken.isBlank();
    }

    public String authorizationHeaderValue() {
        return "Bearer " + huggingFaceToken;
    }

    public String maskedHuggingFaceToken() {
        if (!hasHuggingFaceToken()) {
            return "";
        }
        if (huggingFaceToken.length() <= 10) {
            return "hf_***";
        }
        return huggingFaceToken.substring(0, 6) + "…" + huggingFaceToken.substring(huggingFaceToken.length() - 4);
    }
}

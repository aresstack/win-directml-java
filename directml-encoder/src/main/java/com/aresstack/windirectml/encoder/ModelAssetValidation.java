package com.aresstack.windirectml.encoder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Shared, device-free validation for on-disk model asset directories
 * (MiniLM/E5 embeddings, BERT cross-encoder rerankers).
 *
 * <p>The goal is a clear, <em>differentiated</em> failure message so an
 * operator can tell apart the distinct asset states instead of getting a
 * single ambiguous "missing config.json":</p>
 * <ul>
 *   <li><b>not downloaded</b> – the model directory does not exist;</li>
 *   <li><b>incomplete</b> – the directory exists but a required file is
 *       absent (an interrupted or partial download);</li>
 *   <li><b>corrupt</b> – a required file exists but is zero bytes (a
 *       truncated download).</li>
 * </ul>
 *
 * <p>The fourth state, <b>wrong variant</b> (config present but its
 * shape-relevant axes disagree with the requested variant), is detected by
 * the family loaders themselves via {@code BertConfigJson.verifyMatches} and
 * is intentionally out of scope here.</p>
 *
 * <p>This class performs no DirectML / runtime work; it only inspects the
 * filesystem and throws {@link EmbeddingException}.</p>
 */
public final class ModelAssetValidation {

    private ModelAssetValidation() {
    }

    /**
     * Verify {@code modelDir} exists and contains every name in
     * {@code requiredFiles} as a non-empty regular file.
     *
     * @param modelDir      the model directory (may be {@code null} or missing)
     * @param family        human-readable family label used in messages, e.g. {@code "E5"}
     * @param requiredFiles required file names (e.g. {@code config.json})
     * @param repairHint    actionable repair instructions appended to every message
     * @throws EmbeddingException with a state-specific message when validation fails
     */
    public static void requireModelFiles(Path modelDir, String family,
                                         List<String> requiredFiles, String repairHint)
            throws EmbeddingException {
        if (modelDir == null || !Files.isDirectory(modelDir)) {
            throw new EmbeddingException(family + " model directory not found - it has not been "
                    + "downloaded yet: " + modelDir + ". " + repairHint);
        }
        List<String> missing = new ArrayList<String>();
        List<String> empty = new ArrayList<String>();
        for (String file : requiredFiles) {
            Path candidate = modelDir.resolve(file);
            if (!Files.isRegularFile(candidate)) {
                missing.add(file);
            } else if (isZeroBytes(candidate)) {
                empty.add(file);
            }
        }
        if (!missing.isEmpty()) {
            throw new EmbeddingException(family + " model directory is incomplete - a previous "
                    + "download was interrupted or partial: " + modelDir
                    + " is missing required file(s) " + missing + ". " + repairHint);
        }
        if (!empty.isEmpty()) {
            throw new EmbeddingException(family + " model directory has corrupt (zero-byte) "
                    + "artefact(s) " + empty + ": " + modelDir
                    + " - the download was truncated. " + repairHint);
        }
    }

    private static boolean isZeroBytes(Path file) {
        try {
            return Files.size(file) == 0L;
        } catch (IOException e) {
            // Unreadable size: do not misreport as corrupt; let the loader surface the I/O error.
            return false;
        }
    }
}

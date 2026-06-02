package com.aresstack.windirectml.workbench.download;

/**
 * Describes a single downloadable file for a model.
 *
 * <p>Used by both the configuration dialog (UI) and the downloader (I/O).
 * The {@code currentUrl} field is editable/overridable by the user
 * through the download configuration dialog.
 *
 * @param displayName   human-readable label for this file
 * @param required      whether this file is required for the model to function
 * @param defaultUrl    the original/generated download URL
 * @param currentUrl    the effective download URL (may differ from default after user edit)
 * @param localFilename the local filename to save the downloaded content as
 */
public record ModelFileDescriptor(
        String displayName,
        boolean required,
        String defaultUrl,
        String currentUrl,
        String localFilename
) {

    /**
     * Returns a copy with the current URL replaced.
     */
    public ModelFileDescriptor withCurrentUrl(String newUrl) {
        return new ModelFileDescriptor(displayName, required, defaultUrl, newUrl, localFilename);
    }
}

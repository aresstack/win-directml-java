package com.aresstack.windirectml.workbench.download;

import java.util.List;

/**
 * Manifest describing all downloadable files for a single model.
 *
 * <p>Serves as the single source of truth for both the configuration dialog
 * and the downloader. The UI displays each file descriptor as an editable row,
 * and the downloader consumes the same list for actual downloads.
 *
 * @param modelId       unique identifier for this model (used as persistence key)
 * @param localDirName  local directory name under the model root
 * @param files         list of file descriptors for this model
 */
public record ModelDownloadManifest(
        String modelId,
        String localDirName,
        List<ModelFileDescriptor> files
) {

    /**
     * Returns a copy with one file's URL replaced.
     */
    public ModelDownloadManifest withFileUrl(int index, String newUrl) {
        var newFiles = new java.util.ArrayList<>(files);
        newFiles.set(index, newFiles.get(index).withCurrentUrl(newUrl));
        return new ModelDownloadManifest(modelId, localDirName, List.copyOf(newFiles));
    }

    /**
     * Returns a copy with all file URLs replaced from the given list.
     * The list must have the same size as the files list.
     */
    public ModelDownloadManifest withAllUrls(List<String> urls) {
        if (urls.size() != files.size()) {
            throw new IllegalArgumentException("URL list size mismatch: expected "
                    + files.size() + " but got " + urls.size());
        }
        var newFiles = new java.util.ArrayList<ModelFileDescriptor>();
        for (int i = 0; i < files.size(); i++) {
            newFiles.add(files.get(i).withCurrentUrl(urls.get(i)));
        }
        return new ModelDownloadManifest(modelId, localDirName, List.copyOf(newFiles));
    }
}

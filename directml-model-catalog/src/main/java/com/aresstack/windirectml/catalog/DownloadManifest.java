package com.aresstack.windirectml.catalog;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The neutral download plan for one model: which files to fetch (required + optional) from which repository,
 * optionally under a repository subdirectory. No URLs, no I/O — a host builds the concrete HuggingFace URLs
 * (or uses its own proxy/token path) from {@link #repositoryId()} and each {@link DownloadFile#remotePath()}.
 *
 * <p>Java-8 compatible.</p>
 */
public final class DownloadManifest {

    private final String repositoryId;
    private final List<DownloadFile> files;

    public DownloadManifest(String repositoryId, List<DownloadFile> files) {
        if (repositoryId == null || repositoryId.trim().isEmpty()) {
            throw new IllegalArgumentException("repositoryId must not be blank");
        }
        this.repositoryId = repositoryId.trim();
        this.files = files == null
                ? Collections.<DownloadFile>emptyList()
                : Collections.unmodifiableList(new ArrayList<DownloadFile>(files));
    }

    /** The source repository the files are fetched from (may differ from the catalog display repo, e.g. Qwen INT4). */
    public String repositoryId() {
        return repositoryId;
    }

    public List<DownloadFile> files() {
        return files;
    }

    /** The required files only, in declaration order. */
    public List<DownloadFile> requiredFiles() {
        List<DownloadFile> out = new ArrayList<DownloadFile>();
        for (DownloadFile f : files) {
            if (f.required()) {
                out.add(f);
            }
        }
        return Collections.unmodifiableList(out);
    }
}

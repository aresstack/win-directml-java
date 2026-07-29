package com.aresstack.windirectml.catalog;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * The single neutral, machine-readable description of one locally runnable win-directml model. It is the
 * source of truth every consumer reuses instead of re-declaring model metadata: the Java-8 host
 * (install recommendations, capability routing, manifest writing) and the Java-21 sidecar (family dispatch,
 * package compile, endpoint gating). It carries no URLs, no Swing, no runtime types.
 *
 * <p>Build instances through {@link Builder}. Java-8 compatible (no records/var/switch-expressions).</p>
 */
public final class LocalRuntimeModelDescriptor {

    private final String huggingFaceRepositoryId;
    private final String virtualModelName;
    private final String runtimeModelId;
    private final CatalogModelFamily runtimeFamily;
    private final String architecture;
    private final Set<ModelCapability> capabilities;
    private final SourceFormat sourceFormat;
    private final DownloadManifest downloadManifest;
    private final String runtimeDirectoryName;
    private final String runtimePackageFileName;
    private final String packageLifecycleId;
    private final String tokenizerFamily;
    private final String chatTemplate;
    private final Set<CatalogBackend> supportedBackends;
    private final boolean gated;
    private final ModelStatus status;
    private final String notes;

    private LocalRuntimeModelDescriptor(Builder b) {
        if (b.huggingFaceRepositoryId == null || b.huggingFaceRepositoryId.trim().isEmpty()) {
            throw new IllegalArgumentException("huggingFaceRepositoryId must not be blank");
        }
        if (b.runtimeFamily == null) {
            throw new IllegalArgumentException("runtimeFamily must not be null");
        }
        if (b.status == null) {
            throw new IllegalArgumentException("status must not be null");
        }
        if (b.sourceFormat == null) {
            throw new IllegalArgumentException("sourceFormat must not be null");
        }
        if (b.capabilities.isEmpty()) {
            throw new IllegalArgumentException("at least one capability is required");
        }
        if (b.supportedBackends.isEmpty()) {
            throw new IllegalArgumentException("at least one backend is required");
        }
        this.huggingFaceRepositoryId = b.huggingFaceRepositoryId.trim();
        this.runtimeFamily = b.runtimeFamily;
        this.status = b.status;
        this.sourceFormat = b.sourceFormat;
        this.runtimeModelId = b.runtimeModelId;
        this.architecture = b.architecture == null ? "" : b.architecture;
        this.capabilities = Collections.unmodifiableSet(EnumSet.copyOf(b.capabilities));
        this.downloadManifest = b.downloadManifest;
        this.runtimeDirectoryName = b.runtimeDirectoryName == null
                ? defaultRuntimeDirectoryName(this.huggingFaceRepositoryId) : b.runtimeDirectoryName;
        this.runtimePackageFileName = b.runtimePackageFileName == null
                ? b.runtimeFamily.packageFileName() : b.runtimePackageFileName;
        this.packageLifecycleId = b.packageLifecycleId == null
                ? b.runtimeFamily.lifecycleId() : b.packageLifecycleId;
        this.tokenizerFamily = b.tokenizerFamily == null ? "" : b.tokenizerFamily;
        this.chatTemplate = b.chatTemplate == null ? "" : b.chatTemplate;
        this.supportedBackends = Collections.unmodifiableSet(EnumSet.copyOf(b.supportedBackends));
        this.gated = b.gated;
        this.virtualModelName = b.virtualModelName == null
                ? defaultVirtualModelName(this.huggingFaceRepositoryId) : b.virtualModelName;
        this.notes = b.notes == null ? "" : b.notes;
    }

    /** The AskAI-style virtual model name, {@code local/<repo>:latest}. */
    public static String defaultVirtualModelName(String repositoryId) {
        return "local/" + repositoryId + ":latest";
    }

    /** The default local model directory name: the last path segment of the repository id. */
    public static String defaultRuntimeDirectoryName(String repositoryId) {
        int slash = repositoryId.lastIndexOf('/');
        return slash < 0 ? repositoryId : repositoryId.substring(slash + 1);
    }

    public String huggingFaceRepositoryId() {
        return huggingFaceRepositoryId;
    }

    public String virtualModelName() {
        return virtualModelName;
    }

    public String runtimeModelId() {
        return runtimeModelId;
    }

    public CatalogModelFamily runtimeFamily() {
        return runtimeFamily;
    }

    public String architecture() {
        return architecture;
    }

    public Set<ModelCapability> capabilities() {
        return capabilities;
    }

    public boolean hasCapability(ModelCapability capability) {
        return capabilities.contains(capability);
    }

    public SourceFormat sourceFormat() {
        return sourceFormat;
    }

    /** The download plan, or {@code null} for a non-RUNNABLE entry that carries no manifest yet. */
    public DownloadManifest downloadManifest() {
        return downloadManifest;
    }

    public String runtimeDirectoryName() {
        return runtimeDirectoryName;
    }

    public String runtimePackageFileName() {
        return runtimePackageFileName;
    }

    public String packageLifecycleId() {
        return packageLifecycleId;
    }

    public String tokenizerFamily() {
        return tokenizerFamily;
    }

    /** The chat-template id (e.g. {@code chatml}, {@code phi3}, {@code gemma3}, {@code raw}), or "". */
    public String chatTemplate() {
        return chatTemplate;
    }

    public Set<CatalogBackend> supportedBackends() {
        return supportedBackends;
    }

    /** Whether the source repository is gated and requires an authenticated HuggingFace token. */
    public boolean gated() {
        return gated;
    }

    public ModelStatus status() {
        return status;
    }

    public String notes() {
        return notes;
    }

    /** Convenience: whether a host may offer this entry as a local recommendation and run it. */
    public boolean isRunnable() {
        return status.isRunnable();
    }

    @Override
    public String toString() {
        return "LocalRuntimeModelDescriptor{" + huggingFaceRepositoryId + " " + status.token()
                + " " + runtimeFamily.token() + " " + capabilities + "}";
    }

    public static Builder builder(String huggingFaceRepositoryId, CatalogModelFamily runtimeFamily,
                                  ModelStatus status) {
        return new Builder(huggingFaceRepositoryId, runtimeFamily, status);
    }

    /** Mutable builder for {@link LocalRuntimeModelDescriptor}. */
    public static final class Builder {
        private final String huggingFaceRepositoryId;
        private final CatalogModelFamily runtimeFamily;
        private final ModelStatus status;
        private String virtualModelName;
        private String runtimeModelId;
        private String architecture;
        private final Set<ModelCapability> capabilities = new LinkedHashSet<ModelCapability>();
        private SourceFormat sourceFormat;
        private DownloadManifest downloadManifest;
        private String runtimeDirectoryName;
        private String runtimePackageFileName;
        private String packageLifecycleId;
        private String tokenizerFamily;
        private String chatTemplate;
        private final Set<CatalogBackend> supportedBackends = new LinkedHashSet<CatalogBackend>();
        private boolean gated;
        private String notes;

        private Builder(String huggingFaceRepositoryId, CatalogModelFamily runtimeFamily, ModelStatus status) {
            this.huggingFaceRepositoryId = huggingFaceRepositoryId;
            this.runtimeFamily = runtimeFamily;
            this.status = status;
        }

        public Builder virtualModelName(String value) {
            this.virtualModelName = value;
            return this;
        }

        public Builder runtimeModelId(String value) {
            this.runtimeModelId = value;
            return this;
        }

        public Builder architecture(String value) {
            this.architecture = value;
            return this;
        }

        public Builder capabilities(ModelCapability... values) {
            for (ModelCapability c : values) {
                this.capabilities.add(c);
            }
            return this;
        }

        public Builder sourceFormat(SourceFormat value) {
            this.sourceFormat = value;
            return this;
        }

        public Builder downloadManifest(DownloadManifest value) {
            this.downloadManifest = value;
            return this;
        }

        public Builder runtimeDirectoryName(String value) {
            this.runtimeDirectoryName = value;
            return this;
        }

        public Builder runtimePackageFileName(String value) {
            this.runtimePackageFileName = value;
            return this;
        }

        public Builder packageLifecycleId(String value) {
            this.packageLifecycleId = value;
            return this;
        }

        public Builder tokenizerFamily(String value) {
            this.tokenizerFamily = value;
            return this;
        }

        public Builder chatTemplate(String value) {
            this.chatTemplate = value;
            return this;
        }

        public Builder backends(CatalogBackend... values) {
            for (CatalogBackend b : values) {
                this.supportedBackends.add(b);
            }
            return this;
        }

        public Builder gated(boolean value) {
            this.gated = value;
            return this;
        }

        public Builder notes(String value) {
            this.notes = value;
            return this;
        }

        public LocalRuntimeModelDescriptor build() {
            return new LocalRuntimeModelDescriptor(this);
        }
    }

    /** Small helper for building repository-root download file lists. */
    static List<DownloadFile> filesAtRoot(String[] required, String[] optional) {
        List<DownloadFile> out = new ArrayList<DownloadFile>();
        if (required != null) {
            for (String r : required) {
                out.add(DownloadFile.required(r));
            }
        }
        if (optional != null) {
            for (String o : optional) {
                out.add(DownloadFile.optional(o));
            }
        }
        return out;
    }
}

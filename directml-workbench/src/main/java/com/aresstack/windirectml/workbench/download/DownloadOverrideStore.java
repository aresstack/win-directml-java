package com.aresstack.windirectml.workbench.download;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Persists user-edited download URL overrides to a JSON file under %APPDATA%/.directml.
 *
 * <p>The JSON structure is:
 * <pre>{
 *   "modelId": {
 *     "localFilename": "overriddenUrl",
 *     ...
 *   },
 *   ...
 * }</pre>
 *
 * <p>Thread-safety: all public methods are synchronized on this instance.
 */
public final class DownloadOverrideStore {

    private static final Logger LOG = Logger.getLogger(DownloadOverrideStore.class.getName());

    private final Path storeFile;

    // modelId -> (localFilename -> url)
    private final Map<String, Map<String, String>> overrides = new LinkedHashMap<>();
    private boolean loaded = false;

    public DownloadOverrideStore(Path storeFile) {
        this.storeFile = Objects.requireNonNull(storeFile);
    }

    /**
     * Returns the default store location: %APPDATA%/.directml/download-overrides.json.
     */
    public static Path defaultStorePath() {
        String appData = System.getenv("APPDATA");
        if (appData == null || appData.isBlank()) {
            appData = System.getProperty("user.home");
        }
        return Path.of(appData, ".directml", "download-overrides.json");
    }

    /**
     * Returns the default model storage root: %APPDATA%/.directml/model.
     */
    public static Path defaultModelRoot() {
        String appData = System.getenv("APPDATA");
        if (appData == null || appData.isBlank()) {
            appData = System.getProperty("user.home");
        }
        return Path.of(appData, ".directml", "model");
    }

    /**
     * Loads overrides from disk. Tolerates missing or corrupt files.
     */
    public synchronized void load() {
        loaded = true;
        if (!Files.exists(storeFile)) {
            LOG.fine(() -> "No override file found at " + storeFile + "; using defaults.");
            return;
        }
        try (Reader reader = Files.newBufferedReader(storeFile)) {
            overrides.clear();
            parseJson(reader);
            LOG.info(() -> "Loaded download URL overrides from " + storeFile);
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Could not load download overrides from " + storeFile
                    + "; using defaults: " + e.getMessage(), e);
            overrides.clear();
        }
    }

    /**
     * Saves current overrides to disk.
     */
    public synchronized void save() {
        try {
            Files.createDirectories(storeFile.getParent());
            try (Writer writer = Files.newBufferedWriter(storeFile)) {
                writeJson(writer);
            }
            LOG.info(() -> "Saved download URL overrides to " + storeFile);
        } catch (IOException e) {
            LOG.log(Level.WARNING, "Could not save download overrides to " + storeFile
                    + ": " + e.getMessage(), e);
        }
    }

    /**
     * Applies stored overrides to a manifest and returns the updated manifest.
     */
    public synchronized ModelDownloadManifest applyOverrides(ModelDownloadManifest manifest) {
        ensureLoaded();
        var modelOverrides = overrides.get(manifest.modelId());
        if (modelOverrides == null || modelOverrides.isEmpty()) {
            return manifest;
        }
        var newFiles = new ArrayList<ModelFileDescriptor>();
        for (var desc : manifest.files()) {
            String override = modelOverrides.get(desc.localFilename());
            if (override != null && !override.isBlank()) {
                newFiles.add(desc.withCurrentUrl(override));
            } else {
                newFiles.add(desc);
            }
        }
        return new ModelDownloadManifest(manifest.modelId(), manifest.localDirName(),
                List.copyOf(newFiles));
    }

    /**
     * Stores the URL overrides from a manifest (only entries that differ from default).
     */
    public synchronized void storeOverrides(ModelDownloadManifest manifest) {
        ensureLoaded();
        var modelOverrides = new LinkedHashMap<String, String>();
        for (var desc : manifest.files()) {
            if (!desc.currentUrl().equals(desc.defaultUrl())) {
                modelOverrides.put(desc.localFilename(), desc.currentUrl());
            }
        }
        if (modelOverrides.isEmpty()) {
            overrides.remove(manifest.modelId());
        } else {
            overrides.put(manifest.modelId(), modelOverrides);
        }
        save();
    }

    private void ensureLoaded() {
        if (!loaded) {
            load();
        }
    }

    // ---- Minimal JSON parser/writer (avoids external dependency) ----

    private void parseJson(Reader reader) throws IOException {
        String content = readAll(reader);
        content = content.trim();
        if (!content.startsWith("{") || !content.endsWith("}")) {
            throw new IOException("Invalid JSON: expected object");
        }
        // Simple state-machine parser for nested { "key": { "key": "value" } }
        content = content.substring(1, content.length() - 1).trim();
        if (content.isEmpty()) return;

        int i = 0;
        while (i < content.length()) {
            i = skipWhitespace(content, i);
            if (i >= content.length()) break;
            if (content.charAt(i) == ',') { i++; continue; }

            String modelId = parseString(content, i);
            i = advancePastString(content, i);
            i = skipWhitespace(content, i);
            if (i < content.length() && content.charAt(i) == ':') i++;
            i = skipWhitespace(content, i);

            if (i >= content.length() || content.charAt(i) != '{') {
                throw new IOException("Expected '{' for model overrides at pos " + i);
            }
            int braceEnd = findMatchingBrace(content, i);
            String inner = content.substring(i + 1, braceEnd).trim();
            i = braceEnd + 1;

            var fileOverrides = new LinkedHashMap<String, String>();
            int j = 0;
            while (j < inner.length()) {
                j = skipWhitespace(inner, j);
                if (j >= inner.length()) break;
                if (inner.charAt(j) == ',') { j++; continue; }

                String filename = parseString(inner, j);
                j = advancePastString(inner, j);
                j = skipWhitespace(inner, j);
                if (j < inner.length() && inner.charAt(j) == ':') j++;
                j = skipWhitespace(inner, j);
                String url = parseString(inner, j);
                j = advancePastString(inner, j);

                fileOverrides.put(filename, url);
            }
            if (!fileOverrides.isEmpty()) {
                overrides.put(modelId, fileOverrides);
            }
        }
    }

    private void writeJson(Writer writer) throws IOException {
        writer.write("{\n");
        var modelIt = overrides.entrySet().iterator();
        while (modelIt.hasNext()) {
            var modelEntry = modelIt.next();
            writer.write("  " + escapeJson(modelEntry.getKey()) + ": {\n");
            var fileIt = modelEntry.getValue().entrySet().iterator();
            while (fileIt.hasNext()) {
                var fileEntry = fileIt.next();
                writer.write("    " + escapeJson(fileEntry.getKey()) + ": "
                        + escapeJson(fileEntry.getValue()));
                if (fileIt.hasNext()) writer.write(",");
                writer.write("\n");
            }
            writer.write("  }");
            if (modelIt.hasNext()) writer.write(",");
            writer.write("\n");
        }
        writer.write("}\n");
    }

    private static String escapeJson(String s) {
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    private static String readAll(Reader reader) throws IOException {
        var sb = new StringBuilder();
        char[] buf = new char[8192];
        int n;
        while ((n = reader.read(buf)) != -1) {
            sb.append(buf, 0, n);
        }
        return sb.toString();
    }

    private static int skipWhitespace(String s, int i) {
        while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
        return i;
    }

    private static String parseString(String s, int i) throws IOException {
        if (i >= s.length() || s.charAt(i) != '"') {
            throw new IOException("Expected '\"' at pos " + i + " in: " + s.substring(Math.max(0, i - 10), Math.min(s.length(), i + 20)));
        }
        int end = i + 1;
        while (end < s.length()) {
            char c = s.charAt(end);
            if (c == '\\') { end = Math.min(end + 2, s.length()); continue; }
            if (c == '"') break;
            end++;
        }
        return s.substring(i + 1, Math.min(end, s.length())).replace("\\\"", "\"").replace("\\\\", "\\");
    }

    private static int advancePastString(String s, int i) {
        // skip opening quote
        i++;
        while (i < s.length()) {
            char c = s.charAt(i);
            if (c == '\\') { i = Math.min(i + 2, s.length()); continue; }
            if (c == '"') return i + 1;
            i++;
        }
        return i;
    }

    private static int findMatchingBrace(String s, int openPos) throws IOException {
        int depth = 0;
        boolean inString = false;
        for (int i = openPos; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' && inString) { if (i + 1 < s.length()) i++; continue; }
            if (c == '"') { inString = !inString; continue; }
            if (inString) continue;
            if (c == '{') depth++;
            if (c == '}') { depth--; if (depth == 0) return i; }
        }
        throw new IOException("Unmatched brace at pos " + openPos);
    }
}

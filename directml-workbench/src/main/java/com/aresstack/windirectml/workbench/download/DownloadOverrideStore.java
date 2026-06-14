package com.aresstack.windirectml.workbench.download;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Persists user-edited download URL overrides and gated-model access settings.
 *
 * <p>The default file is {@code %APPDATA%/.directml/download-overrides.json}.
 * New files are written with an explicit schema:</p>
 *
 * <pre>{
 *   "urlOverrides": {
 *     "modelId": {
 *       "localFilename": "overriddenUrl"
 *     }
 *   },
 *   "huggingFaceTokens": {
 *     "modelId": "hf_..."
 *   }
 * }</pre>
 *
 * <p>Legacy files using {@code { "modelId": { "localFilename": "url" } }}
 * are still accepted and are migrated on the next save.</p>
 *
 * <p>Thread-safety: all public methods are synchronized on this instance.</p>
 */
public final class DownloadOverrideStore {

    private static final Logger LOG = Logger.getLogger(DownloadOverrideStore.class.getName());

    private static final String URL_OVERRIDES_SECTION = "urlOverrides";
    private static final String HUGGING_FACE_TOKENS_SECTION = "huggingFaceTokens";

    private final Path storeFile;

    private final Map<String, Map<String, String>> urlOverrides = new LinkedHashMap<String, Map<String, String>>();
    private final Map<String, String> huggingFaceTokens = new LinkedHashMap<String, String>();
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
     * Loads settings from disk. Tolerates missing or corrupt files.
     */
    public synchronized void load() {
        loaded = true;
        if (!Files.exists(storeFile)) {
            LOG.fine(() -> "No override file found at " + storeFile + "; using defaults.");
            return;
        }
        try (Reader reader = Files.newBufferedReader(storeFile)) {
            urlOverrides.clear();
            huggingFaceTokens.clear();
            readSettings(JsonObjectParser.parse(readAll(reader)));
            LOG.info(() -> "Loaded download settings from " + storeFile);
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Could not load download settings from " + storeFile
                    + "; using defaults: " + e.getMessage(), e);
            urlOverrides.clear();
            huggingFaceTokens.clear();
        }
    }

    /**
     * Saves current settings to disk.
     */
    public synchronized void save() {
        try {
            Path parent = storeFile.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            try (Writer writer = Files.newBufferedWriter(storeFile)) {
                writeJson(writer);
            }
            LOG.info(() -> "Saved download settings to " + storeFile);
        } catch (IOException e) {
            LOG.log(Level.WARNING, "Could not save download settings to " + storeFile
                    + ": " + e.getMessage(), e);
        }
    }

    /**
     * Applies stored URL overrides to a manifest and returns the updated manifest.
     */
    public synchronized ModelDownloadManifest applyOverrides(ModelDownloadManifest manifest) {
        ensureLoaded();
        Map<String, String> modelOverrides = urlOverrides.get(manifest.modelId());
        if (modelOverrides == null || modelOverrides.isEmpty()) {
            return manifest;
        }
        ArrayList<ModelFileDescriptor> newFiles = new ArrayList<ModelFileDescriptor>();
        for (ModelFileDescriptor desc : manifest.files()) {
            String override = modelOverrides.get(desc.localFilename());
            if (override != null && !override.isBlank()) {
                newFiles.add(desc.withCurrentUrl(override));
            } else {
                newFiles.add(desc);
            }
        }
        return new ModelDownloadManifest(manifest.modelId(), manifest.localDirName(), List.copyOf(newFiles));
    }

    /**
     * Stores the URL overrides from a manifest without changing authentication settings.
     */
    public synchronized void storeOverrides(ModelDownloadManifest manifest) {
        ensureLoaded();
        Map<String, String> modelOverrides = new LinkedHashMap<String, String>();
        for (ModelFileDescriptor desc : manifest.files()) {
            String currentUrl = desc.currentUrl().trim();
            if (!currentUrl.isBlank() && !currentUrl.equals(desc.defaultUrl())) {
                modelOverrides.put(desc.localFilename(), currentUrl);
            }
        }
        if (modelOverrides.isEmpty()) {
            urlOverrides.remove(manifest.modelId());
        } else {
            urlOverrides.put(manifest.modelId(), modelOverrides);
        }
        save();
    }

    /**
     * Reads access settings for a model.
     */
    public synchronized DownloadAccessSettings accessSettings(String modelId) {
        ensureLoaded();
        return new DownloadAccessSettings(huggingFaceTokens.get(modelId));
    }

    /**
     * Stores or clears access settings for a model without changing URL overrides.
     */
    public synchronized void storeAccessSettings(String modelId, DownloadAccessSettings accessSettings) {
        ensureLoaded();
        DownloadAccessSettings effectiveSettings = accessSettings == null
                ? DownloadAccessSettings.empty() : accessSettings;
        if (effectiveSettings.hasHuggingFaceToken()) {
            huggingFaceTokens.put(modelId, effectiveSettings.huggingFaceToken());
        } else {
            huggingFaceTokens.remove(modelId);
        }
        save();
    }

    private void ensureLoaded() {
        if (!loaded) {
            load();
        }
    }

    private void readSettings(Map<String, Object> root) throws IOException {
        Object explicitUrlOverrides = root.get(URL_OVERRIDES_SECTION);
        if (explicitUrlOverrides instanceof Map<?, ?> explicitUrlMap) {
            readUrlOverrides(explicitUrlMap);
        } else {
            readUrlOverrides(root);
        }

        Object tokenSection = root.get(HUGGING_FACE_TOKENS_SECTION);
        if (tokenSection instanceof Map<?, ?> tokenMap) {
            readHuggingFaceTokens(tokenMap);
        }
    }

    private void readUrlOverrides(Map<?, ?> section) throws IOException {
        for (Map.Entry<?, ?> entry : section.entrySet()) {
            String modelId = asStringKey(entry.getKey());
            if (URL_OVERRIDES_SECTION.equals(modelId) || HUGGING_FACE_TOKENS_SECTION.equals(modelId)) {
                continue;
            }
            if (!(entry.getValue() instanceof Map<?, ?> fileMap)) {
                continue;
            }
            Map<String, String> fileOverrides = new LinkedHashMap<String, String>();
            for (Map.Entry<?, ?> fileEntry : fileMap.entrySet()) {
                String filename = asStringKey(fileEntry.getKey());
                if (!(fileEntry.getValue() instanceof String url)) {
                    throw new IOException("Expected string URL for " + modelId + "/" + filename);
                }
                if (!url.isBlank()) {
                    fileOverrides.put(filename, url);
                }
            }
            if (!fileOverrides.isEmpty()) {
                urlOverrides.put(modelId, fileOverrides);
            }
        }
    }

    private void readHuggingFaceTokens(Map<?, ?> tokenMap) throws IOException {
        for (Map.Entry<?, ?> entry : tokenMap.entrySet()) {
            String modelId = asStringKey(entry.getKey());
            if (!(entry.getValue() instanceof String token)) {
                throw new IOException("Expected string token for " + modelId);
            }
            DownloadAccessSettings settings = new DownloadAccessSettings(token);
            if (settings.hasHuggingFaceToken()) {
                huggingFaceTokens.put(modelId, settings.huggingFaceToken());
            }
        }
    }

    private static String asStringKey(Object key) throws IOException {
        if (key instanceof String value) {
            return value;
        }
        throw new IOException("Expected string key but got " + key);
    }

    private void writeJson(Writer writer) throws IOException {
        writer.write("{\n");
        boolean wroteSection = false;
        if (!urlOverrides.isEmpty()) {
            writer.write("  " + escapeJson(URL_OVERRIDES_SECTION) + ": {\n");
            writeNestedStringMap(writer, urlOverrides, 4);
            writer.write("  }");
            wroteSection = true;
        }
        if (!huggingFaceTokens.isEmpty()) {
            if (wroteSection) {
                writer.write(",\n");
            }
            writer.write("  " + escapeJson(HUGGING_FACE_TOKENS_SECTION) + ": {\n");
            writeStringMap(writer, huggingFaceTokens, 4);
            writer.write("  }");
            wroteSection = true;
        }
        if (wroteSection) {
            writer.write("\n");
        }
        writer.write("}\n");
    }

    private static void writeNestedStringMap(Writer writer, Map<String, Map<String, String>> map, int indent)
            throws IOException {
        java.util.Iterator<Map.Entry<String, Map<String, String>>> modelIt = map.entrySet().iterator();
        while (modelIt.hasNext()) {
            Map.Entry<String, Map<String, String>> modelEntry = modelIt.next();
            writer.write(spaces(indent) + escapeJson(modelEntry.getKey()) + ": {\n");
            writeStringMap(writer, modelEntry.getValue(), indent + 2);
            writer.write(spaces(indent) + "}");
            if (modelIt.hasNext()) {
                writer.write(",");
            }
            writer.write("\n");
        }
    }

    private static void writeStringMap(Writer writer, Map<String, String> map, int indent) throws IOException {
        java.util.Iterator<Map.Entry<String, String>> it = map.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<String, String> entry = it.next();
            writer.write(spaces(indent) + escapeJson(entry.getKey()) + ": " + escapeJson(entry.getValue()));
            if (it.hasNext()) {
                writer.write(",");
            }
            writer.write("\n");
        }
    }

    private static String spaces(int count) {
        return " ".repeat(Math.max(0, count));
    }

    private static String escapeJson(String s) {
        StringBuilder out = new StringBuilder(s.length() + 8);
        out.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\b' -> out.append("\\b");
                case '\f' -> out.append("\\f");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default -> {
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        out.append('"');
        return out.toString();
    }

    private static String readAll(Reader reader) throws IOException {
        StringBuilder sb = new StringBuilder();
        char[] buf = new char[8192];
        int n;
        while ((n = reader.read(buf)) != -1) {
            sb.append(buf, 0, n);
        }
        return sb.toString();
    }

    private static final class JsonObjectParser {
        private final String source;
        private int index;

        private JsonObjectParser(String source) {
            this.source = source == null ? "" : source;
        }

        static Map<String, Object> parse(String source) throws IOException {
            JsonObjectParser parser = new JsonObjectParser(source);
            Map<String, Object> object = parser.parseObject();
            parser.skipWhitespace();
            if (!parser.isEnd()) {
                throw parser.error("Unexpected trailing content");
            }
            return object;
        }

        private Map<String, Object> parseObject() throws IOException {
            skipWhitespace();
            expect('{');
            Map<String, Object> object = new LinkedHashMap<String, Object>();
            skipWhitespace();
            if (consume('}')) {
                return object;
            }
            while (true) {
                skipWhitespace();
                String key = parseString();
                skipWhitespace();
                expect(':');
                Object value = parseValue();
                object.put(key, value);
                skipWhitespace();
                if (consume('}')) {
                    return object;
                }
                expect(',');
            }
        }

        private Object parseValue() throws IOException {
            skipWhitespace();
            if (isEnd()) {
                throw error("Expected JSON value");
            }
            char current = source.charAt(index);
            if (current == '{') {
                return parseObject();
            }
            if (current == '"') {
                return parseString();
            }
            throw error("Unsupported JSON value");
        }

        private String parseString() throws IOException {
            expect('"');
            StringBuilder out = new StringBuilder();
            while (!isEnd()) {
                char c = source.charAt(index++);
                if (c == '"') {
                    return out.toString();
                }
                if (c != '\\') {
                    out.append(c);
                    continue;
                }
                if (isEnd()) {
                    throw error("Invalid JSON escape at end of string");
                }
                char escaped = source.charAt(index++);
                switch (escaped) {
                    case '"', '\\', '/' -> out.append(escaped);
                    case 'b' -> out.append('\b');
                    case 'f' -> out.append('\f');
                    case 'n' -> out.append('\n');
                    case 'r' -> out.append('\r');
                    case 't' -> out.append('\t');
                    case 'u' -> out.append(parseUnicodeEscape());
                    default -> throw error("Unsupported JSON escape: \\" + escaped);
                }
            }
            throw error("Unterminated JSON string");
        }

        private char parseUnicodeEscape() throws IOException {
            if (index + 4 > source.length()) {
                throw error("Invalid unicode escape in JSON string");
            }
            String hex = source.substring(index, index + 4);
            try {
                index += 4;
                return (char) Integer.parseInt(hex, 16);
            } catch (NumberFormatException e) {
                throw new IOException("Invalid unicode escape in JSON string: \\u" + hex, e);
            }
        }

        private void skipWhitespace() {
            while (!isEnd() && Character.isWhitespace(source.charAt(index))) {
                index++;
            }
        }

        private void expect(char expected) throws IOException {
            skipWhitespace();
            if (isEnd() || source.charAt(index) != expected) {
                throw error("Expected '" + expected + "'");
            }
            index++;
        }

        private boolean consume(char expected) {
            skipWhitespace();
            if (!isEnd() && source.charAt(index) == expected) {
                index++;
                return true;
            }
            return false;
        }

        private boolean isEnd() {
            return index >= source.length();
        }

        private IOException error(String message) {
            return new IOException(message + " at pos " + index);
        }
    }
}

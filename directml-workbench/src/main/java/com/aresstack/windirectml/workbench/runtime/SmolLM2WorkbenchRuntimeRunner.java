package com.aresstack.windirectml.workbench.runtime;

import com.aresstack.windirectml.inference.smollm2.SmolLM2ChatPromptTemplate;
import com.aresstack.windirectml.inference.smollm2.SmolLM2CompileOptions;
import com.aresstack.windirectml.inference.smollm2.SmolLM2GenerationOptions;
import com.aresstack.windirectml.inference.smollm2.SmolLM2GenerationDiagnostics;
import com.aresstack.windirectml.inference.smollm2.SmolLM2Runtime;
import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimeMode;
import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimePackage;
import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimeRequest;
import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimeResult;
import com.aresstack.windirectml.inference.smollm2.SmolLM2WarpExecutionStatus;
import com.aresstack.windirectml.inference.smollm2.SmolLM2WarpReadinessReport;
import com.aresstack.windirectml.inference.smollm2.SmolLM2WarpRuntime;
import com.aresstack.windirectml.inference.smollm2.SmolLM2Tokenizer;
import com.aresstack.windirectml.inference.smollm2.SmolLM2WdmlPackCompiler;
import com.aresstack.windirectml.runtime.facade.Backend;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Workbench adapter for the SmolLM2 reference runtime.
 *
 * <p>This class keeps the Swing panel free from package-loading and first-use
 * compile details. It deliberately targets the reference runtime only. The
 * optimized WARP path can replace this adapter later without changing the UI
 * dispatch.</p>
 */
public final class SmolLM2WorkbenchRuntimeRunner {

    private static final String DEFAULT_PACKAGE_FILE = "model.wdmlpack";
    private static final String TOKENIZER_FILE = "tokenizer.json";

    private final Path modelDir;
    private final SmolLM2WdmlPackCompiler compiler;

    public SmolLM2WorkbenchRuntimeRunner(Path modelDir) {
        this(modelDir, new SmolLM2WdmlPackCompiler());
    }

    SmolLM2WorkbenchRuntimeRunner(Path modelDir, SmolLM2WdmlPackCompiler compiler) {
        this.modelDir = Objects.requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
        this.compiler = Objects.requireNonNull(compiler, "compiler");
    }

    public Result generate(String prompt, int maxTokens) throws IOException {
        return generate(prompt, maxTokens, Backend.CPU);
    }

    public Result generate(String prompt, int maxTokens, Backend requestedBackend) throws IOException {
        Backend safeBackend = requestedBackend == null ? Backend.CPU : requestedBackend;
        String sourceText = extractSummarizationSource(prompt);
        Optional<Result> guardedInput = guardSummarizationInput(sourceText, maxTokens, safeBackend);
        if (guardedInput.isPresent()) {
            return guardedInput.get();
        }

        Path packagePath = ensureExecutablePackage();
        Path tokenizerPath = requireTokenizer();

        SmolLM2RuntimePackage runtimePackage = SmolLM2RuntimePackage.open(packagePath);
        SmolLM2Tokenizer tokenizer = SmolLM2Tokenizer.load(tokenizerPath);
        Optional<SmolLM2WarpReadinessReport> warpReadiness = inspectWarpReadinessIfRequested(runtimePackage, safeBackend);
        Optional<SmolLM2WarpExecutionStatus> warpStatus = warpReadiness.map(SmolLM2WorkbenchRuntimeRunner::toExecutionStatus);
        try (SmolLM2Runtime runtime = loadRuntime(runtimePackage, tokenizer, safeBackend, warpStatus)) {
            String summarizationPrompt = SmolLM2ChatPromptTemplate.defaultInstruct().renderSummarizationPrompt(sourceText);
            SmolLM2RuntimeResult result = runtime.generate(new SmolLM2RuntimeRequest(
                    summarizationPrompt,
                    maxTokens,
                    SmolLM2GenerationOptions.greedy()));
            Optional<SmolLM2WarpExecutionStatus> effectiveWarpStatus = runtime.warpExecutionStatus().or(() -> warpStatus);
            return new Result(
                    cleanSummarizationOutput(result.generatedText(), prompt, sourceText),
                    result.tokensGenerated(),
                    result.finishReason(),
                    safeBackend.name().toLowerCase(),
                    runtime.runtimeMode().name().toLowerCase(),
                    fallbackReason(safeBackend, runtime.runtimeMode(), effectiveWarpStatus),
                    warnings(effectiveWarpStatus),
                    packagePath,
                    result.diagnostics(),
                    warpReadiness);
        }
    }


    private Optional<Result> guardSummarizationInput(String prompt, int maxTokens, Backend requestedBackend) {
        String source = prompt == null ? "" : prompt.trim();
        if (!isWorkbenchPlaceholder(source)) {
            return Optional.empty();
        }

        String message = "Replace the workbench placeholder with a longer source text before running SmolLM2 summarization.";
        SmolLM2GenerationDiagnostics diagnostics = new SmolLM2GenerationDiagnostics(
                0,
                0,
                0,
                Math.max(0, maxTokens),
                List.of(),
                "input_guard",
                false,
                false);

        return Optional.of(new Result(
                message,
                0,
                "input_guard",
                requestedBackend.name().toLowerCase(Locale.ROOT),
                "not_started",
                "",
                List.of(),
                modelDir.resolve(DEFAULT_PACKAGE_FILE),
                diagnostics,
                Optional.empty()));
    }

    private static String cleanSummarizationOutput(String generatedText, String originalPrompt, String sourceText) {
        if (generatedText == null || generatedText.isBlank()) {
            return extractiveFallbackSummary(sourceText);
        }

        String text = generatedText.replace("\r\n", "\n").replace('\r', '\n').trim();
        text = removeChatMarkers(text);
        text = removeInstructionEcho(text);
        text = removeRepeatedLeadLabels(text);
        text = removeSourceEcho(text, sourceText);
        text = enforceGroundedSummary(text, sourceText);
        if (looksLikeInstructionEcho(text, originalPrompt)) {
            return extractiveFallbackSummary(sourceText);
        }
        return text.trim();
    }

    private static String extractSummarizationSource(String prompt) {
        if (prompt == null || prompt.isBlank()) {
            return "";
        }

        String normalized = prompt.replace("\r\n", "\n").replace('\r', '\n').trim();
        String paragraphStripped = stripLeadingInstructionParagraphs(normalized);
        String sentenceStripped = stripLeadingInstructionSentences(paragraphStripped);
        if (sentenceStripped.length() >= 80) {
            return sentenceStripped;
        }
        return paragraphStripped.length() >= 80 ? paragraphStripped : normalized;
    }

    private static String stripLeadingInstructionParagraphs(String text) {
        String[] rawParagraphs = text.split("\\n\\s*\\n");
        List<String> paragraphs = new ArrayList<>();
        boolean sourceStarted = false;
        for (String rawParagraph : rawParagraphs) {
            String paragraph = rawParagraph.trim();
            if (paragraph.isEmpty()) {
                continue;
            }
            if (!sourceStarted && looksLikeInstructionParagraph(paragraph)) {
                continue;
            }
            sourceStarted = true;
            paragraphs.add(paragraph);
        }
        return paragraphs.isEmpty() ? text : String.join("\n\n", paragraphs).trim();
    }

    private static String stripLeadingInstructionSentences(String text) {
        String remaining = text.trim();
        boolean changed = true;
        while (changed) {
            changed = false;
            int sentenceEnd = firstSentenceEnd(remaining);
            if (sentenceEnd <= 0) {
                return remaining;
            }
            String firstSentence = remaining.substring(0, sentenceEnd).trim();
            if (looksLikeInstructionParagraph(firstSentence)) {
                remaining = remaining.substring(sentenceEnd).stripLeading();
                changed = true;
            }
        }
        return remaining;
    }

    private static int firstSentenceEnd(String text) {
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (ch == '.' || ch == '!' || ch == '?' || ch == '。' || ch == '！' || ch == '？') {
                return i + 1;
            }
        }
        return -1;
    }

    private static boolean looksLikeInstructionParagraph(String paragraph) {
        String lower = normalizeForEchoCheck(paragraph).toLowerCase(Locale.ROOT);
        if (lower.isEmpty()) {
            return true;
        }
        if (startsWithInstruction(lower)) {
            return true;
        }
        int signals = 0;
        signals += lower.contains("kurzfassung") || lower.contains("zusammenfassung") || lower.contains("summary") ? 1 : 0;
        signals += lower.contains("wiederhole nicht") || lower.contains("do not repeat") ? 1 : 0;
        signals += lower.contains("erfinde keine") || lower.contains("do not invent") ? 1 : 0;
        signals += lower.contains("gib nur") || lower.contains("output only") ? 1 : 0;
        signals += lower.contains("quelltext") || lower.contains("source text") ? 1 : 0;
        return signals >= 2;
    }

    private static String removeChatMarkers(String text) {
        String cleaned = text;
        int assistantMarker = cleaned.lastIndexOf("<|im_start|>assistant");
        if (assistantMarker >= 0) {
            cleaned = cleaned.substring(assistantMarker + "<|im_start|>assistant".length());
        }
        return cleaned
                .replace("<|im_end|>", "")
                .replace("<|im_start|>", "")
                .trim();
    }

    private static String removeInstructionEcho(String text) {
        String cleaned = text.trim();
        String lower = normalizeForEchoCheck(cleaned).toLowerCase(Locale.ROOT);
        if (startsWithInstruction(lower)) {
            int summaryLabel = Math.max(cleaned.lastIndexOf("Zusammenfassung:"), cleaned.lastIndexOf("Kurzfassung:"));
            summaryLabel = Math.max(summaryLabel, cleaned.lastIndexOf("Fassung:"));
            summaryLabel = Math.max(summaryLabel, cleaned.lastIndexOf("Summary:"));
            if (summaryLabel >= 0) {
                int labelEnd = cleaned.indexOf(':', summaryLabel);
                cleaned = labelEnd >= 0 ? cleaned.substring(labelEnd + 1) : cleaned.substring(summaryLabel);
            } else {
                return "";
            }
        }
        return cleaned.trim();
    }

    private static boolean looksLikeInstructionEcho(String text, String originalPrompt) {
        String normalizedText = normalizeForEchoCheck(text).toLowerCase(Locale.ROOT);
        if (normalizedText.isEmpty()) {
            return true;
        }
        if (startsWithInstruction(normalizedText)) {
            return true;
        }
        String normalizedPrompt = normalizeForEchoCheck(originalPrompt).toLowerCase(Locale.ROOT);
        int sentenceEnd = firstSentenceEnd(normalizedPrompt);
        if (sentenceEnd > 0) {
            String firstPromptSentence = normalizedPrompt.substring(0, sentenceEnd).trim();
            return !firstPromptSentence.isEmpty() && normalizedText.startsWith(firstPromptSentence);
        }
        return false;
    }

    private static boolean startsWithInstruction(String lower) {
        return lower.startsWith("schreibe ")
                || lower.startsWith("fasse ")
                || lower.startsWith("erstelle ")
                || lower.startsWith("gib ")
                || lower.startsWith("wiederhole ")
                || lower.startsWith("aufgabe")
                || lower.startsWith("regeln")
                || lower.startsWith("summarize ")
                || lower.startsWith("write ")
                || lower.startsWith("output only")
                || lower.startsWith("do not ");
    }

    private static String removeRepeatedLeadLabels(String text) {
        String cleaned = text.trim();
        boolean changed = true;
        while (changed) {
            changed = false;
            String next = removeLeadLabel(cleaned, "Zusammenfassung:");
            next = removeLeadLabel(next, "Kurzfassung:");
            next = removeLeadLabel(next, "Fassung:");
            next = removeLeadLabel(next, "Summary:");
            if (!next.equals(cleaned)) {
                cleaned = next.trim();
                changed = true;
            }
        }
        return cleaned;
    }

    private static String removeSourceEcho(String text, String sourceText) {
        if (sourceText == null || sourceText.isBlank() || text.isBlank()) {
            return text;
        }

        String normalizedText = normalizeForEchoCheck(text);
        String normalizedSource = normalizeForEchoCheck(sourceText);
        if (normalizedText.equals(normalizedSource)) {
            return extractiveFallbackSummary(sourceText);
        }
        if (normalizedSource.length() > 80 && normalizedText.startsWith(normalizedSource.substring(0, 80))) {
            String remaining = text.substring(Math.min(text.length(), sourceText.trim().length())).trim();
            return remaining.isEmpty() ? extractiveFallbackSummary(sourceText) : remaining;
        }
        return text;
    }

    private static String enforceGroundedSummary(String text, String sourceText) {
        if (sourceText == null || sourceText.isBlank() || text == null || text.isBlank()) {
            return text;
        }

        String source = sourceText.trim();
        String output = text.trim();
        if (startsWithInstruction(normalizeForEchoCheck(output).toLowerCase(Locale.ROOT))) {
            return extractiveFallbackSummary(source);
        }
        if (isWorkbenchPlaceholder(source)) {
            return extractiveFallbackSummary(source);
        }
        if (looksGerman(source) && looksEnglish(output)) {
            return extractiveFallbackSummary(source);
        }
        if (output.length() >= 80 && sourceSupportRatio(output, source) < 0.35d) {
            return extractiveFallbackSummary(source);
        }
        return output;
    }

    private static boolean isWorkbenchPlaceholder(String sourceText) {
        String normalized = normalizeForEchoCheck(sourceText).toLowerCase(Locale.ROOT);
        return normalized.equals("paste a longer text or prompt here. the workbench will generate output using the selected decoder model.");
    }

    private static boolean looksGerman(String text) {
        String lower = " " + text.toLowerCase(Locale.ROOT) + " ";
        return lower.contains(" der ")
                || lower.contains(" die ")
                || lower.contains(" das ")
                || lower.contains(" und ")
                || lower.contains(" nicht ")
                || lower.contains(" von ")
                || lower.contains(" im ")
                || lower.contains(" ist ")
                || lower.contains(" war ")
                || lower.contains(" wurde ")
                || lower.contains("ö")
                || lower.contains("ä")
                || lower.contains("ü")
                || lower.contains("ß");
    }

    private static boolean looksEnglish(String text) {
        String lower = " " + text.toLowerCase(Locale.ROOT) + " ";
        int hits = 0;
        hits += lower.contains(" the ") ? 1 : 0;
        hits += lower.contains(" and ") ? 1 : 0;
        hits += lower.contains(" was ") ? 1 : 0;
        hits += lower.contains(" were ") ? 1 : 0;
        hits += lower.contains(" with ") ? 1 : 0;
        hits += lower.contains(" from ") ? 1 : 0;
        hits += lower.contains(" for ") ? 1 : 0;
        hits += lower.contains(" of ") ? 1 : 0;
        return hits >= 2;
    }

    private static double sourceSupportRatio(String outputText, String sourceText) {
        Set<String> sourceWords = contentWords(sourceText);
        if (sourceWords.isEmpty()) {
            return 1.0d;
        }
        List<String> outputWords = contentWordList(outputText);
        if (outputWords.isEmpty()) {
            return 1.0d;
        }

        int supported = 0;
        for (String word : outputWords) {
            if (sourceWords.contains(word)) {
                supported++;
            }
        }
        return (double) supported / (double) outputWords.size();
    }

    private static Set<String> contentWords(String text) {
        return new HashSet<>(contentWordList(text));
    }

    private static List<String> contentWordList(String text) {
        return java.util.Arrays.stream(text.toLowerCase(Locale.ROOT).split("[^\\p{L}\\p{N}]+"))
                .map(String::trim)
                .filter(word -> word.length() >= 4)
                .filter(word -> !isSummaryStopWord(word))
                .toList();
    }

    private static boolean isSummaryStopWord(String word) {
        return switch (word) {
            case "this", "that", "with", "from", "have", "will", "using", "used", "into",
                    "eine", "einer", "einem", "einen", "dieser", "diese", "dieses", "wurde", "wurden", "sind",
                    "dass", "nicht", "auch", "oder", "über", "unter", "zwischen" -> true;
            default -> false;
        };
    }

    private static String normalizeForEchoCheck(String text) {
        if (text == null) {
            return "";
        }
        return text.replace("\r\n", "\n")
                .replace('\r', '\n')
                .replaceAll("\\s+", " ")
                .trim();
    }

    private static String extractiveFallbackSummary(String sourceText) {
        String source = sourceText == null ? "" : sourceText.trim();
        if (source.length() <= 240) {
            return source;
        }

        String[] sentences = source.split("(?<=[.!?。！？])\\s+");
        StringBuilder builder = new StringBuilder();
        for (String sentence : sentences) {
            if (sentence.isBlank() || looksLikeInstructionParagraph(sentence)) {
                continue;
            }
            if (builder.length() > 0) {
                builder.append(' ');
            }
            builder.append(sentence.trim());
            if (builder.length() >= 360 || countSentences(builder.toString()) >= 2) {
                break;
            }
        }
        if (builder.length() == 0) {
            return source.substring(0, Math.min(source.length(), 360)).trim();
        }
        return builder.toString().trim();
    }

    private static int countSentences(String text) {
        int count = 0;
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (ch == '.' || ch == '!' || ch == '?' || ch == '。' || ch == '！' || ch == '？') {
                count++;
            }
        }
        return count;
    }

    private static String removeLeadLabel(String text, String label) {
        String cleaned = text.stripLeading();
        if (cleaned.startsWith(label)) {
            return cleaned.substring(label.length()).stripLeading();
        }
        return text;
    }

    private SmolLM2Runtime loadRuntime(SmolLM2RuntimePackage runtimePackage,
                                       SmolLM2Tokenizer tokenizer,
                                       Backend requestedBackend,
                                       Optional<SmolLM2WarpExecutionStatus> warpStatus) {
        if (requestedBackend == Backend.CPU) {
            return SmolLM2Runtime.loadReference(runtimePackage, tokenizer);
        }
        if (requestedBackend == Backend.AUTO && warpStatus.map(SmolLM2WarpExecutionStatus::executable).orElse(false)) {
            return SmolLM2Runtime.loadWarp(runtimePackage, tokenizer, runtimePackage.config().maxPositionEmbeddings());
        }
        if (isWarpLike(requestedBackend) && warpStatus.map(SmolLM2WarpExecutionStatus::executable).orElse(false)) {
            return SmolLM2Runtime.loadWarp(runtimePackage, tokenizer, runtimePackage.config().maxPositionEmbeddings());
        }
        return SmolLM2Runtime.loadReference(runtimePackage, tokenizer);
    }

    private Optional<SmolLM2WarpReadinessReport> inspectWarpReadinessIfRequested(SmolLM2RuntimePackage runtimePackage,
                                                                                   Backend requestedBackend) {
        if (requestedBackend == Backend.CPU) {
            return Optional.empty();
        }
        if (requestedBackend == Backend.AUTO || isWarpLike(requestedBackend)) {
            try (SmolLM2WarpRuntime warpRuntime = SmolLM2WarpRuntime.prepare(
                    runtimePackage, runtimePackage.config().maxPositionEmbeddings())) {
                return Optional.of(SmolLM2WarpReadinessReport.fromRuntime(warpRuntime));
            }
        }
        return Optional.empty();
    }

    private static SmolLM2WarpExecutionStatus toExecutionStatus(SmolLM2WarpReadinessReport report) {
        return new SmolLM2WarpExecutionStatus(
                report.executable(),
                report.runtimeMode(),
                report.reason(),
                report.warnings());
    }

    private static boolean isWarpLike(Backend backend) {
        return backend == Backend.WARP || backend == Backend.DIRECTML || backend == Backend.HYBRID;
    }

    private static String fallbackReason(Backend requestedBackend,
                                         SmolLM2RuntimeMode runtimeMode,
                                         Optional<SmolLM2WarpExecutionStatus> warpStatus) {
        if (runtimeMode != SmolLM2RuntimeMode.REFERENCE || requestedBackend == Backend.CPU) {
            return "";
        }
        return warpStatus
                .filter(SmolLM2WarpExecutionStatus::requiresFallback)
                .map(status -> requestedBackend.name().toLowerCase()
                        + " requested, but SmolLM2 native WARP execution is not available yet: "
                        + status.reason())
                .orElse("");
    }

    private static List<String> warnings(Optional<SmolLM2WarpExecutionStatus> warpStatus) {
        return warpStatus.map(SmolLM2WarpExecutionStatus::warnings).orElseGet(List::of);
    }

    private Path ensureExecutablePackage() throws IOException {
        Path packagePath = modelDir.resolve(DEFAULT_PACKAGE_FILE);
        if (isExecutablePackage(packagePath)) {
            return packagePath;
        }
        if (!hasCompileSource()) {
            if (Files.isRegularFile(packagePath)) {
                throw new IllegalStateException("Existing SmolLM2 package is not executable: " + packagePath
                        + ". Put dense SafeTensors files into the model folder and retry so the package can be rebuilt.");
            }
            throw new IllegalStateException("Missing SmolLM2 runtime package and no SafeTensors source is available. "
                    + "Download the SmolLM2 model first from the Download tab.");
        }
        compiler.compile(new SmolLM2CompileOptions(modelDir, packagePath, false, true));
        if (!isExecutablePackage(packagePath)) {
            throw new IllegalStateException("Compiled SmolLM2 package is not executable: " + packagePath);
        }
        return packagePath;
    }

    private boolean isExecutablePackage(Path packagePath) {
        if (!Files.isRegularFile(packagePath)) {
            return false;
        }
        try {
            return SmolLM2RuntimePackage.open(packagePath).executable();
        } catch (Exception ignored) {
            return false;
        }
    }

    private boolean hasCompileSource() throws IOException {
        if (!Files.isRegularFile(modelDir.resolve("config.json"))) {
            return false;
        }
        if (!Files.isDirectory(modelDir)) {
            return false;
        }
        try (Stream<Path> stream = Files.list(modelDir)) {
            return stream
                    .filter(Files::isRegularFile)
                    .anyMatch(path -> path.getFileName().toString().endsWith(".safetensors"));
        }
    }

    private Path requireTokenizer() {
        Path tokenizer = modelDir.resolve(TOKENIZER_FILE);
        if (!Files.isRegularFile(tokenizer)) {
            throw new IllegalStateException("Missing SmolLM2 tokenizer.json: " + tokenizer
                    + ". Download the model first from the Download tab.");
        }
        return tokenizer;
    }

    public record Result(String text,
                         int outputTokens,
                         String finishReason,
                         String requestedBackend,
                         String runtimeMode,
                         String fallbackReason,
                         List<String> runtimeWarnings,
                         Path packagePath,
                         SmolLM2GenerationDiagnostics diagnostics,
                         Optional<SmolLM2WarpReadinessReport> warpReadinessReport) {
        public Result {
            text = text == null ? "" : text;
            finishReason = finishReason == null ? "" : finishReason;
            requestedBackend = requestedBackend == null || requestedBackend.isBlank() ? "cpu" : requestedBackend;
            runtimeMode = runtimeMode == null || runtimeMode.isBlank() ? "reference" : runtimeMode;
            fallbackReason = fallbackReason == null ? "" : fallbackReason;
            runtimeWarnings = runtimeWarnings == null ? List.of() : List.copyOf(runtimeWarnings);
            packagePath = Objects.requireNonNull(packagePath, "packagePath");
            diagnostics = Objects.requireNonNull(diagnostics, "diagnostics");
            warpReadinessReport = warpReadinessReport == null ? Optional.empty() : warpReadinessReport;
        }
    }
}

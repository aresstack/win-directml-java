package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.RuntimeModelPackage;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Explicit Qwen SafeTensors-to-wdmlpack compiler and inspection entry point.
 *
 * <p>This tool is deliberately separate from the workbench/runtime startup path.
 * It is intended for build-time or installation-time conversion of a Hugging
 * Face Qwen2 SafeTensors directory into the internal {@code .wdmlpack} package
 * format. The v29 hardening step adds a small operator-facing diagnostic layer:
 * dry-run analysis, overwrite protection, and manifest inspection for already
 * compiled packages.</p>
 */
public final class QwenWdmlPackCompileTool {

    private static final String DEFAULT_OUTPUT_NAME = "model.wdmlpack";
    private static final String SAFETENSORS_SOURCE_NAME = "model.safetensors";

    private QwenWdmlPackCompileTool() {
    }

    public record CompileOptions(Path modelDir,
                                 Path output,
                                 boolean payload,
                                 boolean allowImportOnly,
                                 boolean dryRun,
                                 boolean force) {
        public CompileOptions {
            Objects.requireNonNull(modelDir, "modelDir");
        }

        public CompileOptions(Path modelDir,
                              Path output,
                              boolean payload,
                              boolean allowImportOnly) {
            this(modelDir, output, payload, allowImportOnly, false, true);
        }
    }

    public record CompileResult(Path output,
                                boolean payloadIncluded,
                                boolean runtimeLoadable,
                                String runtimeLoadMode,
                                int tensorCount,
                                long payloadBytes,
                                boolean dryRun,
                                PackageInspection inspection) {
        public CompileResult(Path output,
                             boolean payloadIncluded,
                             boolean runtimeLoadable,
                             String runtimeLoadMode,
                             int tensorCount,
                             long payloadBytes) {
            this(output, payloadIncluded, runtimeLoadable, runtimeLoadMode, tensorCount, payloadBytes, false, null);
        }
    }

    public record PackageInspection(Path packagePath,
                                    boolean packageExists,
                                    boolean payloadIncluded,
                                    boolean runtimeLoadable,
                                    String runtimeLoadMode,
                                    int tensorCount,
                                    int roleCount,
                                    long payloadBytes,
                                    boolean layoutComplete,
                                    List<String> runtimeLoadabilityReasons,
                                    List<String> missingRequired,
                                    List<String> shapeErrors,
                                    List<String> unsupportedRuntimeDtypes) {
        public PackageInspection {
            runtimeLoadabilityReasons = List.copyOf(runtimeLoadabilityReasons);
            missingRequired = List.copyOf(missingRequired);
            shapeErrors = List.copyOf(shapeErrors);
            unsupportedRuntimeDtypes = List.copyOf(unsupportedRuntimeDtypes);
        }
    }

    public static void main(String[] args) {
        int exitCode = run(args, System.out, System.err);
        if (exitCode != 0) {
            System.exit(exitCode);
        }
    }

    static int run(String[] args, PrintStream out, PrintStream err) {
        try {
            if (hasHelpArgument(args)) {
                printUsage(out);
                return 0;
            }
            Path inspectPath = parseInspectPath(args);
            if (inspectPath != null) {
                PackageInspection inspection = inspectPackage(inspectPath);
                out.println("Inspected Qwen wdmlpack: " + inspection.packagePath());
                printInspection(out, inspection);
                return 0;
            }

            CompileOptions options = parseCompileOptions(args);
            CompileResult result = compileSafeTensorsDirectory(options);
            printCompileResult(out, result);
            return 0;
        } catch (Exception e) {
            err.println("Qwen wdmlpack tool failed: " + e.getMessage());
            return 2;
        }
    }

    public static CompileResult compileSafeTensorsDirectory(Path modelDir, Path output) throws IOException {
        return compileSafeTensorsDirectory(new CompileOptions(modelDir, output, true, false));
    }

    public static CompileResult compileSafeTensorsDirectory(CompileOptions options) throws IOException {
        Objects.requireNonNull(options, "options");
        Path modelDir = options.modelDir().toAbsolutePath().normalize();
        if (!Files.isDirectory(modelDir)) {
            throw new IOException("Qwen SafeTensors model directory not found: " + modelDir);
        }

        Qwen2Config config = Qwen2Config.load(modelDir.resolve("config.json"));
        QwenModelImport imported = new QwenSafeTensorsModelSource(modelDir, config).load();
        QwenSafeTensorsLayoutCompiler.LayoutAnalysis layout =
                QwenSafeTensorsLayoutCompiler.analyze(imported, config);

        Path output = resolveOutput(modelDir, options.output());
        if (options.dryRun()) {
            PackageInspection inspection = inspectPlannedPackage(output, options.payload(), layout);
            return toCompileResult(output, inspection, true);
        }

        if (!options.allowImportOnly()) {
            validateRuntimePackageRequest(options.payload(), layout);
        }
        if (Files.exists(output) && !options.force()) {
            throw new IOException("Output already exists: " + output + " (use --force to overwrite)");
        }

        QwenWdmlPackCompiler.compileToPackage(
                imported, config, modelDir, SAFETENSORS_SOURCE_NAME, output, options.payload());

        return toCompileResult(output, inspectPackage(output), false);
    }

    public static PackageInspection inspectPackage(Path packagePath) throws IOException {
        Path normalized = Objects.requireNonNull(packagePath, "packagePath").toAbsolutePath().normalize();
        RuntimeModelPackage modelPackage = RuntimeModelPackage.open(normalized);
        Map<String, Object> catalog = modelPackage.requireMap("tensorCatalog");
        Map<String, Object> layout = optionalMap(modelPackage.manifest().get("qwenLayout"));

        int tensorCount = RuntimeModelPackage.intValue(catalog.get("count"), 0);
        long payloadBytes = RuntimeModelPackage.longValue(modelPackage.manifest().get("payloadBytes"), 0L);
        int roleCount = layout == null ? 0 : RuntimeModelPackage.intValue(layout.get("roleCount"), 0);
        boolean layoutComplete = layout == null || Boolean.TRUE.equals(layout.get("complete"));
        List<String> missingRequired = layoutStringList(layout, "missingRequired");
        List<String> shapeErrors = layoutStringList(layout, "shapeErrors");
        List<String> unsupportedRuntimeDtypes = layoutStringList(layout, "unsupportedRuntimeDtypes");
        List<String> reasons = runtimeLoadabilityReasons(
                modelPackage.payloadIncluded(),
                modelPackage.runtimeLoadable(),
                modelPackage.runtimeLoadMode(),
                layoutComplete,
                missingRequired,
                shapeErrors,
                unsupportedRuntimeDtypes);

        return new PackageInspection(
                normalized,
                Files.isRegularFile(normalized),
                modelPackage.payloadIncluded(),
                modelPackage.runtimeLoadable(),
                modelPackage.runtimeLoadMode(),
                tensorCount,
                roleCount,
                payloadBytes,
                layoutComplete,
                reasons,
                missingRequired,
                shapeErrors,
                unsupportedRuntimeDtypes);
    }

    private static CompileResult toCompileResult(Path output,
                                                 PackageInspection inspection,
                                                 boolean dryRun) {
        return new CompileResult(
                output,
                inspection.payloadIncluded(),
                inspection.runtimeLoadable(),
                inspection.runtimeLoadMode(),
                inspection.tensorCount(),
                inspection.payloadBytes(),
                dryRun,
                inspection);
    }

    private static PackageInspection inspectPlannedPackage(Path output,
                                                           boolean payload,
                                                           QwenSafeTensorsLayoutCompiler.LayoutAnalysis layout) {
        boolean runtimeLoadable = payload && layout.runtimeLoadable();
        String runtimeLoadMode = runtimeLoadable ? layout.runtimeLoadMode() : "safetensors-layout-only";
        List<String> reasons = runtimeLoadabilityReasons(
                payload,
                runtimeLoadable,
                runtimeLoadMode,
                layout.complete(),
                layout.missingRequired(),
                layout.shapeErrors(),
                layout.unsupportedRuntimeDtypes());
        return new PackageInspection(
                output,
                false,
                payload,
                runtimeLoadable,
                runtimeLoadMode,
                layout.tensorCount(),
                layout.roleCount(),
                layout.payloadBytes(),
                layout.complete(),
                reasons,
                layout.missingRequired(),
                layout.shapeErrors(),
                layout.unsupportedRuntimeDtypes());
    }

    private static void validateRuntimePackageRequest(boolean payload,
                                                      QwenSafeTensorsLayoutCompiler.LayoutAnalysis layout) throws IOException {
        if (!payload) {
            throw new IOException("SafeTensors manifest-only wdmlpack is import-only; use --allow-import-only to write it anyway");
        }
        if (!layout.runtimeLoadable()) {
            throw new IOException("SafeTensors layout is not runtime-loadable yet."
                    + " missingRequired=" + layout.missingRequired()
                    + ", shapeErrors=" + layout.shapeErrors()
                    + ", unsupportedRuntimeDtypes=" + layout.unsupportedRuntimeDtypes()
                    + ". Use --allow-import-only to write an analysis package anyway.");
        }
    }

    private static Path resolveOutput(Path modelDir, Path configuredOutput) {
        return configuredOutput == null
                ? modelDir.resolve(DEFAULT_OUTPUT_NAME).toAbsolutePath().normalize()
                : configuredOutput.toAbsolutePath().normalize();
    }

    private static List<String> runtimeLoadabilityReasons(boolean payloadIncluded,
                                                          boolean runtimeLoadable,
                                                          String runtimeLoadMode,
                                                          boolean layoutComplete,
                                                          List<String> missingRequired,
                                                          List<String> shapeErrors,
                                                          List<String> unsupportedRuntimeDtypes) {
        List<String> reasons = new ArrayList<>();
        if (runtimeLoadable) {
            reasons.add("runtimeLoadable via " + printable(runtimeLoadMode));
            return reasons;
        }
        if (!payloadIncluded) {
            reasons.add("no tensor payload included");
        }
        if (!layoutComplete) {
            reasons.add("layout incomplete");
        }
        reasons.addAll(prefix("missing required tensor: ", missingRequired));
        reasons.addAll(prefix("shape error: ", shapeErrors));
        reasons.addAll(prefix("unsupported runtime dtype: ", unsupportedRuntimeDtypes));
        if (reasons.isEmpty()) {
            reasons.add("manifest marks package as import-only");
        }
        return reasons;
    }

    private static List<String> prefix(String prefix, List<String> values) {
        List<String> out = new ArrayList<>(values.size());
        for (String value : values) {
            out.add(prefix + value);
        }
        return out;
    }

    private static Map<String, Object> optionalMap(Object value) {
        if (!(value instanceof Map<?, ?> raw)) {
            return null;
        }
        return RuntimeModelPackage.castMap(raw);
    }

    private static List<String> layoutStringList(Map<String, Object> layout, String key) {
        if (layout == null) {
            return List.of();
        }
        return RuntimeModelPackage.stringList(layout.get(key));
    }

    private static boolean hasHelpArgument(String[] args) {
        for (String arg : args) {
            if ("--help".equals(arg) || "-h".equals(arg)) {
                return true;
            }
        }
        return false;
    }

    private static Path parseInspectPath(String[] args) {
        Path inspect = null;
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if ("--inspect".equals(arg)) {
                inspect = Path.of(requireValue(args, ++i, arg));
            }
        }
        return inspect;
    }

    private static CompileOptions parseCompileOptions(String[] args) {
        Path modelDir = null;
        Path output = null;
        boolean payload = true;
        boolean allowImportOnly = false;
        boolean dryRun = false;
        boolean force = false;

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            switch (arg) {
                case "--model-dir" -> modelDir = Path.of(requireValue(args, ++i, arg));
                case "--output", "-o" -> output = Path.of(requireValue(args, ++i, arg));
                case "--manifest-only" -> payload = false;
                case "--allow-import-only" -> allowImportOnly = true;
                case "--dry-run" -> dryRun = true;
                case "--force" -> force = true;
                default -> {
                    if (arg.startsWith("-")) {
                        throw new IllegalArgumentException("Unknown argument: " + arg);
                    }
                    if (modelDir != null) {
                        throw new IllegalArgumentException("Multiple model directories supplied: " + modelDir + " and " + arg);
                    }
                    modelDir = Path.of(arg);
                }
            }
        }
        if (modelDir == null) {
            throw new IllegalArgumentException("Missing --model-dir <directory>");
        }
        return new CompileOptions(modelDir, output, payload, allowImportOnly, dryRun, force);
    }

    private static String requireValue(String[] args, int index, String option) {
        if (index >= args.length || args[index].startsWith("-")) {
            throw new IllegalArgumentException("Missing value for " + option);
        }
        return args[index];
    }

    private static void printCompileResult(PrintStream out, CompileResult result) {
        if (result.dryRun()) {
            out.println("Dry run: no Qwen wdmlpack was written");
            out.println("Target Qwen wdmlpack: " + result.output());
        } else {
            out.println("Wrote Qwen wdmlpack: " + result.output());
        }
        printInspection(out, result.inspection());
    }

    private static void printInspection(PrintStream out, PackageInspection inspection) {
        out.println("  payloadIncluded=" + yesNo(inspection.payloadIncluded()));
        out.println("  runtimeLoadable=" + yesNo(inspection.runtimeLoadable()));
        out.println("  runtimeLoadMode=" + printable(inspection.runtimeLoadMode()));
        out.println("  layoutComplete=" + yesNo(inspection.layoutComplete()));
        out.println("  tensors=" + inspection.tensorCount());
        out.println("  roles=" + inspection.roleCount());
        out.println("  payload=" + QwenWdmlPackCompiler.formatBytes(inspection.payloadBytes()));
        out.println("  reasons:");
        for (String reason : inspection.runtimeLoadabilityReasons()) {
            out.println("    - " + reason);
        }
    }

    private static String yesNo(boolean value) {
        return value ? "yes" : "no";
    }

    private static String printable(String value) {
        return value == null || value.isBlank() ? "<none>" : value;
    }

    private static void printUsage(PrintStream out) {
        out.println("Usage:");
        out.println("  java ... com.aresstack.windirectml.inference.qwen.QwenWdmlPackCompileTool \\");
        out.println("      --model-dir <hf-qwen-safetensors-dir> [--output <model.wdmlpack>] [--dry-run] [--force] [--allow-import-only]");
        out.println("  java ... com.aresstack.windirectml.inference.qwen.QwenWdmlPackCompileTool \\");
        out.println("      --inspect <model.wdmlpack>");
        out.println();
        out.println("Options:");
        out.println("  --model-dir <dir>       Hugging Face Qwen2 directory containing config.json and *.safetensors");
        out.println("  --output, -o <file>     Target .wdmlpack file; defaults to <model-dir>/model.wdmlpack");
        out.println("  --dry-run               Analyze layout completeness and runtime loadability without writing");
        out.println("  --force                 Overwrite an existing target package");
        out.println("  --manifest-only         Write only an analysis manifest; requires --allow-import-only");
        out.println("  --allow-import-only     Allow writing a package that the runtime must not load yet");
        out.println("  --inspect <file>        Inspect an existing .wdmlpack manifest");
    }
}

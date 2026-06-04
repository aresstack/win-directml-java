package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.RuntimeModelPackage;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;

/**
 * Explicit Qwen SafeTensors-to-wdmlpack compiler entry point.
 *
 * <p>This tool is deliberately separate from the workbench/runtime startup path.
 * It is intended for build-time or installation-time conversion of a Hugging
 * Face Qwen2 SafeTensors directory into the internal {@code .wdmlpack} package
 * format. Normal Qwen startup still prefers the existing wdmlpack payload cache
 * and only falls back to ONNX when the cache is missing or stale.</p>
 */
public final class QwenWdmlPackCompileTool {

    private static final String DEFAULT_OUTPUT_NAME = "model.wdmlpack";
    private static final String SAFETENSORS_SOURCE_NAME = "model.safetensors";

    private QwenWdmlPackCompileTool() {
    }

    public record CompileOptions(Path modelDir,
                                 Path output,
                                 boolean payload,
                                 boolean allowImportOnly) {
        public CompileOptions {
            Objects.requireNonNull(modelDir, "modelDir");
        }
    }

    public record CompileResult(Path output,
                                boolean payloadIncluded,
                                boolean runtimeLoadable,
                                String runtimeLoadMode,
                                int tensorCount,
                                long payloadBytes) {
    }

    public static void main(String[] args) {
        int exitCode = run(args, System.out, System.err);
        if (exitCode != 0) {
            System.exit(exitCode);
        }
    }

    static int run(String[] args, PrintStream out, PrintStream err) {
        try {
            CompileOptions options = parse(args);
            if (options == null) {
                printUsage(out);
                return 0;
            }
            CompileResult result = compileSafeTensorsDirectory(options);
            out.println("Wrote Qwen wdmlpack: " + result.output());
            out.println("  payloadIncluded=" + result.payloadIncluded());
            out.println("  runtimeLoadable=" + result.runtimeLoadable());
            out.println("  runtimeLoadMode=" + result.runtimeLoadMode());
            out.println("  tensors=" + result.tensorCount());
            out.println("  payload=" + QwenWdmlPackCompiler.formatBytes(result.payloadBytes()));
            return 0;
        } catch (Exception e) {
            err.println("Could not compile Qwen SafeTensors directory to wdmlpack: " + e.getMessage());
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

        if (!options.allowImportOnly()) {
            validateRuntimePackageRequest(options.payload(), layout);
        }

        Path output = options.output() == null
                ? modelDir.resolve(DEFAULT_OUTPUT_NAME).toAbsolutePath().normalize()
                : options.output().toAbsolutePath().normalize();

        QwenWdmlPackCompiler.compileToPackage(
                imported, config, modelDir, SAFETENSORS_SOURCE_NAME, output, options.payload());

        RuntimeModelPackage modelPackage = RuntimeModelPackage.open(output);
        Map<String, Object> catalog = modelPackage.requireMap("tensorCatalog");
        int tensorCount = RuntimeModelPackage.intValue(catalog.get("count"), 0);
        long payloadBytes = RuntimeModelPackage.longValue(modelPackage.manifest().get("payloadBytes"), 0L);
        return new CompileResult(output,
                modelPackage.payloadIncluded(),
                modelPackage.runtimeLoadable(),
                modelPackage.runtimeLoadMode(),
                tensorCount,
                payloadBytes);
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

    private static CompileOptions parse(String[] args) {
        Path modelDir = null;
        Path output = null;
        boolean payload = true;
        boolean allowImportOnly = false;

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            switch (arg) {
                case "--help", "-h" -> {
                    return null;
                }
                case "--model-dir" -> modelDir = Path.of(requireValue(args, ++i, arg));
                case "--output", "-o" -> output = Path.of(requireValue(args, ++i, arg));
                case "--manifest-only" -> payload = false;
                case "--allow-import-only" -> allowImportOnly = true;
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
        return new CompileOptions(modelDir, output, payload, allowImportOnly);
    }

    private static String requireValue(String[] args, int index, String option) {
        if (index >= args.length || args[index].startsWith("-")) {
            throw new IllegalArgumentException("Missing value for " + option);
        }
        return args[index];
    }

    private static void printUsage(PrintStream out) {
        out.println("Usage:");
        out.println("  java ... com.aresstack.windirectml.inference.qwen.QwenWdmlPackCompileTool \\");
        out.println("      --model-dir <hf-qwen-safetensors-dir> [--output <model.wdmlpack>] [--allow-import-only]");
        out.println();
        out.println("Options:");
        out.println("  --model-dir <dir>       Hugging Face Qwen2 directory containing config.json and *.safetensors");
        out.println("  --output, -o <file>     Target .wdmlpack file; defaults to <model-dir>/model.wdmlpack");
        out.println("  --manifest-only         Write only an analysis manifest; requires --allow-import-only");
        out.println("  --allow-import-only     Allow writing a package that the runtime must not load yet");
    }
}

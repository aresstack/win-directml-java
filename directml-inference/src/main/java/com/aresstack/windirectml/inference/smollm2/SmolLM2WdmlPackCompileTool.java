package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.RuntimeModelPackage;
import com.aresstack.windirectml.inference.model.WdmlPackManifest;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * SmolLM2 model-directory analysis and manifest-only wdmlpack compile tool.
 */
public final class SmolLM2WdmlPackCompileTool {

    private SmolLM2WdmlPackCompileTool() {
    }

    public static void main(String[] args) {
        int exitCode = run(args, System.out, System.err);
        if (exitCode != 0) {
            System.exit(exitCode);
        }
    }

    static int run(String[] args, PrintStream out, PrintStream err) {
        try {
            if (has(args, "--help")) {
                printHelp(out);
                return 0;
            }
            Path inspectPath = optionPath(args, "--inspect");
            if (inspectPath != null) {
                inspect(inspectPath, out);
                return 0;
            }
            SmolLM2CompileOptions options = parseCompileOptions(args);
            SmolLM2CompileReport report = new SmolLM2WdmlPackCompiler().compile(options);
            printReport(out, report);
            return 0;
        } catch (Exception e) {
            err.println("SmolLM2 wdmlpack tool failed: " + e.getMessage());
            return 2;
        }
    }

    private static SmolLM2CompileOptions parseCompileOptions(String[] args) throws IOException {
        Path modelDir = optionPath(args, "--model-dir");
        if (modelDir == null) {
            throw new IOException("missing --model-dir");
        }
        Path output = optionPath(args, "--output");
        boolean dryRun = has(args, "--dry-run");
        boolean force = has(args, "--force");
        validateOptions(args);
        if (output != null && Files.exists(output) && !force && !dryRun) {
            throw new IOException("output already exists: " + output + " (use --force to overwrite)");
        }
        return new SmolLM2CompileOptions(modelDir, output, dryRun, force);
    }

    private static void validateOptions(String[] args) throws IOException {
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            switch (arg) {
                case "--model-dir", "--output", "--inspect" -> i++;
                case "--dry-run", "--force", "--help" -> {
                }
                default -> throw new IOException("invalid option combination: unknown option " + arg);
            }
        }
    }

    private static void inspect(Path packagePath, PrintStream out) throws IOException {
        RuntimeModelPackage modelPackage = RuntimeModelPackage.open(packagePath);
        Map<String, Object> manifest = modelPackage.manifest();
        out.println("modelFamily=" + value(manifest.get("modelFamily")));
        out.println("architecture=" + value(manifest.get("architecture")));
        out.println("payloadIncluded=" + yesNo(modelPackage.payloadIncluded()));
        out.println("runtimeLoadable=" + yesNo(modelPackage.runtimeLoadable()));
        out.println("runtimeLoadMode=" + modelPackage.runtimeLoadMode());
        out.println("layoutComplete=" + yesNo(Boolean.TRUE.equals(manifest.get("layoutComplete"))));
    }

    private static void printReport(PrintStream out, SmolLM2CompileReport report) {
        SmolLM2LayoutReport layout = report.layoutReport();
        out.println("modelFamily=smollm2");
        out.println("modelType=" + report.config().modelType());
        out.println("architecture=llama-causal-decoder");
        out.println("layoutComplete=" + yesNo(layout.layoutComplete()));
        out.println("runtimeLoadable=no");
        out.println("runtimeLoadableReason=" + report.runtimeLoadableReason());
        out.println("payloadIncluded=" + yesNo(report.payloadIncluded()));
        out.println("tensors=" + layout.foundTensorCount());
        out.println("roles=" + layout.knownTensorCount());
        out.println("missingRequired=" + layout.missingRequiredRoles());
        out.println("shapeErrors=" + layout.shapeErrors());
        out.println("output=" + report.output());
        out.println("dryRun=" + yesNo(report.dryRun()));
    }

    private static void printHelp(PrintStream out) {
        out.println("SmolLM2 wdmlpack compile tool");
        out.println("  --model-dir <dir>     Hugging Face SmolLM2 model directory");
        out.println("  --output <file>       Output wdmlpack file, default model.wdmlpack");
        out.println("  --dry-run             Analyze only; do not write output");
        out.println("  --inspect <file>      Inspect an existing wdmlpack manifest");
        out.println("  --force               Overwrite an existing output file");
        out.println("  --help                Print this help");
    }

    private static Path optionPath(String[] args, String option) throws IOException {
        for (int i = 0; i < args.length; i++) {
            if (option.equals(args[i])) {
                if (i + 1 >= args.length || args[i + 1].startsWith("--")) {
                    throw new IOException("missing value for " + option);
                }
                return Path.of(args[i + 1]).toAbsolutePath().normalize();
            }
        }
        return null;
    }

    private static boolean has(String[] args, String option) {
        for (String arg : args) {
            if (option.equals(arg)) {
                return true;
            }
        }
        return false;
    }

    private static String yesNo(boolean value) {
        return value ? "yes" : "no";
    }

    private static String value(Object value) {
        return value == null ? "" : String.valueOf(value);
    }
}

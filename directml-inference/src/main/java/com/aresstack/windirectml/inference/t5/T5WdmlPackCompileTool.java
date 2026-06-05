package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.WdmlPackWriter;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.Map;

/**
 * T5 SafeTensors-to-wdmlpack compile and inspection entry point.
 */
public final class T5WdmlPackCompileTool {
    private T5WdmlPackCompileTool() {
    }

    public static void main(String[] args) {
        int exitCode = run(args, System.out, System.err);
        if (exitCode != 0) {
            System.exit(exitCode);
        }
    }

    static int run(String[] args, PrintStream out, PrintStream err) {
        try {
            if (args.length == 0 || has(args, "--help")) {
                printUsage(out);
                return args.length == 0 ? 2 : 0;
            }
            if (has(args, "--inspect")) {
                Path inspectPath = requiredPathAfter(args, "--inspect");
                inspect(inspectPath, out);
                return 0;
            }
            T5CompileOptions options = parseCompileOptions(args);
            T5WdmlPackCompiler.T5CompileResult result = T5WdmlPackCompiler.compile(options);
            new T5CompileReportPrinter().printCompileResult(out, result);
            return 0;
        } catch (Exception e) {
            err.println("T5 wdmlpack tool failed: " + e.getMessage());
            return 2;
        }
    }

    static void inspect(Path packagePath, PrintStream out) throws IOException {
        Map<String, Object> manifest = WdmlPackWriter.readManifest(packagePath);
        T5RuntimePackage.validateT5Manifest(manifest);
        new T5CompileReportPrinter().printInspection(out, packagePath.toAbsolutePath().normalize(), manifest);
    }

    private static T5CompileOptions parseCompileOptions(String[] args) {
        boolean dryRun = has(args, "--dry-run");
        boolean force = has(args, "--force");
        java.util.List<String> positional = new java.util.ArrayList<>();
        for (String arg : args) {
            if (arg.startsWith("--")) {
                continue;
            }
            positional.add(arg);
        }
        if (positional.isEmpty()) {
            throw new IllegalArgumentException("Missing <hf-model-dir>");
        }
        Path modelDir = Path.of(positional.get(0));
        Path output = positional.size() > 1 ? Path.of(positional.get(1)) : null;
        return new T5CompileOptions(modelDir, output, dryRun, force);
    }

    private static boolean has(String[] args, String flag) {
        for (String arg : args) {
            if (flag.equals(arg)) {
                return true;
            }
        }
        return false;
    }

    private static Path requiredPathAfter(String[] args, String flag) {
        for (int i = 0; i < args.length; i++) {
            if (flag.equals(args[i])) {
                if (i + 1 >= args.length) {
                    throw new IllegalArgumentException("Missing path after " + flag);
                }
                return Path.of(args[i + 1]);
            }
        }
        throw new IllegalArgumentException("Missing " + flag);
    }

    private static void printUsage(PrintStream out) {
        out.println("Usage:");
        out.println("  T5WdmlPackCompileTool --dry-run <hf-model-dir>");
        out.println("  T5WdmlPackCompileTool --inspect <model.wdmlpack>");
        out.println("  T5WdmlPackCompileTool [--force] <hf-model-dir> <output.wdmlpack>");
    }
}

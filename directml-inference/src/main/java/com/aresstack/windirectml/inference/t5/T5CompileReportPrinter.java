package com.aresstack.windirectml.inference.t5;

import java.io.PrintStream;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Prints operator-facing T5 compile and inspection reports.
 */
final class T5CompileReportPrinter {

    void printCompileResult(PrintStream out, T5WdmlPackCompiler.T5CompileResult result) {
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(result, "result");
        out.println(result.written() ? "T5 wdmlpack written: " + result.output() : "T5 wdmlpack dry-run: " + result.output());
        printLayout(out, result.layout());
    }

    void printInspection(PrintStream out, Path packagePath, Map<String, Object> manifest) {
        Objects.requireNonNull(out, "out");
        out.println("T5 wdmlpack manifest: " + packagePath);
        out.println("modelFamily=" + value(manifest, "modelFamily"));
        out.println("sourceFormat=" + value(manifest, "sourceFormat"));
        out.println("runtimeLoadable=" + yesNo(Boolean.TRUE.equals(manifest.get("runtimeLoadable"))));
        out.println("runtimeLoadMode=" + value(manifest, "runtimeLoadMode"));
        out.println("reason: " + value(manifest, "reason"));
        Object layout = manifest.get("layout");
        if (layout instanceof Map<?, ?> raw) {
            printLayoutMap(out, raw);
        }
    }

    void printLayout(PrintStream out, T5LayoutManifest layout) {
        out.println("modelFamily=t5");
        out.println("sourceFormat=safetensors");
        out.println("layoutComplete=" + yesNo(layout.complete()));
        out.println("runtimeLoadable=no");
        out.println("runtimeLoadMode=" + layout.runtimeLoadMode());
        out.println("tensors=" + layout.tensorCount());
        out.println("roles=" + layout.roleCount());
        out.println("payloadBytes=" + layout.payloadBytes());
        out.println("missingRequired=" + layout.missingRequired().size());
        out.println("shapeErrors=" + layout.shapeErrors().size());
        out.println("unsupportedRuntimeDtypes=" + layout.unsupportedRuntimeDtypes().size());
        out.println("reason: " + layout.reason());
        for (String missing : layout.missingRequired()) {
            out.println("missing: " + missing);
        }
        for (String shapeError : layout.shapeErrors()) {
            out.println("shape: " + shapeError);
        }
        for (String dtype : layout.unsupportedRuntimeDtypes()) {
            out.println("dtype: " + dtype);
        }
    }

    private void printLayoutMap(PrintStream out, Map<?, ?> layout) {
        out.println("layoutComplete=" + yesNo(Boolean.TRUE.equals(layout.get("complete"))));
        out.println("roles=" + number(layout.get("roleCount")));
        out.println("tensors=" + number(layout.get("tensorCount")));
        out.println("payloadBytes=" + number(layout.get("payloadBytes")));
        printList(out, "missing", layout.get("missingRequired"));
        printList(out, "shape", layout.get("shapeErrors"));
        printList(out, "dtype", layout.get("unsupportedRuntimeDtypes"));
    }

    private static void printList(PrintStream out, String prefix, Object value) {
        if (!(value instanceof List<?> list)) {
            return;
        }
        for (Object item : list) {
            out.println(prefix + ": " + item);
        }
    }

    private static String value(Map<String, Object> manifest, String key) {
        Object value = manifest.get(key);
        return value == null ? "" : String.valueOf(value);
    }

    private static String number(Object value) {
        return value == null ? "0" : String.valueOf(value);
    }

    private static String yesNo(boolean value) {
        return value ? "yes" : "no";
    }
}

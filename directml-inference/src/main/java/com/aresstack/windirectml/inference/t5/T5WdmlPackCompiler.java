package com.aresstack.windirectml.inference.t5;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Compiles a Hugging Face T5/CodeT5 SafeTensors directory into a payload-backed wdmlpack.
 */
public final class T5WdmlPackCompiler {
    private T5WdmlPackCompiler() {
    }

    public static T5CompileResult compile(T5CompileOptions options) throws IOException {
        Objects.requireNonNull(options, "options");
        Path modelDir = options.modelDir().toAbsolutePath().normalize();
        if (!Files.isDirectory(modelDir)) {
            throw new IOException("T5 model directory not found: " + modelDir);
        }
        T5Config config = T5Config.load(modelDir.resolve("config.json"));
        T5ModelImport imported = T5ModelImport.loadSafeTensorsDirectory(modelDir);
        T5LayoutManifest layout = T5SafeTensorsLayoutCompiler.analyze(imported, config);
        Path output = options.resolveOutput();

        if (options.dryRun()) {
            return new T5CompileResult(output, false, layout, null);
        }
        if (Files.exists(output) && !options.force()) {
            throw new IOException("T5 wdmlpack already exists; pass --force to replace it: " + output);
        }
        new T5WdmlPackManifestWriter().writeWithPayload(output, modelDir, imported, config, layout);
        return new T5CompileResult(output, true, layout, T5RuntimePackage.open(output));
    }

    public record T5CompileResult(Path output,
                           boolean written,
                           T5LayoutManifest layout,
                           T5RuntimePackage runtimePackage) {
    }
}

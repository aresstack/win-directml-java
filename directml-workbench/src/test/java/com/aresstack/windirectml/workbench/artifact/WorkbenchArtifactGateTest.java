package com.aresstack.windirectml.workbench.artifact;

import com.aresstack.windirectml.inference.artifact.ModelFamily;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The runtime-tab gate: encoder/reranker have no wdmlpack compiler, so the gate must fail fast
 * (not-executable) and never let a tab fall through to a direct SafeTensors/ONNX load, and never write
 * anything. Phi-3 is now compiler-backed (PHI3-WORKBENCH-RUNNABLE-1) but, with no package present, still
 * fails fast -- now with the "Missing ... runtime package -> Convert" hint rather than "compiler not implemented".
 */
class WorkbenchArtifactGateTest {

    @TempDir
    Path tempDir;

    @Test
    void embeddingAndRerankerFailFastWithMissingPackageAndWriteNothing() throws IOException {
        // Raw files present, but no wdmlpack yet: must fail fast with the Convert hint, never load raw.
        Files.writeString(tempDir.resolve("config.json"), "{}");
        Files.writeString(tempDir.resolve("tokenizer.json"), "{}");
        Files.writeString(tempDir.resolve("model.safetensors"), "x");

        for (ModelFamily family : new ModelFamily[]{ModelFamily.EMBEDDING, ModelFamily.RERANKER}) {
            IllegalStateException error = assertThrows(IllegalStateException.class,
                    () -> WorkbenchArtifactGate.requireExecutablePackage(family, tempDir),
                    family + " must fail fast");
            assertTrue(error.getMessage().contains("Use Download tab -> Check, then Convert"),
                    family + " message: " + error.getMessage());
            assertFalse(WorkbenchArtifactGate.inspect(family, tempDir).ready(), family + " must not be ready");
        }

        // Phi-3 is compiler-backed now, but with no model_phi3.wdmlpack present it still fails fast (not executable)
        // and points at Convert -- it must never run from raw ONNX.
        IllegalStateException phi3 = assertThrows(IllegalStateException.class,
                () -> WorkbenchArtifactGate.requireExecutablePackage(ModelFamily.PHI3, tempDir));
        assertTrue(phi3.getMessage().contains("Use Download tab -> Check, then Convert"), phi3.getMessage());
        assertFalse(WorkbenchArtifactGate.inspect(ModelFamily.PHI3, tempDir).ready(), "Phi-3 must not be ready");

        try (var stream = Files.list(tempDir)) {
            assertFalse(stream.anyMatch(p -> p.getFileName().toString().endsWith(".wdmlpack")),
                    "the gate must never write a package");
        }
    }
}

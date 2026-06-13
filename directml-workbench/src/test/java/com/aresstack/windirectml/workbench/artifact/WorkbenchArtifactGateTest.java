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
 * The runtime-tab gate: encoder/reranker/Phi-3 have no wdmlpack compiler yet, so the gate must fail
 * fast (not-executable) and never let a tab fall through to a direct SafeTensors/ONNX load, and never
 * write anything.
 */
class WorkbenchArtifactGateTest {

    @TempDir
    Path tempDir;

    @Test
    void compilerMissingFamiliesFailFastWithActionableMessageAndWriteNothing() throws IOException {
        // Even with raw model files present, these families are not executable until a compiler exists.
        Files.writeString(tempDir.resolve("config.json"), "{}");
        Files.writeString(tempDir.resolve("tokenizer.json"), "{}");
        Files.writeString(tempDir.resolve("model.safetensors"), "x");

        for (ModelFamily family : new ModelFamily[]{ModelFamily.EMBEDDING, ModelFamily.RERANKER, ModelFamily.PHI3}) {
            IllegalStateException error = assertThrows(IllegalStateException.class,
                    () -> WorkbenchArtifactGate.requireExecutablePackage(family, tempDir),
                    family + " must fail fast");
            assertTrue(error.getMessage().contains("Package compiler not implemented"),
                    family + " message: " + error.getMessage());
            assertFalse(WorkbenchArtifactGate.inspect(family, tempDir).ready(), family + " must not be ready");
        }

        try (var stream = Files.list(tempDir)) {
            assertFalse(stream.anyMatch(p -> p.getFileName().toString().endsWith(".wdmlpack")),
                    "the gate must never write a package");
        }
    }
}

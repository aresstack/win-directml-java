package com.aresstack.windirectml.inference.smollm2;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SmolLM2RuntimeTest {

    @TempDir
    Path tempDir;

    @Test
    void runtimePackageAcceptsSmolLM2PackageMetadata() throws Exception {
        SmolLM2RuntimePackage runtimePackage = createRuntimePackage();

        assertFalse(runtimePackage.executable());
        assertEquals(SmolLM2LayoutReport.RUNTIME_NOT_IMPLEMENTED, runtimePackage.runtimeLoadableReason());
    }

    @Test
    void loadCreatesRuntimeShell() throws Exception {
        try (SmolLM2Runtime runtime = SmolLM2Runtime.load(createRuntimePackage())) {
            assertNotNull(runtime.runtimePackage());
        }
    }

    @Test
    void generateRequiresTokenizer() throws Exception {
        try (SmolLM2Runtime runtime = SmolLM2Runtime.load(createRuntimePackage())) {
            SmolLM2RuntimeUnsupportedException exception = assertThrows(SmolLM2RuntimeUnsupportedException.class,
                    () -> runtime.generate(new SmolLM2RuntimeRequest("hello", 4, null)));
            assertTrue(exception.getMessage().contains("requires a SmolLM2Tokenizer"));
        }
    }

    @Test
    void generateUsesTokenizerAndReferenceRuntimeWhenWeightsAreAvailable() throws Exception {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);
        SmolLM2TestFixtures.writeModelDirectory(tempDir, config, true);
        Path packagePath = tempDir.resolve("runtime-with-weights.wdmlpack");
        new SmolLM2WdmlPackCompiler().compile(new SmolLM2CompileOptions(tempDir, packagePath, false, false));
        Path tokenizerJson = tempDir.resolve("tokenizer.json");
        SmolLM2TestFixtures.writeTokenizerJson(tokenizerJson);

        try (SmolLM2Runtime runtime = SmolLM2Runtime.load(
                SmolLM2RuntimePackage.open(packagePath), SmolLM2Tokenizer.load(tokenizerJson))) {
            SmolLM2RuntimeResult result = runtime.generate(new SmolLM2RuntimeRequest("a", 1, null));

            assertEquals("a", result.generatedText());
            assertEquals(List.of(0), result.generatedTokenIds());
            assertEquals(1, result.tokensGenerated());
            assertEquals("length", result.finishReason());
        }
    }

    @Test
    void closeIsIdempotent() throws Exception {
        SmolLM2Runtime runtime = SmolLM2Runtime.load(createRuntimePackage());
        runtime.close();
        runtime.close();
    }

    @Test
    void requestRejectsNullPrompt() {
        assertThrows(IllegalArgumentException.class, () -> new SmolLM2RuntimeRequest(null, 1, null));
    }

    @Test
    void requestRejectsNonPositiveMaxNewTokens() {
        assertThrows(IllegalArgumentException.class, () -> new SmolLM2RuntimeRequest("x", 0, null));
    }

    @Test
    void stopTokenPolicyStopsOnEosToken() {
        SmolLM2StopTokenPolicy policy = new SmolLM2StopTokenPolicy(2);

        assertTrue(policy.shouldStop(2));
        assertFalse(policy.shouldStop(1));
    }

    private SmolLM2RuntimePackage createRuntimePackage() throws Exception {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);
        SmolLM2LayoutReport report = new SmolLM2LayoutValidator().validate(
                config, SmolLM2TestFixtures.completeCatalog(config, true));
        Path output = tempDir.resolve("runtime.wdmlpack");
        new SmolLM2WdmlPackManifestWriter().writeManifestOnly(output, config, report);
        return SmolLM2RuntimePackage.open(output);
    }
}

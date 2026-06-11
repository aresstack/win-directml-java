package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.prompt.PromptInput;
import com.aresstack.windirectml.inference.prompt.RawPromptStrategy;
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
        assertEquals(SmolLM2LayoutReport.MISSING_DENSE_PAYLOAD, runtimePackage.runtimeLoadableReason());
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
                    () -> runtime.generate(new SmolLM2RuntimeRequest(PromptInput.raw("hello"), 4, null)));
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

        try (SmolLM2Runtime runtime = SmolLM2Runtime.loadReference(
                SmolLM2RuntimePackage.open(packagePath), SmolLM2Tokenizer.load(tokenizerJson),
                RawPromptStrategy.INSTANCE)) {
            SmolLM2RuntimeResult result = runtime.generate(new SmolLM2RuntimeRequest(PromptInput.raw("a"), 1, null));

            assertEquals("a", result.generatedText());
            assertEquals(List.of(0), result.generatedTokenIds());
            assertEquals(1, result.tokensGenerated());
            assertEquals("length", result.finishReason());
            assertTrue(result.diagnostics().inputTokenCount() > 0);
            assertEquals(1, result.diagnostics().outputTokenCount());
            assertEquals(List.of(0), result.diagnostics().generatedTokenIds());
            assertFalse(result.diagnostics().immediateEos());
        }
    }

    @Test
    void diagnosticsPreviewLimitsGeneratedTokenIds() {
        SmolLM2GenerationDiagnostics diagnostics = new SmolLM2GenerationDiagnostics(
                3, 4, 7, 8, List.of(1, 2, 3, 4), "length", false, false);

        assertEquals("[1, 2, ...]", diagnostics.generatedTokenIdsPreview(2));
        assertEquals("[]", diagnostics.generatedTokenIdsPreview(0));
    }

    @Test
    void tokenGenerationRejectsManifestOnlyPackage() throws Exception {
        try (SmolLM2Runtime runtime = SmolLM2Runtime.load(createRuntimePackage())) {
            SmolLM2RuntimeUnsupportedException exception = assertThrows(SmolLM2RuntimeUnsupportedException.class,
                    () -> runtime.generateTokenIds(new SmolLM2TokenRuntimeRequest(List.of(0), 1, null)));
            assertEquals(SmolLM2LayoutReport.MISSING_DENSE_PAYLOAD, exception.getMessage());
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
        assertThrows(IllegalArgumentException.class, () -> new SmolLM2RuntimeRequest(PromptInput.raw("x"), 0, null));
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

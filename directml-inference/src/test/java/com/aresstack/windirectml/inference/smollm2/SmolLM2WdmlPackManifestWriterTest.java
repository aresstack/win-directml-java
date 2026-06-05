package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.RuntimeModelPackage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SmolLM2WdmlPackManifestWriterTest {

    @TempDir
    Path tempDir;

    @Test
    void writesModelFamilyArchitectureAndRuntimeFlag() throws Exception {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);
        SmolLM2LayoutReport report = new SmolLM2LayoutValidator().validate(
                config, SmolLM2TestFixtures.completeCatalog(config, true));
        Path output = tempDir.resolve("smol.wdmlpack");

        new SmolLM2WdmlPackManifestWriter().writeManifestOnly(output, config, report);

        RuntimeModelPackage modelPackage = RuntimeModelPackage.open(output);
        Map<String, Object> manifest = modelPackage.manifest();
        assertEquals("smollm2", manifest.get("modelFamily"));
        assertEquals("llama-causal-decoder", manifest.get("architecture"));
        assertFalse(modelPackage.runtimeLoadable());
        assertEquals("manifest-only", modelPackage.runtimeLoadMode());
        assertEquals(SmolLM2LayoutReport.MISSING_DENSE_PAYLOAD, manifest.get("runtimeLoadableReason"));
        assertEquals(true, manifest.get("layoutComplete"));
    }

    @Test
    void writesDensePayloadAsReferenceRuntimeLoadable() throws Exception {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);
        SmolLM2TestFixtures.writeModelDirectory(tempDir, config, true);
        var catalog = new SmolLM2ModelDirectory(tempDir).readTensorCatalog();
        SmolLM2LayoutReport report = new SmolLM2LayoutValidator().validate(config, catalog);
        Path output = tempDir.resolve("smol-reference.wdmlpack");

        new SmolLM2WdmlPackManifestWriter().writeWithDensePayload(output, config, report, catalog);

        RuntimeModelPackage modelPackage = RuntimeModelPackage.open(output);
        Map<String, Object> manifest = modelPackage.manifest();
        assertTrue(modelPackage.runtimeLoadable());
        assertEquals("reference", modelPackage.runtimeLoadMode());
        assertEquals(SmolLM2LayoutReport.REFERENCE_RUNTIME_AVAILABLE, manifest.get("runtimeLoadableReason"));
        assertEquals(true, manifest.get("layoutComplete"));
    }

    @Test
    void writesLayoutDiagnostics() throws Exception {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);
        SmolLM2LayoutReport report = new SmolLM2LayoutValidator().validate(
                config, SmolLM2TestFixtures.completeCatalog(config, false));
        Path output = tempDir.resolve("diagnostics.wdmlpack");

        new SmolLM2WdmlPackManifestWriter().writeManifestOnly(output, config, report);

        RuntimeModelPackage modelPackage = RuntimeModelPackage.open(output);
        @SuppressWarnings("unchecked")
        Map<String, Object> layout = (Map<String, Object>) modelPackage.manifest().get("smollm2Layout");
        assertEquals(false, layout.get("layoutComplete"));
        assertTrue(layout.toString().contains("LM_HEAD"));
    }
}

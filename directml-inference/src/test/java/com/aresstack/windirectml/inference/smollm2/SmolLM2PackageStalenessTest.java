package com.aresstack.windirectml.inference.smollm2;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies the version/source metadata that lets the workbench reject a stale
 * {@code model.wdmlpack} and rebuild it on first use.
 */
class SmolLM2PackageStalenessTest {

    @TempDir
    Path tempDir;

    @Test
    void compiledPackageStampsCurrentVersionsAndSourceFingerprint() throws Exception {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);
        SmolLM2TestFixtures.writeModelDirectory(tempDir, config, true);
        Path output = tempDir.resolve("model.wdmlpack");

        new SmolLM2WdmlPackCompiler().compile(new SmolLM2CompileOptions(tempDir, output, false, true));

        SmolLM2RuntimePackage pkg = SmolLM2RuntimePackage.open(output);
        assertTrue(pkg.executable());
        assertEquals(SmolLM2WdmlPackManifest.COMPILER_VERSION, pkg.compilerVersion());
        assertEquals(SmolLM2WdmlPackManifest.SCHEMA_VERSION, pkg.schemaVersion());

        Optional<String> stored = pkg.sourceFingerprint();
        assertTrue(stored.isPresent(), "compiled package should carry a source fingerprint");
        assertEquals(new SmolLM2ModelDirectory(tempDir).sourceAggregate().fingerprint(), stored.get());
    }

    @Test
    void sourceFingerprintChangesWhenSafeTensorsChange() throws Exception {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);
        SmolLM2TestFixtures.writeModelDirectory(tempDir, config, true);

        String before = new SmolLM2ModelDirectory(tempDir).sourceAggregate().fingerprint();

        // Re-write the shard with a different size so the cheap fingerprint changes.
        Path shard = tempDir.resolve("model.safetensors");
        byte[] grown = new byte[(int) Files.size(shard) + 16];
        System.arraycopy(Files.readAllBytes(shard), 0, grown, 0, (int) Files.size(shard));
        Files.write(shard, grown);

        String after = new SmolLM2ModelDirectory(tempDir).sourceAggregate().fingerprint();
        assertNotEquals(before, after);
    }
}

package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.WdmlPackWriter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class T5RuntimeTest {

    @TempDir
    Path tempDir;

    @Test
    void loadsT5RuntimePackageMetadata() throws Exception {
        Path pack = writePack("t5", "encoder-decoder");

        T5RuntimePackage runtimePackage = T5RuntimePackage.open(pack);

        assertEquals(4, runtimePackage.metadata().dModel());
        assertFalse(runtimePackage.runtimeLoadable());
    }

    @Test
    void rejectsNonT5PackageMetadata() throws Exception {
        Path pack = writePack("qwen2", "decoder-only");

        Exception error = assertThrows(Exception.class, () -> T5RuntimePackage.open(pack));

        assertTrue(error.getMessage().contains("Not a T5"));
    }

    @Test
    void generateThrowsClearUnsupportedRuntimeException() {
        T5RuntimePackage runtimePackage = T5RuntimePackage.fromMetadata(T5PackageMetadata.from(T5TestFixtures.tinyConfig(false)));
        T5Runtime runtime = T5Runtime.load(runtimePackage);

        T5UnsupportedRuntimeException error = assertThrows(T5UnsupportedRuntimeException.class,
                () -> runtime.generate(T5RuntimeRequest.greedy(new int[]{1, 2}, 4, T5TestFixtures.tinyConfig(false).specialTokens())));

        assertEquals(T5Runtime.UNSUPPORTED_MESSAGE, error.getMessage());
    }

    private Path writePack(String family, String architecture) throws Exception {
        Path pack = tempDir.resolve(family + ".wdmlpack");
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("format", "wdmlpack");
        manifest.put("version", WdmlPackWriter.VERSION);
        manifest.put("modelFamily", family);
        manifest.put("architecture", architecture);
        manifest.put("runtimeLoadable", false);
        manifest.put("payloadIncluded", false);
        manifest.put("runtimeLoadMode", "not-implemented");
        manifest.put("t5", T5PackageMetadata.from(T5TestFixtures.tinyConfig(false)).toManifest());
        WdmlPackWriter.writeManifestOnly(pack, manifest);
        return pack;
    }
}

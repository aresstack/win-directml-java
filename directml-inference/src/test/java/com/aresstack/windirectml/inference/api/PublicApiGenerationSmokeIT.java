package com.aresstack.windirectml.inference.api;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.catalog.CatalogModelFamily;
import com.aresstack.windirectml.catalog.LocalModelCatalog;
import com.aresstack.windirectml.catalog.LocalRuntimeModelDescriptor;
import com.aresstack.windirectml.inference.artifact.Gemma3PackageLifecycle;
import com.aresstack.windirectml.inference.artifact.Phi3PackageLifecycle;
import com.aresstack.windirectml.inference.artifact.QwenPackageLifecycle;
import com.aresstack.windirectml.inference.artifact.SmolLM2PackageLifecycle;
import com.aresstack.windirectml.inference.artifact.T5PackageLifecycle;
import com.aresstack.windirectml.modelpack.ModelPackageLifecycle;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/**
 * Real-model certification smoke for the public {@link GenerationRuntime} API. Opt-in and driven by
 * system properties so the normal suite stays green offline; when requested it fails loudly rather
 * than skipping. Certifies the full production chain for one model:
 *
 * <pre>
 * download (external) -&gt; compile to *.wdmlpack (lifecycle.convert)
 *   -&gt; package-only directory WITHOUT raw weights
 *   -&gt; load via GenerationRuntime (PACKAGE_ONLY) -&gt; inference -&gt; unload -&gt; reload -&gt; inference
 * </pre>
 *
 * <p>Run one model, e.g.:
 * <pre>
 * gradlew :directml-inference:test --tests '*PublicApiGenerationSmokeIT' \
 *   -Dsmoke.repo=google-t5/t5-small -Dsmoke.rawDir=C:/Projects/smoke/raw/t5-small -Dsmoke.backend=cpu
 * </pre>
 */
class PublicApiGenerationSmokeIT {

    private static final String[] RAW_WEIGHT_SUFFIXES = {
        ".safetensors", ".onnx", ".onnx.data", ".bin", ".pt", ".pth", ".gguf"
    };

    @Test
    void certifyModelThroughPublicApi() throws IOException {
        String repo = System.getProperty("smoke.repo");
        assumeTrue(repo != null && !repo.isBlank(),
                "smoke.repo not set — opt-in real-model certification test");

        String rawDirProp = System.getProperty("smoke.rawDir");
        // When the certification is requested (smoke.repo set) a missing model must FAIL, not skip.
        assertNotNull(rawDirProp, "smoke.rawDir must be set when smoke.repo is set (no silent skip)");
        Path rawDir = Path.of(rawDirProp);
        assertTrue(Files.isDirectory(rawDir), "smoke.rawDir does not exist: " + rawDir);

        LocalRuntimeModelDescriptor descriptor = LocalModelCatalog.findByRepositoryId(repo);
        assertNotNull(descriptor, "catalog has no entry for " + repo);

        CatalogBackend backend = resolveBackend(descriptor);
        assertTrue(descriptor.supportedBackends().contains(backend),
                backend + " is not in the matrix for " + repo + ": " + descriptor.supportedBackends());

        int maxNewTokens = Integer.getInteger("smoke.maxTokens", 32);

        // 1) Compile the raw source to a runtime package (the only write path).
        ModelPackageLifecycle lifecycle = lifecycleFor(descriptor.runtimeFamily());
        lifecycle.convert(rawDir, true);
        Path compiled = lifecycle.existingPackage(rawDir)
                .orElseThrow(() -> new AssertionError("convert did not produce a package in " + rawDir));
        assertTrue(Files.isRegularFile(compiled), "compiled package missing: " + compiled);

        // 2) Build a package-only directory: the wdmlpack + tokenizer/config assets, NO raw weights.
        Path pkgDir = Files.createTempDirectory("wdml-pkgonly-" + descriptor.runtimeDirectoryName() + "-");
        copyPackageOnly(rawDir, pkgDir);
        assertTrue(Files.isRegularFile(pkgDir.resolve(descriptor.runtimePackageFileName())),
                "package-only dir is missing the wdmlpack: " + descriptor.runtimePackageFileName());
        assertNoRawWeights(pkgDir);

        String prompt = promptFor(descriptor);

        // 3) Load via the public API (PACKAGE_ONLY) — must load from the package, never raw weights.
        GenerationResult first;
        AtomicInteger streamed = new AtomicInteger();
        try (GenerationModelHandle handle =
                GenerationRuntime.load(descriptor, pkgDir, backend, LoadPolicy.PACKAGE_ONLY)) {
            assertTrue(handle.backend() == backend, "handle backend mismatch");
            first = handle.generate(GenerationRequest.builder(prompt).maxNewTokens(maxNewTokens).build(),
                    token -> {
                        assertNotNull(token);
                        streamed.incrementAndGet();
                    });
            assertGenerationSane(first, repo, backend);
            assertTrue(streamed.get() > 0, "no tokens were streamed for " + repo);
        }

        // 4) Reload from the same package-only directory and generate again (proves clean unload/reload).
        try (GenerationModelHandle reopened =
                GenerationRuntime.load(descriptor, pkgDir, backend, LoadPolicy.PACKAGE_ONLY)) {
            GenerationResult second = reopened.generate(
                    GenerationRequest.builder(prompt).maxNewTokens(maxNewTokens).build());
            assertGenerationSane(second, repo, backend);
        }

        // 5) Proof of fail-closed package-only: a directory with the assets but NO package must not load.
        Path noPkgDir = Files.createTempDirectory("wdml-nopkg-" + descriptor.runtimeDirectoryName() + "-");
        copyPackageOnly(rawDir, noPkgDir);
        Files.deleteIfExists(noPkgDir.resolve(descriptor.runtimePackageFileName()));
        GenerationException missing = assertThrows(GenerationException.class,
                () -> GenerationRuntime.load(descriptor, noPkgDir, backend, LoadPolicy.PACKAGE_ONLY));
        assertTrue(missing.errorCode() == GenerationErrorCode.PACKAGE_MISSING,
                "expected PACKAGE_MISSING, got " + missing.errorCode());
    }

    private static void assertGenerationSane(GenerationResult r, String repo, CatalogBackend backend) {
        assertNotNull(r, "null result for " + repo);
        assertNotNull(r.finishReason(), "null finishReason for " + repo);
        assertTrue(r.backend() == backend, "result backend mismatch for " + repo);
        assertTrue(r.text() != null && !r.text().trim().isEmpty(),
                "empty generated text for " + repo + " on " + backend);
        assertTrue(r.generatedTokenCount() > 0,
                "no generated tokens for " + repo + " on " + backend);
        assertFalse(r.text().contains("NaN") || r.text().contains("Infinity"),
                "output contains NaN/Infinity marker for " + repo);
    }

    private static CatalogBackend resolveBackend(LocalRuntimeModelDescriptor descriptor) {
        String prop = System.getProperty("smoke.backend");
        if (prop != null && !prop.isBlank()) {
            return CatalogBackend.valueOf(prop.trim().toUpperCase(Locale.ROOT));
        }
        // Most robust default: CPU reference where allowed, else the first supported backend.
        if (descriptor.supportedBackends().contains(CatalogBackend.CPU)) {
            return CatalogBackend.CPU;
        }
        return descriptor.supportedBackends().iterator().next();
    }

    private static ModelPackageLifecycle lifecycleFor(CatalogModelFamily family) {
        switch (family) {
            case QWEN:
                return new QwenPackageLifecycle();
            case T5:
                return new T5PackageLifecycle();
            case PHI3:
                return new Phi3PackageLifecycle();
            case SMOLLM2:
                return new SmolLM2PackageLifecycle();
            case GEMMA3:
                return new Gemma3PackageLifecycle();
            default:
                throw new AssertionError("no generation lifecycle for family " + family);
        }
    }

    private static String promptFor(LocalRuntimeModelDescriptor descriptor) {
        if (descriptor.runtimeFamily() == CatalogModelFamily.T5) {
            String repo = descriptor.huggingFaceRepositoryId().toLowerCase(Locale.ROOT);
            if (repo.contains("codet5")) {
                return "summarize: public int add(int a, int b) { return a + b; }";
            }
            return "summarize: Java is a programming language used for many applications.";
        }
        return "Respond with one short sentence: What is Java?";
    }

    private static void copyPackageOnly(Path rawDir, Path pkgDir) throws IOException {
        try (Stream<Path> entries = Files.list(rawDir)) {
            List<Path> files = new ArrayList<>();
            entries.filter(Files::isRegularFile).forEach(files::add);
            for (Path file : files) {
                if (isRawWeight(file)) {
                    continue;
                }
                Files.copy(file, pkgDir.resolve(file.getFileName().toString()));
            }
        }
    }

    private static void assertNoRawWeights(Path dir) throws IOException {
        try (Stream<Path> entries = Files.list(dir)) {
            List<String> offenders = new ArrayList<>();
            entries.filter(Files::isRegularFile)
                    .filter(PublicApiGenerationSmokeIT::isRawWeight)
                    .forEach(p -> offenders.add(p.getFileName().toString()));
            assertTrue(offenders.isEmpty(), "package-only dir must contain no raw weights, found: " + offenders);
        }
    }

    private static boolean isRawWeight(Path file) {
        String name = file.getFileName().toString().toLowerCase(Locale.ROOT);
        for (String suffix : RAW_WEIGHT_SUFFIXES) {
            if (name.endsWith(suffix)) {
                return true;
            }
        }
        return false;
    }
}

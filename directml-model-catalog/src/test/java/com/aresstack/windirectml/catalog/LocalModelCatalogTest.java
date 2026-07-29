package com.aresstack.windirectml.catalog;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the neutral catalog: the exact RUNNABLE set the host may recommend, the capability routing, and the
 * deliberate non-promotion of the unverified L-12 reranker.
 */
class LocalModelCatalogTest {

    private static Set<String> repos(List<LocalRuntimeModelDescriptor> ds) {
        Set<String> out = new HashSet<String>();
        for (LocalRuntimeModelDescriptor d : ds) {
            out.add(d.huggingFaceRepositoryId());
        }
        return out;
    }

    @Test
    void runnableSetIsExactlyTheBindingFamilies() {
        Set<String> expected = new HashSet<String>(Arrays.asList(
                "sentence-transformers/all-MiniLM-L6-v2",
                "intfloat/e5-small-v2",
                "intfloat/e5-base-v2",
                "intfloat/e5-large-v2",
                "cross-encoder/ms-marco-MiniLM-L6-v2",
                "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                "HuggingFaceTB/SmolLM2-360M-Instruct",
                "google/gemma-3-270m-it",
                "microsoft/Phi-3-mini-4k-instruct-onnx",
                "google-t5/t5-small",
                "google/flan-t5-small",
                "Salesforce/codet5-small",
                "Salesforce/codet5-base-multi-sum"));
        assertEquals(expected, repos(LocalModelCatalog.runnable()));
    }

    @Test
    void l12RerankerIsPresentButUnverifiedAndNeverRunnable() {
        LocalRuntimeModelDescriptor l12 =
                LocalModelCatalog.findByRepositoryId("cross-encoder/ms-marco-MiniLM-L12-v2");
        assertNotNull(l12);
        assertEquals(ModelStatus.UNVERIFIED, l12.status());
        assertFalse(l12.isRunnable());
        assertFalse(repos(LocalModelCatalog.runnable()).contains("cross-encoder/ms-marco-MiniLM-L12-v2"));
    }

    @Test
    void capabilityRoutingIsExact() {
        assertEquals(new HashSet<String>(Arrays.asList(
                        "sentence-transformers/all-MiniLM-L6-v2", "intfloat/e5-small-v2",
                        "intfloat/e5-base-v2", "intfloat/e5-large-v2")),
                repos(LocalModelCatalog.runnableByCapability(ModelCapability.EMBEDDING)));
        assertEquals(new HashSet<String>(Arrays.asList("cross-encoder/ms-marco-MiniLM-L6-v2")),
                repos(LocalModelCatalog.runnableByCapability(ModelCapability.RERANK)));
        // Chat/completion excludes encoders and rerankers entirely.
        Set<String> chat = repos(LocalModelCatalog.runnableByCapability(ModelCapability.CHAT));
        assertTrue(chat.contains("Qwen/Qwen2.5-Coder-0.5B-Instruct"));
        assertTrue(chat.contains("google/gemma-3-270m-it"));
        assertFalse(chat.contains("sentence-transformers/all-MiniLM-L6-v2"));
        assertFalse(chat.contains("cross-encoder/ms-marco-MiniLM-L6-v2"));
        // Seq2seq is exactly the T5 family.
        assertEquals(new HashSet<String>(Arrays.asList(
                        "google-t5/t5-small", "google/flan-t5-small",
                        "Salesforce/codet5-small", "Salesforce/codet5-base-multi-sum")),
                repos(LocalModelCatalog.runnableByCapability(ModelCapability.SEQ2SEQ)));
    }

    private static Set<CatalogBackend> backends(String repo) {
        LocalRuntimeModelDescriptor d = LocalModelCatalog.findByRepositoryId(repo);
        assertNotNull(d, repo);
        return d.supportedBackends();
    }

    @Test
    void backendMatrixMirrorsTheVerifiedRuntimeCode() {
        // Code-verified per family (QwenInferenceEngine / SmolLM2WorkbenchRuntimeRunner /
        // Phi3InferenceEngine / T5InferenceEngine / SummarizerPanel gemma path). WARP = software adapter,
        // AUTO = hardware adapter of the same DirectML path; Phi-3 deliberately has NO WARP path.
        Set<CatalogBackend> warpAutoCpu = new HashSet<CatalogBackend>(Arrays.asList(
                CatalogBackend.WARP, CatalogBackend.AUTO, CatalogBackend.CPU));
        assertEquals(warpAutoCpu, backends("Qwen/Qwen2.5-Coder-0.5B-Instruct"));
        assertEquals(warpAutoCpu, backends("HuggingFaceTB/SmolLM2-135M-Instruct"));
        assertEquals(warpAutoCpu, backends("HuggingFaceTB/SmolLM2-360M-Instruct"));
        assertEquals(warpAutoCpu, backends("google-t5/t5-small"));
        assertEquals(warpAutoCpu, backends("google/flan-t5-small"));
        assertEquals(warpAutoCpu, backends("Salesforce/codet5-small"));
        assertEquals(warpAutoCpu, backends("Salesforce/codet5-base-multi-sum"));

        assertEquals(new HashSet<CatalogBackend>(Arrays.asList(CatalogBackend.WARP, CatalogBackend.AUTO)),
                backends("google/gemma-3-270m-it"));

        // Phi-3: cpu/directml/auto — NO WARP (runtime engine has no WARP path).
        assertEquals(new HashSet<CatalogBackend>(Arrays.asList(
                        CatalogBackend.CPU, CatalogBackend.DIRECTML, CatalogBackend.AUTO)),
                backends("microsoft/Phi-3-mini-4k-instruct-onnx"));
        assertFalse(backends("microsoft/Phi-3-mini-4k-instruct-onnx").contains(CatalogBackend.WARP),
                "Phi-3 has no WARP path");
    }

    @Test
    void gemmaOffersOnlyWarpAndAutoNeverCpu() {
        LocalRuntimeModelDescriptor gemma = LocalModelCatalog.findByRepositoryId("google/gemma-3-270m-it");
        assertNotNull(gemma);
        assertTrue(gemma.supportedBackends().contains(CatalogBackend.WARP));
        assertTrue(gemma.supportedBackends().contains(CatalogBackend.AUTO));
        assertFalse(gemma.supportedBackends().contains(CatalogBackend.CPU),
                "Gemma must never offer the CPU (Python) path");
        assertTrue(gemma.gated());
    }

    @Test
    void everyRunnableEntryHasAUsableDownloadManifestAndPackageName() {
        for (LocalRuntimeModelDescriptor d : LocalModelCatalog.runnable()) {
            assertNotNull(d.downloadManifest(), d.huggingFaceRepositoryId() + " needs a manifest");
            assertFalse(d.downloadManifest().requiredFiles().isEmpty(),
                    d.huggingFaceRepositoryId() + " needs required files");
            assertTrue(d.runtimePackageFileName().endsWith(".wdmlpack"),
                    d.huggingFaceRepositoryId() + " needs a wdmlpack package name");
            assertFalse(d.packageLifecycleId().isEmpty());
            assertFalse(d.virtualModelName().isEmpty());
            assertTrue(d.virtualModelName().startsWith("local/"));
        }
    }

    /** A naive downloader resolves exactly {@code <repositoryId>/<remotePath>} — nothing more. */
    private static String resolve(DownloadManifest m, DownloadFile f) {
        return m.repositoryId() + "/" + f.remotePath();
    }

    private static DownloadFile fileByLocalName(DownloadManifest m, String localName) {
        for (DownloadFile f : m.files()) {
            if (f.localName().equals(localName)) {
                return f;
            }
        }
        throw new AssertionError("no file " + localName + " in " + m.repositoryId());
    }

    @Test
    void onnxSubdirectoriesResolveExactlyOnceAndNeverDoublePrefix() {
        DownloadManifest qwen =
                LocalModelCatalog.findByRepositoryId("Qwen/Qwen2.5-Coder-0.5B-Instruct").downloadManifest();
        assertEquals("onnx-community/Qwen2.5-Coder-0.5B-Instruct", qwen.repositoryId());
        DownloadFile qwenGraph = fileByLocalName(qwen, "model_q4f16.onnx");
        assertEquals("onnx/model_q4f16.onnx", qwenGraph.remotePath());
        assertEquals("onnx-community/Qwen2.5-Coder-0.5B-Instruct/onnx/model_q4f16.onnx",
                resolve(qwen, qwenGraph));

        // Qwen keeps config/tokenizer at the repo root (only the graph is under onnx/).
        assertEquals("config.json", fileByLocalName(qwen, "config.json").remotePath());

        DownloadManifest phi3 = LocalModelCatalog
                .findByRepositoryId("microsoft/Phi-3-mini-4k-instruct-onnx").downloadManifest();
        DownloadFile phiGraph = fileByLocalName(phi3, "model.onnx");
        assertEquals("directml/directml-int4-awq-block-128/model.onnx", phiGraph.remotePath());
        assertEquals("microsoft/Phi-3-mini-4k-instruct-onnx/"
                + "directml/directml-int4-awq-block-128/model.onnx", resolve(phi3, phiGraph));
        // Phi-3 ships config/tokenizer inside the SAME INT4 subdir — verified against ModelDownloader.
        assertEquals("directml/directml-int4-awq-block-128/config.json",
                fileByLocalName(phi3, "config.json").remotePath());

        // No file's resolved path ever contains a doubled subdirectory segment, and every local name is flat.
        for (LocalRuntimeModelDescriptor d : LocalModelCatalog.runnable()) {
            DownloadManifest m = d.downloadManifest();
            for (DownloadFile f : m.files()) {
                String resolved = resolve(m, f);
                assertFalse(resolved.contains("/onnx/onnx/"), resolved);
                assertFalse(resolved.contains("directml-int4-awq-block-128/directml-int4-awq-block-128"),
                        resolved);
                assertFalse(f.localName().contains("/"), "local name must be flat: " + f.localName());
            }
        }
    }
}

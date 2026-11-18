package com.aresstack.windirectml.inference.artifact;

import com.aresstack.windirectml.catalog.DownloadFile;
import com.aresstack.windirectml.catalog.LocalModelCatalog;
import com.aresstack.windirectml.catalog.LocalRuntimeModelDescriptor;
import com.aresstack.windirectml.inference.qwen.QwenModelDirValidator;
import com.aresstack.windirectml.modelpack.ModelArtifactStatus;
import com.aresstack.windirectml.modelpack.ModelFamily;
import com.aresstack.windirectml.modelpack.RawAssetState;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end name contract for Qwen: from the real catalog download manifest to the file the compile
 * lifecycle actually resolves. onnx-community ships a self-contained {@code model_q4f16.onnx} (INT4, no
 * external data) which compiles to {@code model_q4f16.wdmlpack} — exactly the package the runtime loads.
 * A regression here previously defaulted the compile source to {@code model.onnx}, which both missed the
 * downloaded file and (being named {@code model.onnx}) demanded a non-existent {@code model.onnx_data}.
 */
class QwenPackageLifecycleTest {

    @TempDir
    Path tempDir;

    @Test
    void isCompilerBacked() {
        assertEquals(ModelFamily.QWEN, new QwenPackageLifecycle().family());
        assertTrue(new QwenPackageLifecycle().hasCompiler());
    }

    @Test
    void catalogManifestFilesResolveToTheVariantPackageWithoutExternalData() throws IOException {
        LocalRuntimeModelDescriptor qwen =
                LocalModelCatalog.findByRepositoryId("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        assertNotNull(qwen, "the catalog must offer the runnable Qwen model");

        // Stage exactly the local files the installer would write from the catalog download manifest.
        String onnxLocalName = null;
        for (DownloadFile file : qwen.downloadManifest().files()) {
            // Non-empty placeholders: the raw inspection only checks presence + non-zero size, not content.
            Files.write(tempDir.resolve(file.localName()), new byte[]{'x'});
            if (file.localName().endsWith(".onnx")) {
                onnxLocalName = file.localName();
            }
        }
        assertEquals("model_q4f16.onnx", onnxLocalName,
                "the catalog ships the self-contained INT4 ONNX build");
        // The self-contained q4f16 build must NOT require an external-data sidecar.
        assertFalse(QwenModelDirValidator.requiresExternalDataFile(onnxLocalName));
        assertFalse(Files.exists(tempDir.resolve("model.onnx")), "no fp32 model.onnx is staged");
        assertFalse(Files.exists(tempDir.resolve("model.onnx_data")), "no external data is staged");

        QwenPackageLifecycle lifecycle = new QwenPackageLifecycle();
        // The lifecycle resolves the staged q4f16 source and targets the package the runtime loads —
        // proving the catalog's runtimePackageFileName and the compile output name are the same file.
        assertEquals(qwen.runtimePackageFileName(),
                lifecycle.defaultPackagePath(tempDir).getFileName().toString());
        assertEquals("model_q4f16.wdmlpack", qwen.runtimePackageFileName());

        // The downloaded source is seen as a complete, convertible raw asset (no model.onnx demanded).
        ModelArtifactStatus status = lifecycle.inspect(tempDir);
        assertEquals(RawAssetState.RAW_VALID, status.rawState());
    }
}

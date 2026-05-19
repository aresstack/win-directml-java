package com.aresstack.windirectml.sidecar.workbench;

import com.aresstack.windirectml.sidecar.client.EmbeddingResult;
import com.aresstack.windirectml.sidecar.client.SidecarClientConfig;
import com.aresstack.windirectml.sidecar.client.SidecarException;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Headless tests for {@link WorkbenchModel}: no Swing, no sidecar, no GPU.
 */
class WorkbenchModelTest {

    @Test
    void configCanBeSetAndRead() {
        WorkbenchModel m = new WorkbenchModel();
        SidecarClientConfig cfg = m.getConfig();
        cfg.setJavaExecutable("java");
        cfg.setSidecarJarPath("foo.jar");
        cfg.setEmbedBackend("directml");
        cfg.setDirectmlDebug(true);
        cfg.setRequestTimeoutMillis(15000L);

        assertEquals("java", m.getConfig().getJavaExecutable());
        assertEquals("foo.jar", m.getConfig().getSidecarJarPath());
        assertEquals("directml", m.getConfig().getEmbedBackend());
        assertTrue(m.getConfig().isDirectmlDebug());
        assertEquals(15000L, m.getConfig().getRequestTimeoutMillis());
        assertFalse(m.isRunning());
    }

    @Test
    void commandLinePreviewWorksBeforeStart() {
        WorkbenchModel m = new WorkbenchModel();
        m.getConfig().setSidecarJarPath("preview.jar");
        m.getConfig().setEmbedBackend("auto");
        List<String> cmd = m.getCommandLine();
        assertNotNull(cmd);
        assertTrue(cmd.contains("--enable-preview"));
        assertTrue(cmd.contains("-Dembed.backend=auto"));
        assertEquals("preview.jar", cmd.get(cmd.size() - 1));
    }

    @Test
    void modelCallsFailLoudlyWhenSidecarNotRunning() {
        WorkbenchModel m = new WorkbenchModel();
        m.getConfig().setSidecarJarPath("foo.jar");
        assertFalse(m.isRunning());
        assertThrows(SidecarException.class, new org.junit.jupiter.api.function.Executable() {
            @Override
            public void execute() throws Throwable {
                m.health();
            }
        });
        assertThrows(SidecarException.class, new org.junit.jupiter.api.function.Executable() {
            @Override
            public void execute() throws Throwable {
                m.embed("hi");
            }
        });
        assertThrows(SidecarException.class, new org.junit.jupiter.api.function.Executable() {
            @Override
            public void execute() throws Throwable {
                m.summarize("hi", 32);
            }
        });
    }

    @Test
    void stopSidecarBeforeStartIsNoOp() {
        WorkbenchModel m = new WorkbenchModel();
        m.stopSidecar();
        assertFalse(m.isRunning());
    }

    @Test
    void cosineSimilarityRoundTripsViaEmbeddingResult() {
        float[] a = {1f, 0f, 0f};
        float[] b = {0f, 1f, 0f};
        float[] c = {1f, 0f, 0f};
        assertEquals(0.0, EmbeddingResult.cosine(a, b), 1e-6);
        assertEquals(1.0, EmbeddingResult.cosine(a, c), 1e-6);
    }
}


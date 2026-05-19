package com.aresstack.windirectml.sidecar.client;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SidecarProcessCommandLineTest {

    @Test
    void buildsBaseCommandLineWithDefaults() {
        SidecarClientConfig cfg = new SidecarClientConfig();
        cfg.setSidecarJarPath("directml-sidecar.jar");
        List<String> cmd = SidecarProcess.buildCommandLine(cfg);

        assertEquals("java", cmd.get(0));
        assertTrue(cmd.contains("--enable-preview"));
        assertTrue(cmd.contains("--enable-native-access=ALL-UNNAMED"));
        assertTrue(cmd.contains("-Dembed.backend=auto"));
        assertTrue(cmd.contains("-Dwindirectml.debug=false"));
        assertTrue(cmd.contains("-jar"));
        assertEquals("directml-sidecar.jar", cmd.get(cmd.size() - 1));
        // No DLL override by default.
        for (String s : cmd) {
            assertFalse(s.startsWith("-Dwindirectml.directml.dll="),
                    "must not include DLL override unless configured");
        }
    }

    @Test
    void buildsCommandLineWithDirectMlForcedAndDllOverride() {
        SidecarClientConfig cfg = new SidecarClientConfig();
        cfg.setJavaExecutable("C:\\\\Program Files\\\\Java\\\\jdk-21\\\\bin\\\\java.exe");
        cfg.setSidecarJarPath("build/libs/directml-sidecar-0.1.0-SNAPSHOT.jar");
        cfg.setEmbedBackend("directml");
        cfg.setDirectmlDebug(true);
        cfg.setDirectmlDllOverride("C:\\\\redist\\\\DirectML.dll");
        cfg.setModelDirectory("model/all-MiniLM-L6-v2");
        cfg.setExtraJvmArgs("-Xmx2g -XX:+UseG1GC");

        List<String> cmd = SidecarProcess.buildCommandLine(cfg);
        assertEquals("C:\\\\Program Files\\\\Java\\\\jdk-21\\\\bin\\\\java.exe", cmd.get(0));
        assertTrue(cmd.contains("-Xmx2g"));
        assertTrue(cmd.contains("-XX:+UseG1GC"));
        assertTrue(cmd.contains("-Dembed.backend=directml"));
        assertTrue(cmd.contains("-Dwindirectml.debug=true"));
        assertTrue(cmd.contains("-Dwindirectml.directml.dll=C:\\\\redist\\\\DirectML.dll"));
        assertTrue(cmd.contains("-Dminilm.modelDir=model/all-MiniLM-L6-v2"));
        assertTrue(cmd.contains("-jar"));
        assertEquals("build/libs/directml-sidecar-0.1.0-SNAPSHOT.jar", cmd.get(cmd.size() - 1));
    }

    @Test
    void rejectsStartWithoutJarPath() {
        SidecarProcess p = new SidecarProcess(new SidecarClientConfig());
        try {
            p.start();
            throw new AssertionError("expected SidecarException");
        } catch (SidecarException e) {
            assertTrue(e.getMessage().contains("sidecarJarPath"),
                    "expected error to mention sidecarJarPath, was: " + e.getMessage());
        }
    }

    @Test
    void buildsCommandLineWithMiniLmDefaults() {
        SidecarClientConfig cfg = new SidecarClientConfig();
        cfg.setSidecarJarPath("directml-sidecar.jar");
        List<String> cmd = SidecarProcess.buildCommandLine(cfg);
        // Default family is minilm, so -Dembed.model=minilm must be set.
        assertTrue(cmd.contains("-Dembed.model=minilm"),
                "default embed.model must be 'minilm', cmd=" + cmd);
        // No -De5.modelDir unless configured.
        for (String s : cmd) {
            assertFalse(s.startsWith("-De5.modelDir="),
                    "must not include -De5.modelDir unless configured");
        }
    }

    @Test
    void buildsCommandLineWithE5SelectionAndVariant() {
        SidecarClientConfig cfg = new SidecarClientConfig();
        cfg.setSidecarJarPath("directml-sidecar.jar");
        cfg.setEmbedModel("e5");
        cfg.setE5Variant("small-v2");
        cfg.setE5ModelDirectory("model/e5-small-v2");
        List<String> cmd = SidecarProcess.buildCommandLine(cfg);
        assertTrue(cmd.contains("-Dembed.model=e5"));
        assertTrue(cmd.contains("-De5.model=small-v2"));
        assertTrue(cmd.contains("-De5.modelDir=model/e5-small-v2"));
    }
}


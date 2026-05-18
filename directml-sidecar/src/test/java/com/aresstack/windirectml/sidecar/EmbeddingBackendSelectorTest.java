package com.aresstack.windirectml.sidecar;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Strukturelle Tests für {@link EmbeddingBackendSelector}: cpu erzwingen,
 * directml erzwingen, auto-Fallback bei DirectML-Fehler. Keine GPU nötig –
 * die Encoder-Loader sind Stubs.
 */
class EmbeddingBackendSelectorTest {

    private static final class StubModel implements EmbeddingModel {
        private final String tag;

        StubModel(String tag) {
            this.tag = tag;
        }

        @Override
        public boolean isReady() {
            return true;
        }

        @Override
        public int dimension() {
            return 384;
        }

        @Override
        public EmbeddingVector embed(EmbeddingRequest r) throws EmbeddingException {
            return new EmbeddingVector(new float[]{0f}, 384, tag, true);
        }
    }

    private static final Path DIR = Path.of("does/not/matter");

    @Test
    void parseAcceptsAllThreeModes() {
        assertEquals(EmbeddingBackendSelector.Mode.CPU,
                EmbeddingBackendSelector.Mode.parse("cpu"));
        assertEquals(EmbeddingBackendSelector.Mode.DIRECTML,
                EmbeddingBackendSelector.Mode.parse("directml"));
        assertEquals(EmbeddingBackendSelector.Mode.DIRECTML,
                EmbeddingBackendSelector.Mode.parse("DML"));
        assertEquals(EmbeddingBackendSelector.Mode.AUTO,
                EmbeddingBackendSelector.Mode.parse("auto"));
        assertEquals(EmbeddingBackendSelector.Mode.AUTO,
                EmbeddingBackendSelector.Mode.parse(null));
        assertEquals(EmbeddingBackendSelector.Mode.AUTO,
                EmbeddingBackendSelector.Mode.parse(""));
    }

    @Test
    void parseRejectsUnknownValue() {
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingBackendSelector.Mode.parse("rocm"));
    }

    @Test
    void cpuModeUsesCpuLoaderOnly() {
        StubModel cpu = new StubModel("cpu");
        EmbeddingBackendSelector sel = new EmbeddingBackendSelector(
                p -> cpu,
                p -> {
                    throw new AssertionError("DirectML loader must not be called in cpu mode");
                });
        EmbeddingBackendSelector.Selection s = sel.select(EmbeddingBackendSelector.Mode.CPU, DIR);
        assertSame(cpu, s.model());
        assertEquals("cpu", s.backend());
        assertFalse(s.fallback());
        assertNull(s.warning());
    }

    @Test
    void cpuModeFailsVisiblyWhenLoaderThrows() {
        EmbeddingBackendSelector sel = new EmbeddingBackendSelector(
                p -> {
                    throw new RuntimeException("no cpu model on disk");
                },
                p -> {
                    throw new AssertionError();
                });
        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> sel.select(EmbeddingBackendSelector.Mode.CPU, DIR));
        assertTrue(ex.getMessage().contains("cpu"));
    }

    @Test
    void directmlModeUsesDirectMlLoaderOnly() {
        StubModel dml = new StubModel("directml");
        EmbeddingBackendSelector sel = new EmbeddingBackendSelector(
                p -> {
                    throw new AssertionError("CPU loader must not be called in directml mode");
                },
                p -> dml);
        EmbeddingBackendSelector.Selection s = sel.select(EmbeddingBackendSelector.Mode.DIRECTML, DIR);
        assertSame(dml, s.model());
        assertEquals("directml", s.backend());
        assertFalse(s.fallback());
    }

    @Test
    void directmlModeFailsVisiblyWhenLoaderThrows() {
        EmbeddingBackendSelector sel = new EmbeddingBackendSelector(
                p -> {
                    throw new AssertionError();
                },
                p -> {
                    throw new RuntimeException("no D3D12 device");
                });
        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> sel.select(EmbeddingBackendSelector.Mode.DIRECTML, DIR));
        assertTrue(ex.getMessage().contains("directml"));
        assertTrue(ex.getMessage().contains("no D3D12 device"));
    }

    @Test
    void autoPrefersDirectMlWhenAvailable() {
        StubModel dml = new StubModel("directml");
        EmbeddingBackendSelector sel = new EmbeddingBackendSelector(
                p -> {
                    throw new AssertionError("CPU loader must not be called when DirectML works");
                },
                p -> dml);
        EmbeddingBackendSelector.Selection s = sel.select(EmbeddingBackendSelector.Mode.AUTO, DIR);
        assertEquals("directml", s.backend());
        assertFalse(s.fallback());
        assertNull(s.warning());
    }

    @Test
    void autoFallsBackToCpuOnDirectMlFailure() {
        StubModel cpu = new StubModel("cpu");
        EmbeddingBackendSelector sel = new EmbeddingBackendSelector(
                p -> cpu,
                p -> {
                    throw new RuntimeException("DirectML.dll not found");
                });
        EmbeddingBackendSelector.Selection s = sel.select(EmbeddingBackendSelector.Mode.AUTO, DIR);
        assertSame(cpu, s.model());
        assertEquals("cpu", s.backend());
        assertTrue(s.fallback());
        assertNotNull(s.warning());
        assertTrue(s.warning().contains("DirectML"));
    }

    @Test
    void autoFailsWhenBothBackendsFail() {
        EmbeddingBackendSelector sel = new EmbeddingBackendSelector(
                p -> {
                    throw new RuntimeException("no cpu model");
                },
                p -> {
                    throw new RuntimeException("no dml device");
                });
        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> sel.select(EmbeddingBackendSelector.Mode.AUTO, DIR));
        assertTrue(ex.getMessage().contains("directml"));
        assertTrue(ex.getMessage().contains("cpu"));
    }
}


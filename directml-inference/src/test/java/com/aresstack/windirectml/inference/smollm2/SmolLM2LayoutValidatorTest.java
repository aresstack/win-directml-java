package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.SourceTensor;
import com.aresstack.windirectml.inference.model.SourceTensorCatalog;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

class SmolLM2LayoutValidatorTest {

    private final SmolLM2LayoutValidator validator = new SmolLM2LayoutValidator();

    @Test
    void acceptsComplete135MStyleLayout() {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);

        SmolLM2LayoutReport report = validator.validate(config, SmolLM2TestFixtures.completeCatalog(config, true));

        assertTrue(report.layoutComplete());
        assertFalse(report.runtimeLoadable());
        assertEquals(SmolLM2LayoutReport.RUNTIME_NOT_IMPLEMENTED, report.runtimeLoadableReason());
        assertTrue(report.usesGroupedQueryAttention());
    }

    @Test
    void acceptsComplete360MStyleLayout() {
        SmolLM2Config config = SmolLM2TestFixtures.config360(false);

        SmolLM2LayoutReport report = validator.validate(config, SmolLM2TestFixtures.completeCatalog(config, true));

        assertTrue(report.layoutComplete());
        assertEquals(java.util.List.of(0), report.detectedLayers());
    }

    @Test
    void reportsMissingLayerTensor() {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);
        ArrayList<SourceTensor> tensors = new ArrayList<>(SmolLM2TestFixtures.completeCatalog(config, true).entries().values());
        tensors.removeIf(tensor -> tensor.name().equals("model.layers.0.self_attn.q_proj.weight"));

        SmolLM2LayoutReport report = validator.validate(config, new SourceTensorCatalog(tensors));

        assertFalse(report.layoutComplete());
        assertTrue(report.missingRequiredRoles().contains("LAYER_SELF_Q#0"));
    }

    @Test
    void reportsShapeMismatch() {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);
        ArrayList<SourceTensor> tensors = new ArrayList<>(SmolLM2TestFixtures.completeCatalog(config, true).entries().values());
        tensors.removeIf(tensor -> tensor.name().equals("model.embed_tokens.weight"));
        tensors.add(SourceTensor.metadataOnly("model.embed_tokens.weight", 1, new long[]{1, 1}));

        SmolLM2LayoutReport report = validator.validate(config, new SourceTensorCatalog(tensors));

        assertFalse(report.layoutComplete());
        assertTrue(report.shapeErrors().stream().anyMatch(error -> error.contains("TOKEN_EMBEDDING")));
    }

    @Test
    void acceptsMissingLmHeadWhenEmbeddingsAreTied() {
        SmolLM2Config config = SmolLM2TestFixtures.config135(true);

        SmolLM2LayoutReport report = validator.validate(config, SmolLM2TestFixtures.completeCatalog(config, false));

        assertTrue(report.layoutComplete());
    }

    @Test
    void rejectsMissingLmHeadWhenEmbeddingsAreNotTied() {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);

        SmolLM2LayoutReport report = validator.validate(config, SmolLM2TestFixtures.completeCatalog(config, false));

        assertFalse(report.layoutComplete());
        assertTrue(report.missingRequiredRoles().contains("LM_HEAD"));
    }
}

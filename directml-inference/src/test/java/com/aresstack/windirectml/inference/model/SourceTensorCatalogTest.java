package com.aresstack.windirectml.inference.model;

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SourceTensorCatalogTest {

    @Test
    void keepsImportTensorPayloadBehindFormatNeutralCatalog() {
        ByteBuffer payload = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        payload.putInt(0x01020304);
        payload.flip();

        SourceTensor tensor = SourceTensor.inline("model.embed_tokens.weight",
                SourceTensorDataType.FLOAT16, new long[]{2, 1}, 4, payload);
        SourceTensorCatalog catalog = new SourceTensorCatalog(List.of(tensor));

        assertEquals(1, catalog.size());
        assertEquals(4, catalog.inlineBytes());
        assertEquals(0, catalog.externalBytes());
        assertEquals(SourceTensorDataType.FLOAT16, catalog.get("model.embed_tokens.weight").dataType());
        assertEquals(0x04, catalog.get("model.embed_tokens.weight").payloadBuffer().get(0));
        assertEquals(10, catalog.toTensorCatalog().get("model.embed_tokens.weight").dataType());
    }

    @Test
    void mapsSafeTensorsDTypeToRuntimePackageCode() {
        SourceTensorDataType dataType = SourceTensorDataType.fromSafeTensors("BF16", 16);

        assertEquals(16, dataType.onnxCode());
        assertEquals("BFLOAT16", dataType.name());
        assertEquals(2, dataType.bytesPerElement());
        assertFalse(dataType.isDenseRuntimeFloat());
    }
}

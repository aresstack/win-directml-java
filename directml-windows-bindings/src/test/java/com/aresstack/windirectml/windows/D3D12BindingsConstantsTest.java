package com.aresstack.windirectml.windows;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class D3D12BindingsConstantsTest {

    @Test
    void shouldUseCorrectTimestampQueryConstants() {
        assertEquals(1, D3D12Bindings.D3D12_QUERY_HEAP_TYPE_TIMESTAMP);
        assertEquals(2, D3D12Bindings.D3D12_QUERY_TYPE_TIMESTAMP);
    }
}

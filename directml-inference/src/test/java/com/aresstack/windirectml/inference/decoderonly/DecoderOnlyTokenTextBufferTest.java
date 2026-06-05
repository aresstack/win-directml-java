package com.aresstack.windirectml.inference.decoderonly;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DecoderOnlyTokenTextBufferTest {

    @Test
    void appendsTokenTextAndReportsDelta() {
        DecoderOnlyTokenTextBuffer buffer = new DecoderOnlyTokenTextBuffer(16);

        DecoderOnlyTokenTextBuffer.DecoderOnlyTokenText first = buffer.appendDecodedTokenText("Hel");
        DecoderOnlyTokenTextBuffer.DecoderOnlyTokenText second = buffer.appendDecodedTokenText("lo");

        assertEquals("Hel", first.fullText());
        assertEquals("Hel", first.delta());
        assertEquals("Hello", second.fullText());
        assertEquals("lo", second.delta());
        assertEquals("Hello", buffer.fullText());
    }

    @Test
    void preservesUtf8Text() {
        DecoderOnlyTokenTextBuffer buffer = new DecoderOnlyTokenTextBuffer(16);

        DecoderOnlyTokenTextBuffer.DecoderOnlyTokenText text = buffer.appendDecodedTokenText("Moin 😊");

        assertEquals("Moin 😊", text.fullText());
        assertEquals("Moin 😊", text.delta());
    }
}

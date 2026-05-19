package com.aresstack.windirectml.sidecar.workbench.panels;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Headless test for {@link BatchEmbeddingsPanel}'s package-private
 * helpers – the UI itself is exercised by {@code WorkbenchFrameSmokeTest}.
 */
class BatchEmbeddingsPanelTest {

    @Test
    void parseTextsSplitsLinesAndDropsBlanks() {
        List<String> got = BatchEmbeddingsPanel.parseTexts(
                "one\n  two  \n\n   \nthree\r\nfour");
        assertEquals(4, got.size());
        assertEquals("one",   got.get(0));
        assertEquals("two",   got.get(1));
        assertEquals("three", got.get(2));
        assertEquals("four",  got.get(3));
    }

    @Test
    void parseTextsHandlesNullAndEmpty() {
        assertTrue(BatchEmbeddingsPanel.parseTexts(null).isEmpty());
        assertTrue(BatchEmbeddingsPanel.parseTexts("").isEmpty());
        assertTrue(BatchEmbeddingsPanel.parseTexts("   \n  \r\n  ").isEmpty());
    }
}


package com.aresstack.windirectml.inference.piper;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class EspeakNgCliPhonemizerTest {

    @Test
    void convertsTextThroughIndependentUtf8IpaCommand() throws Exception {
        EspeakNgCliPhonemizer converter = new EspeakNgCliPhonemizer(
                Path.of("espeak-ng"), Duration.ofSeconds(5), (command, input, timeout) -> {
                    assertEquals("espeak-ng", command.get(0));
                    assertTrue(command.contains("--ipa=1"));
                    assertTrue(command.contains("--stdin"));
                    assertTrue(command.contains("-q"));
                    assertEquals("de", command.get(command.indexOf("-v") + 1));
                    assertEquals("Grüße aus Kiel!", input);
                    return "ɡʀˈyːsə\n";
                });

        assertEquals("ɡʀˈyːsə", converter.phonemize("Grüße aus Kiel!", "de"));
    }

    @Test
    void rejectsBlankTextInvalidVoicesAndEmptyProcessOutput() {
        AtomicInteger invoked = new AtomicInteger();
        EspeakNgCliPhonemizer converter = new EspeakNgCliPhonemizer(
                Path.of("espeak-ng"), Duration.ofSeconds(3), (command, input, timeout) -> {
                    invoked.incrementAndGet();
                    return " ";
                });

        assertThrows(IllegalArgumentException.class, () -> converter.phonemize("", "de"));
        assertThrows(IllegalArgumentException.class,
                () -> converter.phonemize("Hallo", "de;rm -rf ."));
        assertThrows(IOException.class, () -> converter.phonemize("Hallo", "de"));
        assertEquals(1, invoked.get());
    }

    @Test
    void propagatesProcessErrorsInsteadOfReturningMisleadingPhonemes() {
        EspeakNgCliPhonemizer converter = new EspeakNgCliPhonemizer(
                Path.of("espeak-ng"), Duration.ofSeconds(3), (command, input, timeout) -> {
                    throw new IOException("Voice not installed");
                });

        IOException failure = assertThrows(IOException.class,
                () -> converter.phonemize("Hallo", "de"));
        assertTrue(failure.getMessage().contains("Voice not installed"));
    }
}

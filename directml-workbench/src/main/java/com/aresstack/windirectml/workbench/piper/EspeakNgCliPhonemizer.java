package com.aresstack.windirectml.workbench.piper;

import com.aresstack.windirectml.inference.piper.PiperPhonemizer;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Convert text to IPA using an independently installed eSpeak-NG executable.
 * Keep GPL-licensed phonemizer binaries and data outside the inference JAR.
 */
public final class EspeakNgCliPhonemizer implements PiperPhonemizer {

    private final Path executable;
    private final Duration timeout;
    private final CommandRunner runner;

    public EspeakNgCliPhonemizer(Path executable) {
        this(executable, Duration.ofSeconds(20), new EspeakNgProcessRunner());
    }

    EspeakNgCliPhonemizer(Path executable, Duration timeout, CommandRunner runner) {
        this.executable = Objects.requireNonNull(executable, "executable");
        this.timeout = Objects.requireNonNull(timeout, "timeout");
        this.runner = Objects.requireNonNull(runner, "runner");
        if (timeout.isZero() || timeout.isNegative()) {
            throw new IllegalArgumentException("eSpeak-NG timeout must be positive");
        }
    }

    @Override
    public String phonemize(String text, String espeakVoice) throws Exception {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(espeakVoice, "espeakVoice");
        if (text.isBlank()) {
            throw new IllegalArgumentException("Cannot phonemize blank text");
        }
        if (!espeakVoice.matches("[A-Za-z0-9_+.-]{1,40}")) {
            throw new IllegalArgumentException("Invalid eSpeak-NG voice name");
        }
        List<String> command = new ArrayList<>();
        command.add(executable.toString());
        command.add("-q");
        command.add("--ipa=1");
        command.add("-b");
        command.add("1");
        command.add("-v");
        command.add(espeakVoice);
        command.add("--stdin");

        String output = runner.run(List.copyOf(command), text, timeout);
        if (output == null || output.isBlank()) {
            throw new IOException("eSpeak-NG did not return IPA phonemes");
        }
        return output.strip().replaceAll("[\\r\\n]+", " ");
    }

    @FunctionalInterface
    interface CommandRunner {
        String run(List<String> command, String input, Duration timeout) throws Exception;
    }
}

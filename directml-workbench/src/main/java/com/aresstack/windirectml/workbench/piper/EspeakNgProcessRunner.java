package com.aresstack.windirectml.workbench.piper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/** Run the optional eSpeak-NG phoneme conversion with bounded waiting and no shell. */
final class EspeakNgProcessRunner implements EspeakNgCliPhonemizer.CommandRunner {

    @Override
    public String run(List<String> command, String input, Duration timeout) throws IOException {
        Process process = new ProcessBuilder(command)
                .redirectErrorStream(true).start();
        ExecutorService outputReader = Executors.newSingleThreadExecutor(job -> {
            Thread worker = new Thread(job, "piper-espeak-output");
            worker.setDaemon(true);
            return worker;
        });
        try {
            Future<String> captured = outputReader.submit(() ->
                    new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8));
            try (java.io.OutputStream stdin = process.getOutputStream()) {
                stdin.write(input.getBytes(StandardCharsets.UTF_8));
            }
            if (!process.waitFor(timeout.toMillis(), TimeUnit.MILLISECONDS)) {
                process.destroyForcibly();
                throw new IOException("eSpeak-NG phonemization timed out");
            }
            String output = captured.get(5, TimeUnit.SECONDS);
            if (process.exitValue() != 0) {
                throw new IOException("eSpeak-NG exited with code " + process.exitValue()
                        + ": " + output.strip());
            }
            return output;
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            process.destroyForcibly();
            throw new IOException("eSpeak-NG phonemization interrupted", interrupted);
        } catch (ExecutionException | TimeoutException failure) {
            process.destroyForcibly();
            throw new IOException("Unable to capture eSpeak-NG phonemes", failure);
        } finally {
            if (process.isAlive()) {
                process.destroyForcibly();
            }
            outputReader.shutdownNow();
        }
    }
}

package com.aresstack.windirectml.workbench.whisper;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.Mixer;
import javax.sound.sampled.TargetDataLine;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/** Capture 16-bit mono PCM in a separate thread without blocking the Swing UI. */
final class MicrophoneRecorder implements AutoCloseable {
    private static final float[] SAMPLE_RATES = {16000f, 48000f, 44100f, 32000f};
    private volatile boolean recording;
    private volatile IOException captureFailure;
    private TargetDataLine line;
    private Thread captureThread;
    private ChunkedPcmAudioSource frames;
    private int sampleRate;

    static List<Device> availableDevices() {
        List<Device> found = new ArrayList<>();
        for (Mixer.Info info : AudioSystem.getMixerInfo()) {
            Mixer mixer = AudioSystem.getMixer(info);
            for (float rate : SAMPLE_RATES) {
                AudioFormat format = new AudioFormat(rate, 16, 1, true, false);
                if (mixer.isLineSupported(new DataLine.Info(TargetDataLine.class, format))) {
                    found.add(new Device(info, format));
                    break;
                }
            }
        }
        return found;
    }

    synchronized void start(Device device) throws LineUnavailableException {
        if (line != null) {
            throw new IllegalStateException("Recording is already active");
        }
        if (device == null) {
            throw new IllegalArgumentException("Select a microphone");
        }
        TargetDataLine target = (TargetDataLine) AudioSystem.getMixer(device.mixer)
                .getLine(new DataLine.Info(TargetDataLine.class, device.format));
        target.open(device.format);
        target.start();
        line = target;
        sampleRate = Math.round(device.format.getSampleRate());
        frames = new ChunkedPcmAudioSource(sampleRate);
        captureFailure = null;
        recording = true;
        captureThread = new Thread(this::record, "whisper-capture");
        captureThread.setDaemon(true);
        captureThread.start();
    }

    synchronized ChunkedPcmAudioSource stop() throws IOException {
        if (line == null) {
            throw new IllegalStateException("No microphone recording is active");
        }
        recording = false;
        line.stop();
        line.close();
        try {
            captureThread.join(3000);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while stopping microphone", ex);
        }
        if (captureThread.isAlive()) {
            throw new IOException("Microphone capture did not stop");
        }
        line = null;
        captureThread = null;
        ChunkedPcmAudioSource completed = frames;
        frames = null;
        completed.finish();
        if (captureFailure != null) {
            completed.close();
            throw captureFailure;
        }
        if (completed.isEmpty()) {
            completed.close();
            throw new IOException("No audio captured. Check Windows microphone permissions");
        }
        return completed;
    }

    private void record() {
        byte[] buffer = new byte[4096];
        try {
            while (recording) {
                int count = line.read(buffer, 0, buffer.length);
                if (count <= 0) {
                    break;
                }
                frames.append(buffer, 0, count - (count & 1));
            }
        } catch (IOException | RuntimeException ex) {
            if (recording) {
                captureFailure = new IOException("Failed to capture microphone audio", ex);
            }
        } finally {
            recording = false;
        }
    }

    @Override
    public synchronized void close() {
        recording = false;
        if (line != null) {
            line.stop();
            line.close();
        }
        if (frames != null) {
            frames.close();
            frames = null;
        }
    }

    static final class Device {
        private final Mixer.Info mixer;
        private final AudioFormat format;

        Device(Mixer.Info mixer, AudioFormat format) {
            this.mixer = mixer;
            this.format = format;
        }

        @Override
        public String toString() {
            return mixer.getName() + " (" + (int) format.getSampleRate() + " Hz)";
        }
    }
}

package com.aresstack.audio.infrastructure;

import com.aresstack.audio.application.AudioCaptureException;
import com.aresstack.audio.application.AudioSource;
import com.aresstack.audio.domain.PcmAudioFormat;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.Mixer;
import javax.sound.sampled.TargetDataLine;

/**
 * Capture PCM from a microphone via Java Sound's {@link TargetDataLine}. Use the system default
 * device when no name is configured. Refuse to start when the requested format is not supported
 * instead of silently recording something else; resampling can be added later as a separate
 * processor without touching this class.
 */
public final class JavaSoundMicrophoneSource implements AudioSource {

    private final PcmAudioFormat format;
    private final String deviceName;

    private volatile TargetDataLine line;
    private volatile Thread captureThread;
    private volatile boolean running;

    public JavaSoundMicrophoneSource(PcmAudioFormat format, String deviceName) {
        this.format = format;
        this.deviceName = deviceName;
    }

    @Override
    public PcmAudioFormat getFormat() {
        return format;
    }

    @Override
    public void start(final SampleListener listener) throws AudioCaptureException {
        if (running) {
            throw new AudioCaptureException("Capture is already running.");
        }
        final AudioFormat javaFormat = new AudioFormat(
                format.getSampleRateHz(), format.getBitsPerSample(), format.getChannels(),
                true /* signed */, false /* little endian */);
        line = openLine(javaFormat);
        try {
            line.open(javaFormat);
        } catch (LineUnavailableException ex) {
            throw new AudioCaptureException("The microphone does not support " + format
                    + " or is in use by another application: " + ex.getMessage(), ex);
        }
        line.start();
        running = true;

        captureThread = new Thread(new Runnable() {
            @Override
            public void run() {
                // Read roughly 50 ms per call; the segmenter downstream cuts exact frames.
                byte[] buffer = new byte[format.samplesForMillis(50) * format.getBytesPerSample()];
                short[] samples = new short[buffer.length / 2];
                while (running) {
                    int bytesRead = line.read(buffer, 0, buffer.length);
                    if (bytesRead <= 0) {
                        continue;
                    }
                    int sampleCount = Pcm16LittleEndianCodec.decode(buffer, bytesRead - (bytesRead % 2), samples);
                    listener.onSamples(samples, sampleCount);
                }
            }
        }, "audio-dsp-capture");
        captureThread.setDaemon(true);
        captureThread.start();
    }

    @Override
    public void stop() {
        running = false;
        TargetDataLine current = line;
        if (current != null) {
            current.stop();
            current.close();
            line = null;
        }
        Thread thread = captureThread;
        if (thread != null) {
            try {
                thread.join(2000);
            } catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
            }
            captureThread = null;
        }
    }

    /** Open the line on the configured device, or on the system default when none is set. */
    private TargetDataLine openLine(AudioFormat javaFormat) throws AudioCaptureException {
        DataLine.Info info = new DataLine.Info(TargetDataLine.class, javaFormat);
        try {
            Mixer.Info device = AvailableAudioDevices.findCaptureDevice(deviceName, javaFormat);
            if (device != null) {
                return (TargetDataLine) AudioSystem.getMixer(device).getLine(info);
            }
            if (!AudioSystem.isLineSupported(info)) {
                throw new AudioCaptureException("No capture device supports " + format
                        + ". Available devices: " + AvailableAudioDevices.listCaptureDeviceNames());
            }
            return (TargetDataLine) AudioSystem.getLine(info);
        } catch (IllegalArgumentException ex) {
            throw new AudioCaptureException(ex.getMessage(), ex);
        } catch (LineUnavailableException ex) {
            throw new AudioCaptureException("Could not open the microphone: " + ex.getMessage(), ex);
        }
    }
}

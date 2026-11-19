package com.aresstack.audio.demo;

import com.aresstack.audio.application.RecordSpeechInputUseCase;
import com.aresstack.audio.application.SpeechCaptureConfiguration;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.dsp.AudioLevelMeter;
import com.aresstack.audio.infrastructure.AvailableAudioDevices;
import com.aresstack.audio.infrastructure.JavaSoundMicrophoneSource;
import com.aresstack.audio.infrastructure.WavFileAudioSink;

import java.util.List;

/**
 * Record five seconds from the default microphone, run the standard speech DSP pipeline and
 * write {@code speech-input.wav}.
 *
 * <p>Usage: {@code java com.aresstack.audio.demo.RecordSpeechDemo [--list-devices] [--device <name>]}</p>
 */
public final class RecordSpeechDemo {

    private static final long RECORD_MILLIS = 5000L;

    private RecordSpeechDemo() {
    }

    public static void main(String[] args) {
        String deviceName = null;
        for (int i = 0; i < args.length; i++) {
            if ("--list-devices".equals(args[i])) {
                listDevices();
                return;
            }
            if ("--device".equals(args[i]) && i + 1 < args.length) {
                deviceName = args[i + 1];
                i++;
            }
        }

        SpeechCaptureConfiguration configuration = SpeechCaptureConfiguration.builder()
                .deviceName(deviceName)
                .build();
        PcmAudioFormat format = new PcmAudioFormat(
                configuration.getSampleRateHz(), configuration.getChannels(),
                configuration.getBitsPerSample());

        JavaSoundMicrophoneSource source =
                new JavaSoundMicrophoneSource(format, configuration.getDeviceName());
        WavFileAudioSink sink = new WavFileAudioSink(configuration.getOutputFile());
        RecordSpeechInputUseCase useCase = new RecordSpeechInputUseCase(source, sink, configuration);

        System.out.println("Recording " + (RECORD_MILLIS / 1000) + " seconds at " + format
                + (deviceName == null ? " from the default microphone" : " from \"" + deviceName + "\"")
                + " ... speak now.");
        try {
            AudioLevelMeter meter = useCase.record(RECORD_MILLIS);
            System.out.println("Done: " + configuration.getOutputFile().getAbsolutePath());
            System.out.println("Levels: " + meter.describe());
        } catch (Exception ex) {
            System.err.println("Recording failed: " + ex.getMessage());
            System.exit(1);
        }
    }

    private static void listDevices() {
        List<String> names = AvailableAudioDevices.listCaptureDeviceNames();
        if (names.isEmpty()) {
            System.out.println("No capture devices found.");
            return;
        }
        System.out.println("Capture devices:");
        for (int i = 0; i < names.size(); i++) {
            System.out.println("  " + (i + 1) + ". " + names.get(i));
        }
    }
}

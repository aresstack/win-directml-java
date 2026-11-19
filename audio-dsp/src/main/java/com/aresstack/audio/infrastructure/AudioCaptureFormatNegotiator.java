package com.aresstack.audio.infrastructure;

import com.aresstack.audio.application.AudioCaptureException;
import com.aresstack.audio.domain.PcmAudioFormat;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.Mixer;
import javax.sound.sampled.TargetDataLine;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Pick the first capture format a device actually supports, instead of assuming the microphone can
 * deliver 16 kHz mono directly. The candidate order prefers formats close to what real microphones
 * expose (48 kHz / 44.1 kHz), which are then normalized down to 16 kHz mono later by the
 * {@code SpeechAudioNormalizer}. The support check is injected via {@link FormatSupportProbe} so the
 * negotiation logic is unit-testable without a sound card.
 */
public final class AudioCaptureFormatNegotiator {

    /** Decides whether a device can capture a given PCM format. */
    public interface FormatSupportProbe {
        boolean supports(String deviceName, PcmAudioFormat format);
    }

    private final FormatSupportProbe probe;

    public AudioCaptureFormatNegotiator(FormatSupportProbe probe) {
        if (probe == null) {
            throw new IllegalArgumentException("FormatSupportProbe must not be null.");
        }
        this.probe = probe;
    }

    /** The default candidate order: common microphone rates first, the STT target last. */
    public static List<PcmAudioFormat> defaultCandidates() {
        return new ArrayList<PcmAudioFormat>(Arrays.asList(
                new PcmAudioFormat(48000, 1, 16),
                new PcmAudioFormat(48000, 2, 16),
                new PcmAudioFormat(44100, 1, 16),
                new PcmAudioFormat(44100, 2, 16),
                new PcmAudioFormat(16000, 1, 16)));
    }

    /**
     * @param deviceName the capture device name, or {@code null}/empty for the system default
     * @param candidates the formats to try, in preference order
     * @return the first supported format
     * @throws AudioCaptureException when the device supports none of the candidates
     */
    public PcmAudioFormat negotiate(String deviceName, List<PcmAudioFormat> candidates)
            throws AudioCaptureException {
        for (int i = 0; i < candidates.size(); i++) {
            PcmAudioFormat candidate = candidates.get(i);
            if (probe.supports(deviceName, candidate)) {
                return candidate;
            }
        }
        String device = deviceName == null || deviceName.trim().isEmpty() ? "system default" : deviceName;
        throw new AudioCaptureException("No supported capture format for microphone '" + device + "'.");
    }

    /** @return a probe backed by Java Sound ({@code TargetDataLine} support on the resolved mixer). */
    public static FormatSupportProbe javaSoundProbe() {
        return new FormatSupportProbe() {
            public boolean supports(String deviceName, PcmAudioFormat format) {
                AudioFormat javaFormat = toJavaFormat(format);
                DataLine.Info info = new DataLine.Info(TargetDataLine.class, javaFormat);
                try {
                    Mixer.Info mixerInfo = AvailableAudioDevices.findCaptureDevice(deviceName, javaFormat);
                    if (mixerInfo == null) {
                        return AudioSystem.isLineSupported(info);
                    }
                    return AudioSystem.getMixer(mixerInfo).isLineSupported(info);
                } catch (IllegalArgumentException ex) {
                    return false; // device not found → treat as unsupported for this format
                }
            }
        };
    }

    /** Signed little-endian PCM, matching {@code JavaSoundMicrophoneSource}. */
    public static AudioFormat toJavaFormat(PcmAudioFormat format) {
        return new AudioFormat(format.getSampleRateHz(), format.getBitsPerSample(),
                format.getChannels(), true, false);
    }
}

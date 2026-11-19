package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Blind single-channel reverberation estimate: measure how fast the short-term energy decays after sound
 * stops. Late reverberation makes the energy fall slowly, so the median decay slope (dB per second) of the
 * signal's falling edges maps to an RT60-style reverberation time and a normalized strength. Analysis only —
 * the audio is never changed. No loudspeaker reference is needed.
 */
public final class RoomReverbAnalyzer {

    private static final double FULL_SCALE = 32768.0d;
    private static final double EPS = 1.0e-9d;

    private final double frameDurationMs;
    private final double minDecayDb;
    private final double maxReverbSeconds;

    public RoomReverbAnalyzer(double frameDurationMs, double minDecayDb, double maxReverbSeconds) {
        this.frameDurationMs = Math.max(5.0d, frameDurationMs);
        this.minDecayDb = Math.max(1.0d, minDecayDb);
        this.maxReverbSeconds = Math.max(0.1d, maxReverbSeconds);
    }

    public RoomProfile analyze(short[] samples, int count, PcmAudioFormat format) {
        int channels = Math.max(1, format.getChannels());
        int rate = format.getSampleRateHz();
        int frames = count / channels;
        int frameLen = Math.max(1, (int) Math.round(frameDurationMs * rate / 1000.0d));
        int frameCount = frames / frameLen;
        if (frameCount < 4) {
            return new RoomProfile(rate, 0.0d, 0.0d, 0.0d);
        }
        double[] db = new double[frameCount];
        double floor = Double.POSITIVE_INFINITY;
        double peak = Double.NEGATIVE_INFINITY;
        for (int f = 0; f < frameCount; f++) {
            double sum = 0.0d;
            int start = f * frameLen;
            for (int i = 0; i < frameLen; i++) {
                double mono = 0.0d;
                int base = (start + i) * channels;
                for (int c = 0; c < channels; c++) {
                    mono += samples[base + c];
                }
                mono /= channels;
                sum += mono * mono;
            }
            double rms = Math.sqrt(sum / frameLen);
            db[f] = 20.0d * Math.log10(Math.max(rms, EPS) / FULL_SCALE);
            floor = Math.min(floor, db[f]);
            peak = Math.max(peak, db[f]);
        }
        double frameSeconds = (double) frameLen / rate;
        double activeThreshold = floor + Math.min(minDecayDb, (peak - floor) * 0.3d);

        List<Double> slopes = new ArrayList<Double>();
        int f = 1;
        while (f < frameCount) {
            if (db[f] < db[f - 1] && db[f - 1] > activeThreshold) {
                int startFrame = f - 1;
                int runStart = f;
                while (f < frameCount && db[f] < db[f - 1] && db[f] > floor + 1.0d) {
                    f++;
                }
                double drop = db[startFrame] - db[f - 1];
                int durationFrames = (f - 1) - startFrame;
                if (drop >= minDecayDb && durationFrames >= 1) {
                    double slope = drop / (durationFrames * frameSeconds); // dB per second, positive
                    if (slope > EPS) {
                        slopes.add(slope);
                    }
                }
                if (f == runStart) {
                    f++; // the frame right after the peak is already at the floor: guarantee progress
                }
            } else {
                f++;
            }
        }
        if (slopes.isEmpty()) {
            return new RoomProfile(rate, 0.0d, 0.0d, 0.0d);
        }
        Collections.sort(slopes);
        double medianSlope = slopes.get(slopes.size() / 2);
        double rt60 = Math.min(maxReverbSeconds, 60.0d / medianSlope);
        double strength = clamp01(rt60 / 0.8d);
        double confidence = clamp01(slopes.size() / 8.0d);
        return new RoomProfile(rate, rt60, strength, confidence);
    }

    private static double clamp01(double value) {
        if (Double.isNaN(value) || value < 0.0d) {
            return 0.0d;
        }
        return value > 1.0d ? 1.0d : value;
    }
}

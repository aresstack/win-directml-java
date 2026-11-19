package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Estimate a {@link NoiseProfile} from interleaved PCM16 by averaging the Hann-windowed STFT magnitude of
 * the frames that are treated as noise. When a {@link SpeechGate} is supplied, only frames whose centre is
 * outside detected speech contribute; otherwise every frame contributes (explicit "learn from silence").
 * The same window/frame size the suppressor uses keeps the magnitudes directly comparable.
 */
public final class NoiseProfileEstimator {

    private final int fftSize;
    private final int hop;
    private final double[] window;
    private final FourierTransform fft;

    public NoiseProfileEstimator(int fftSize, int hop) {
        this.fftSize = fftSize;
        this.hop = hop;
        this.window = WindowFunctions.hann(fftSize);
        this.fft = new CommonsMathFourierTransform();
    }

    /**
     * @param gate     speech gate that returns true inside speech (frames inside speech are skipped), or null
     *                 to treat the whole signal as noise
     * @param maxFrames cap on the number of contributing frames (per channel), or 0 for no cap
     * @return the averaged noise profile, or null when no frame qualified
     */
    public NoiseProfile estimate(short[] samples, int count, PcmAudioFormat format, SpeechGate gate,
                                 int maxFrames) {
        int channels = Math.max(1, format.getChannels());
        int frames = count / channels;
        if (frames < fftSize) {
            // Too short for a meaningful estimate: fall back to a single centred frame if we can.
            if (frames <= 0) {
                return null;
            }
        }
        int bins = fftSize / 2 + 1;
        double[] sum = new double[bins];
        long observed = 0;
        long total = 0;
        double[] real = new double[fftSize];
        double[] imag = new double[fftSize];
        double[] mono = new double[frames];
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < frames; i++) {
                mono[i] = samples[i * channels + c];
            }
            for (int start = 0; start + fftSize <= frames; start += hop) {
                total++;
                int centre = start + fftSize / 2;
                if (gate != null && gate.isSpeech(centre)) {
                    continue;
                }
                if (maxFrames > 0 && observed >= (long) maxFrames * channels) {
                    break;
                }
                for (int i = 0; i < fftSize; i++) {
                    real[i] = mono[start + i] * window[i];
                    imag[i] = 0.0d;
                }
                fft.forward(real, imag);
                for (int k = 0; k < bins; k++) {
                    sum[k] += Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
                }
                observed++;
            }
        }
        if (observed == 0) {
            return null;
        }
        for (int k = 0; k < bins; k++) {
            sum[k] /= observed;
        }
        double confidence = total == 0 ? 0.0d : (double) observed / total;
        return new NoiseProfile(format.getSampleRateHz(), fftSize, sum, confidence);
    }
}

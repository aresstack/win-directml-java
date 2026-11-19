package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Single-channel weighted-prediction-error (WPE) dereverberation in the STFT domain. For each frequency bin
 * it estimates the late reverberation as a weighted linear prediction from delayed past frames and subtracts
 * it, iterating with power-based weights. A prediction delay preserves early reflections; the strength blends
 * between the observed and dereverberated bin; an artifact floor keeps a minimum fraction of the original
 * magnitude to avoid over-processing; optional speech protection reduces the strength inside detected speech.
 *
 * <p>Offline mode processes the whole signal at once. Block-adaptive mode re-estimates the filters per block
 * of frames (carrying the power estimate forward). One instance per run; no state leaks between recordings.</p>
 */
public final class WpeDereverberation {

    private static final int FRAME = 512;
    private static final int HOP = 128;
    private static final double EPS = 1.0e-9d;

    private final WpeDereverberationSettings settings;

    public WpeDereverberation(WpeDereverberationSettings settings) {
        if (settings == null) {
            throw new IllegalArgumentException("WPE settings must not be null.");
        }
        this.settings = settings;
    }

    public void process(short[] samples, int count, PcmAudioFormat format, SpeechGate gate) {
        int channels = Math.max(1, format.getChannels());
        int frames = count / channels;
        if (frames < FRAME) {
            return; // too short to dereverberate meaningfully
        }
        SpeechGate speechGate = gate == null ? SpeechGate.NEVER : gate;
        double[] mono = new double[frames];
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < frames; i++) {
                mono[i] = samples[i * channels + c];
            }
            ShortTimeSpectrogram spec = ShortTimeSpectrogram.forward(mono, FRAME, HOP);
            dereverberate(spec, speechGate);
            double[] out = spec.inverse();
            for (int i = 0; i < frames; i++) {
                samples[i * channels + c] = clamp(out[i]);
            }
        }
    }

    private void dereverberate(ShortTimeSpectrogram spec, SpeechGate gate) {
        int frameCount = spec.getFrameCount();
        int half = FRAME / 2;
        int delay = settings.getEffectivePredictionDelay();
        int taps = settings.getFilterLength();
        int iterations = settings.getIterations();
        double strength = settings.getStrength();
        double floorFrac = 0.02d + 0.5d * settings.getArtifactProtection();
        int blockFrames = settings.getMode() == WpeDereverberationSettings.Mode.OFFLINE
                ? frameCount : Math.min(frameCount, settings.getBlockSizeFrames());

        boolean offline = settings.getMode() == WpeDereverberationSettings.Mode.OFFLINE;
        boolean streaming = settings.getMode() == WpeDereverberationSettings.Mode.STREAMING;
        double adapt = settings.getAdaptationSpeed();
        int history = blockFrames;
        int lookAhead = streaming ? Math.min(8, Math.max(1, blockFrames / 2)) : 0;
        int step = streaming ? Math.max(1, blockFrames / 4) : blockFrames;

        double[] xr = new double[frameCount];
        double[] xi = new double[frameCount];
        double[] dr = new double[frameCount];
        double[] di = new double[frameCount];
        double[] lambda = new double[frameCount];
        double[] gRe = new double[taps];
        double[] gIm = new double[taps];
        double[] carryRe = new double[taps];
        double[] carryIm = new double[taps];

        // Pre-compute per-frame speech flag once (bins share it).
        boolean[] speech = new boolean[frameCount];
        for (int t = 0; t < frameCount; t++) {
            speech[t] = settings.isSpeechProtection()
                    && gate.isSpeech(Math.max(0, t * HOP - FRAME / 2));
        }

        for (int k = 0; k <= half; k++) {
            for (int t = 0; t < frameCount; t++) {
                xr[t] = spec.realFrame(t)[k];
                xi[t] = spec.imagFrame(t)[k];
            }
            boolean haveCarry = false;
            for (int pos = 0; pos < frameCount; pos += step) {
                int applyEnd = Math.min(frameCount, pos + step);
                int estStart = streaming ? Math.max(0, pos - history) : pos;
                int estEnd = streaming ? Math.min(frameCount, applyEnd + lookAhead) : applyEnd;
                if (estimateFilter(xr, xi, estStart, estEnd, delay, taps, iterations, lambda, gRe, gIm)) {
                    if (!offline && haveCarry) {
                        for (int a = 0; a < taps; a++) {
                            gRe[a] = (1.0d - adapt) * carryRe[a] + adapt * gRe[a];
                            gIm[a] = (1.0d - adapt) * carryIm[a] + adapt * gIm[a];
                        }
                    }
                    System.arraycopy(gRe, 0, carryRe, 0, taps);
                    System.arraycopy(gIm, 0, carryIm, 0, taps);
                    haveCarry = true;
                    applyFilter(xr, xi, pos, applyEnd, delay, taps, carryRe, carryIm, dr, di);
                } else if (haveCarry) {
                    applyFilter(xr, xi, pos, applyEnd, delay, taps, carryRe, carryIm, dr, di);
                } else {
                    for (int t = pos; t < applyEnd; t++) {
                        dr[t] = xr[t];
                        di[t] = xi[t];
                    }
                }
            }
            // Blend, apply artifact floor and speech protection, write back (with conjugate mirror).
            for (int t = 0; t < frameCount; t++) {
                double s = speech[t] ? strength * 0.4d : strength;
                double outRe = (1.0d - s) * xr[t] + s * dr[t];
                double outIm = (1.0d - s) * xi[t] + s * di[t];
                double magX = Math.sqrt(xr[t] * xr[t] + xi[t] * xi[t]);
                double magOut = Math.sqrt(outRe * outRe + outIm * outIm);
                double minMag = floorFrac * magX;
                if (magOut < minMag && magOut > EPS) {
                    double scale = minMag / magOut;
                    outRe *= scale;
                    outIm *= scale;
                }
                spec.realFrame(t)[k] = outRe;
                spec.imagFrame(t)[k] = outIm;
                int mirror = FRAME - k;
                if (mirror > k && mirror < FRAME) {
                    spec.realFrame(t)[mirror] = outRe;
                    spec.imagFrame(t)[mirror] = -outIm;
                }
            }
        }
    }

    /**
     * Estimate the WPE prediction filter for one bin over the frame range {@code [from, to)} by iterative
     * reweighting. Fills {@code gRe/gIm} with the taps and returns true; returns false when the range is too
     * short or the normal equations are singular. {@code lambda} is scratch of length >= to.
     */
    private boolean estimateFilter(double[] xr, double[] xi, int from, int to, int delay, int taps,
                                   int iterations, double[] lambda, double[] gRe, double[] gIm) {
        int n = to - from;
        if (n <= delay + taps) {
            return false;
        }
        // Floor the power weights relative to the segment mean, so near-silent frames do not get an
        // astronomically large 1/lambda weight that makes the normal equations ill-conditioned.
        double meanPower = 0.0d;
        for (int t = from; t < to; t++) {
            meanPower += xr[t] * xr[t] + xi[t] * xi[t];
        }
        meanPower /= n;
        double lambdaFloor = Math.max(EPS, 1.0e-4d * meanPower);
        for (int t = from; t < to; t++) {
            lambda[t] = Math.max(lambdaFloor, xr[t] * xr[t] + xi[t] * xi[t]);
        }
        double[][] rRe = new double[taps][taps];
        double[][] rIm = new double[taps][taps];
        double[] pRe = new double[taps];
        double[] pIm = new double[taps];
        int first = from + delay + taps - 1; // first frame with a full history
        boolean solved = false;
        for (int iter = 0; iter < iterations; iter++) {
            for (int a = 0; a < taps; a++) {
                pRe[a] = 0.0d;
                pIm[a] = 0.0d;
                for (int b = 0; b < taps; b++) {
                    rRe[a][b] = 0.0d;
                    rIm[a][b] = 0.0d;
                }
            }
            for (int t = first; t < to; t++) {
                double w = 1.0d / lambda[t];
                for (int a = 0; a < taps; a++) {
                    double uaRe = xr[t - delay - a];
                    double uaIm = xi[t - delay - a];
                    pRe[a] += w * (uaRe * xr[t] + uaIm * xi[t]);
                    pIm[a] += w * (uaIm * xr[t] - uaRe * xi[t]);
                    for (int b = 0; b < taps; b++) {
                        double ubRe = xr[t - delay - b];
                        double ubIm = xi[t - delay - b];
                        rRe[a][b] += w * (uaRe * ubRe + uaIm * ubIm);
                        rIm[a][b] += w * (uaIm * ubRe - uaRe * ubIm);
                    }
                }
            }
            double load = 0.0d;
            for (int a = 0; a < taps; a++) {
                load += rRe[a][a];
            }
            load = 1.0e-6d * (load / taps + EPS);
            for (int a = 0; a < taps; a++) {
                rRe[a][a] += load;
            }
            if (!solveComplex(rRe, rIm, pRe, pIm, taps, gRe, gIm)) {
                break; // singular: keep the last good estimate (if any)
            }
            solved = true;
            // Update the power weights from the residual for the next iteration.
            for (int t = first; t < to; t++) {
                double prRe = 0.0d;
                double prIm = 0.0d;
                for (int a = 0; a < taps; a++) {
                    double uaRe = xr[t - delay - a];
                    double uaIm = xi[t - delay - a];
                    prRe += gRe[a] * uaRe + gIm[a] * uaIm;
                    prIm += gRe[a] * uaIm - gIm[a] * uaRe;
                }
                double residRe = xr[t] - prRe;
                double residIm = xi[t] - prIm;
                lambda[t] = Math.max(lambdaFloor, residRe * residRe + residIm * residIm);
            }
        }
        return solved;
    }

    /** Apply a WPE filter to one bin over {@code [from, to)}, writing the dereverberated bin to {@code dr/di}. */
    private void applyFilter(double[] xr, double[] xi, int from, int to, int delay, int taps,
                             double[] gRe, double[] gIm, double[] dr, double[] di) {
        int first = from + delay + taps - 1;
        for (int t = from; t < to; t++) {
            if (t < first) {
                dr[t] = xr[t];
                di[t] = xi[t];
                continue;
            }
            double prRe = 0.0d;
            double prIm = 0.0d;
            for (int a = 0; a < taps; a++) {
                double uaRe = xr[t - delay - a];
                double uaIm = xi[t - delay - a];
                // conj(g_a) * u_a
                prRe += gRe[a] * uaRe + gIm[a] * uaIm;
                prIm += gRe[a] * uaIm - gIm[a] * uaRe;
            }
            dr[t] = xr[t] - prRe;
            di[t] = xi[t] - prIm;
        }
    }

    /** Solve the complex linear system R g = p (R is taps×taps) with partial-pivot Gaussian elimination. */
    static boolean solveComplex(double[][] rRe, double[][] rIm, double[] pRe, double[] pIm, int taps,
                                        double[] gRe, double[] gIm) {
        double[][] aRe = new double[taps][taps];
        double[][] aIm = new double[taps][taps];
        double[] bRe = new double[taps];
        double[] bIm = new double[taps];
        for (int i = 0; i < taps; i++) {
            System.arraycopy(rRe[i], 0, aRe[i], 0, taps);
            System.arraycopy(rIm[i], 0, aIm[i], 0, taps);
            bRe[i] = pRe[i];
            bIm[i] = pIm[i];
        }
        for (int col = 0; col < taps; col++) {
            int pivot = col;
            double best = aRe[col][col] * aRe[col][col] + aIm[col][col] * aIm[col][col];
            for (int r = col + 1; r < taps; r++) {
                double mag = aRe[r][col] * aRe[r][col] + aIm[r][col] * aIm[r][col];
                if (mag > best) {
                    best = mag;
                    pivot = r;
                }
            }
            if (best < 1.0e-24d) {
                return false;
            }
            if (pivot != col) {
                double[] tr = aRe[pivot]; aRe[pivot] = aRe[col]; aRe[col] = tr;
                double[] ti = aIm[pivot]; aIm[pivot] = aIm[col]; aIm[col] = ti;
                double sbr = bRe[pivot]; bRe[pivot] = bRe[col]; bRe[col] = sbr;
                double sbi = bIm[pivot]; bIm[pivot] = bIm[col]; bIm[col] = sbi;
            }
            double dRe = aRe[col][col];
            double dIm = aIm[col][col];
            double denom = dRe * dRe + dIm * dIm;
            for (int r = col + 1; r < taps; r++) {
                // factor = a[r][col] / a[col][col]
                double fr = (aRe[r][col] * dRe + aIm[r][col] * dIm) / denom;
                double fi = (aIm[r][col] * dRe - aRe[r][col] * dIm) / denom;
                for (int c = col; c < taps; c++) {
                    double cr = aRe[col][c];
                    double ci = aIm[col][c];
                    aRe[r][c] -= fr * cr - fi * ci;
                    aIm[r][c] -= fr * ci + fi * cr;
                }
                bRe[r] -= fr * bRe[col] - fi * bIm[col];
                bIm[r] -= fr * bIm[col] + fi * bRe[col];
            }
        }
        for (int row = taps - 1; row >= 0; row--) {
            double sr = bRe[row];
            double si = bIm[row];
            for (int c = row + 1; c < taps; c++) {
                sr -= aRe[row][c] * gRe[c] - aIm[row][c] * gIm[c];
                si -= aRe[row][c] * gIm[c] + aIm[row][c] * gRe[c];
            }
            double dRe = aRe[row][row];
            double dIm = aIm[row][row];
            double denom = dRe * dRe + dIm * dIm;
            gRe[row] = (sr * dRe + si * dIm) / denom;
            gIm[row] = (si * dRe - sr * dIm) / denom;
        }
        return true;
    }

    private static short clamp(double value) {
        if (Double.isNaN(value)) {
            return 0;
        }
        if (value > Short.MAX_VALUE) {
            return Short.MAX_VALUE;
        }
        if (value < Short.MIN_VALUE) {
            return Short.MIN_VALUE;
        }
        return (short) Math.round(value);
    }
}

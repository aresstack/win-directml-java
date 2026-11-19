package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.dsp.DelayAndSumBeamformer;
import com.aresstack.audio.dsp.MicrophoneArrayProfile;
import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;
import com.aresstack.audio.profile.AudioProcessingProfile;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** Slice 11: the delay-and-sum beamformer enhances the target direction and rejects missing geometry. */
public class BeamformerTest {

    private static final int RATE = 16000;
    private static final double C = 343000.0d; // mm/s
    private static final PcmAudioFormat STEREO = new PcmAudioFormat(RATE, 2, 16);
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void enhancesASignalFromTheTargetDirectionRelativeToWrongSteering() {
        // Two mics on the x-axis, 100 mm apart; plane wave arriving from azimuth 0 (endfire).
        MicrophoneArrayProfile array = MicrophoneArrayProfile.parse("a", "test", "-50,0,0; 50,0,0");
        double[] source = noise(6000);
        double delayMic0 = -50.0d / C * RATE; // proj_0 / c * rate
        double delayMic1 = 50.0d / C * RATE;
        short[] stereo = new short[4000 * 2];
        for (int t = 0; t < 4000; t++) {
            stereo[2 * t] = clamp(interp(source, t + 1000 + delayMic0));
            stereo[2 * t + 1] = clamp(interp(source, t + 1000 + delayMic1));
        }
        short[] onTarget = DelayAndSumBeamformer.beamform(stereo, stereo.length, STEREO, array,
                0.0d, 0.0d, C, null, 1.0d);
        short[] offTarget = DelayAndSumBeamformer.beamform(stereo, stereo.length, STEREO, array,
                180.0d, 0.0d, C, null, 1.0d);
        assertTrue("on-target beam is more coherent: on=" + energy(onTarget) + " off=" + energy(offTarget),
                energy(onTarget) > energy(offTarget) * 1.2d);
    }

    @Test
    public void passesThroughWhenGeometryIsMissing() {
        short[] stereo = new short[400];
        for (int i = 0; i < stereo.length; i++) {
            stereo[i] = (short) (i % 100);
        }
        AudioBuffer out = registry.createProcessor(AudioBlockType.DELAY_AND_SUM_BEAMFORMER)
                .process(new AudioBuffer(stereo.clone(), STEREO),
                        registry.defaultDefinition(AudioBlockType.DELAY_AND_SUM_BEAMFORMER, "b"),
                        new AudioProcessingContext());
        assertEquals(2, out.getFormat().getChannels());
        assertArrayEquals(stereo, out.getSamples());
    }

    @Test
    public void validatorRejectsMissingGeometry() {
        List<AudioBlockDefinition> blocks = new ArrayList<AudioBlockDefinition>();
        blocks.add(registry.defaultDefinition(AudioBlockType.DELAY_AND_SUM_BEAMFORMER, "b"));
        AudioProcessingProfile profile = new AudioProcessingProfile("p", "P", false, blocks);
        AudioProfileValidationResult result = new AudioProfileValidator().validateResult(profile, STEREO);
        boolean hasError = false;
        for (AudioProfileValidationIssue issue : result.getIssues()) {
            if (issue.getSeverity() == AudioValidationSeverity.ERROR
                    && "micPositionsMm".equals(issue.getParameterKey())) {
                hasError = true;
            }
        }
        assertTrue("missing geometry is an error", hasError);
    }

    private static double[] noise(int n) {
        double[] s = new double[n];
        int state = 12321;
        for (int i = 0; i < n; i++) {
            state = state * 1103515245 + 12345;
            s[i] = (state >> 16) % 8000;
        }
        return s;
    }

    private static double interp(double[] s, double pos) {
        int i0 = (int) Math.floor(pos);
        double frac = pos - i0;
        double a = i0 >= 0 && i0 < s.length ? s[i0] : 0.0d;
        double b = i0 + 1 >= 0 && i0 + 1 < s.length ? s[i0 + 1] : 0.0d;
        return a * (1.0d - frac) + b * frac;
    }

    private static double energy(short[] s) {
        double sum = 0.0d;
        for (short v : s) {
            sum += (double) v * v;
        }
        return sum;
    }

    private static short clamp(double v) {
        if (v > Short.MAX_VALUE) {
            return Short.MAX_VALUE;
        }
        if (v < Short.MIN_VALUE) {
            return Short.MIN_VALUE;
        }
        return (short) Math.round(v);
    }
}

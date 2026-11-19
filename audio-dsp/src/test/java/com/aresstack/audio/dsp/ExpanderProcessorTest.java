package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/** The soft downward expander widens dynamics below the threshold, leaves loud signal alone, and stays safe. */
public class ExpanderProcessorTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);
    private static final PcmAudioFormat STEREO = new PcmAudioFormat(RATE, 2, 16);

    @Test
    public void ratioOneIsTransparent() {
        short[] input = tone(1000.0d, 4000, 8000);
        short[] output = process(settings(-45.0d, 1.0d), input, MONO, null);
        assertArrayEquals(input, output);
    }

    @Test
    public void loudSignalAboveThresholdIsLeftLargelyUnchanged() {
        short[] input = tone(1000.0d, RATE, 9000); // ~ -11 dBFS, above a -45 dB threshold
        short[] output = process(settings(-45.0d, 3.0d), input, MONO, null);
        double ratio = rmsSecondHalf(output) / rmsSecondHalf(input);
        assertTrue("above threshold should stay close to unity, got " + ratio, ratio > 0.9d);
    }

    @Test
    public void quietSignalBelowThresholdIsAttenuated() {
        short[] input = tone(1000.0d, RATE, 200); // ~ -44 dB peak, below a -30 dB threshold
        short[] output = process(settings(-30.0d, 3.0d), input, MONO, null);
        double ratio = rmsSecondHalf(output) / rmsSecondHalf(input);
        assertTrue("quiet signal should be attenuated, got " + ratio, ratio < 0.8d);
    }

    @Test
    public void higherRatioAttenuatesMore() {
        short[] input = tone(1000.0d, RATE, 200);
        double mild = rmsSecondHalf(process(settings(-30.0d, 2.0d), input, MONO, null));
        double strong = rmsSecondHalf(process(settings(-30.0d, 8.0d), input, MONO, null));
        assertTrue("higher ratio should attenuate more: mild=" + mild + " strong=" + strong, strong < mild);
    }

    @Test
    public void maximumAttenuationIsRespected() {
        short[] input = tone(1000.0d, RATE, 100);
        ExpanderSettings s = new ExpanderSettings(-20.0d, 20.0d, 0.0d, 1.0d, 20.0d, 0.0d, 6.0d, 10.0d, false, 0.5d);
        short[] output = process(s, input, MONO, null);
        // Max attenuation 6 dB → output RMS must stay above input × 10^(-6/20) minus a small margin.
        double floor = rmsSecondHalf(input) * Math.pow(10.0d, -6.0d / 20.0d) * 0.7d;
        assertTrue("attenuation is capped at the maximum", rmsSecondHalf(output) >= floor);
    }

    @Test
    public void disabledExpanderIsBitIdenticalViaRatioOne() {
        // The pipeline skips disabled blocks; at the processor level ratio 1 is the bit-identical case.
        short[] input = tone(500.0d, 2000, 300);
        assertArrayEquals(input, process(settings(-30.0d, 1.0d), input, MONO, null));
    }

    @Test
    public void silenceStaysFiniteAndDoesNotDrift() {
        short[] output = process(settings(-40.0d, 4.0d), new short[RATE], MONO, null);
        for (short sample : output) {
            assertTrue(sample == 0);
        }
    }

    @Test
    public void stereoUsesOneLinkedGainForBothChannels() {
        // Quiet on both channels → both attenuated by the same factor (image preserved).
        short[] input = stereoTone(1000.0d, RATE, 200, 200);
        short[] output = process(settings(-30.0d, 4.0d), input, STEREO, null);
        double left = channelRms(output, 0) / channelRms(input, 0);
        double right = channelRms(output, 1) / channelRms(input, 1);
        assertTrue("linked gain keeps the channel ratio equal: " + left + " vs " + right,
                Math.abs(left - right) < 0.05d);
    }

    @Test
    public void speechProtectionKeepsTheExpanderOpenOnSpeechFrames() {
        short[] input = tone(1000.0d, RATE, 200); // quiet → would normally be attenuated
        SpeechActivityTrack track = allSpeech(RATE, 1, 320);
        ExpanderSettings s = new ExpanderSettings(-30.0d, 6.0d, 0.0d, 1.0d, 50.0d, 0.0d, 24.0d, 20.0d, true, 0.5d);
        double protectedRms = rmsSecondHalf(process(s, input, MONO, track));
        double unprotectedRms = rmsSecondHalf(process(s, input, MONO, null));
        assertTrue("speech protection should attenuate less than the level-based run: prot="
                + protectedRms + " plain=" + unprotectedRms, protectedRms > unprotectedRms);
    }

    // ------------------------------------------------------------------ helpers

    private static ExpanderSettings settings(double thresholdDb, double ratio) {
        return new ExpanderSettings(thresholdDb, ratio, 0.0d, 1.0d, 50.0d, 0.0d, 40.0d, 10.0d, false, 0.5d);
    }

    private static short[] process(ExpanderSettings s, short[] input, PcmAudioFormat format,
                                   SpeechActivityTrack track) {
        short[] copy = input.clone();
        new ExpanderProcessor(s).process(copy, copy.length, format, new ExpanderState(), track);
        return copy;
    }

    private static SpeechActivityTrack allSpeech(int rate, int channels, int framePerChannel) {
        SpeechActivityTrack track = new SpeechActivityTrack(rate, channels, framePerChannel);
        for (int i = 0; i < rate / framePerChannel + 1; i++) {
            track.add(new SpeechActivityMetadata(0.95d, true, -60.0d, -10.0d));
        }
        return track;
    }

    private static short[] tone(double freq, int n, int amplitude) {
        short[] out = new short[n];
        for (int i = 0; i < n; i++) {
            out[i] = (short) Math.round(amplitude * Math.sin(2.0d * Math.PI * freq * i / RATE));
        }
        return out;
    }

    private static short[] stereoTone(double freq, int frames, int ampL, int ampR) {
        short[] out = new short[frames * 2];
        for (int i = 0; i < frames; i++) {
            out[i * 2] = (short) Math.round(ampL * Math.sin(2.0d * Math.PI * freq * i / RATE));
            out[i * 2 + 1] = (short) Math.round(ampR * Math.sin(2.0d * Math.PI * freq * i / RATE));
        }
        return out;
    }

    private static double rmsSecondHalf(short[] samples) {
        long sum = 0;
        int from = samples.length / 2;
        int count = 0;
        for (int i = from; i < samples.length; i++) {
            sum += (long) samples[i] * samples[i];
            count++;
        }
        return count == 0 ? 0.0d : Math.sqrt((double) sum / count);
    }

    private static double channelRms(short[] samples, int channel) {
        long sum = 0;
        int count = 0;
        for (int i = channel; i < samples.length; i += 2) {
            sum += (long) samples[i] * samples[i];
            count++;
        }
        return count == 0 ? 0.0d : Math.sqrt((double) sum / count);
    }
}

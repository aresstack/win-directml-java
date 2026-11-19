package com.aresstack.audio.application;

import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.infrastructure.WavFileAudioSink;
import com.aresstack.audio.infrastructure.WavFileReader;
import com.aresstack.audio.pipeline.AudioProcessingProfiles;

import org.junit.Test;

import java.io.File;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** End-to-end normalization: 48 kHz stereo raw WAV → canonical 16 kHz mono WAV. */
public class SpeechAudioNormalizerTest {

    private final SpeechAudioNormalizer normalizer = new SpeechAudioNormalizer();

    @Test
    public void normalizes48kStereoToCanonical16kMono() throws Exception {
        File raw = writeRawWav(new PcmAudioFormat(48000, 2, 16), 24000, 8000);   // 0.5 s stereo tone
        File target = File.createTempFile("askai-norm-", ".wav");
        target.deleteOnExit();

        NormalizationResult result = normalizer.normalize(raw, target);

        assertEquals(48000, result.getSourceFormat().getSampleRateHz());
        assertEquals(2, result.getSourceFormat().getChannels());
        assertEquals(16000, result.getTargetFormat().getSampleRateHz());
        assertEquals(1, result.getTargetFormat().getChannels());
        assertTrue("duration=" + result.getDurationMillis(),
                Math.abs(result.getDurationMillis() - 500L) <= 2L);
        assertTrue("peak=" + result.getPeak(), result.getPeak() > 4000);

        // The written file is really 16 kHz mono with ~1/3 the frame count.
        WavFileReader.WavData written = WavFileReader.read(target);
        assertEquals(16000, written.getFormat().getSampleRateHz());
        assertEquals(1, written.getFormat().getChannels());
        assertTrue("samples=" + written.getSamples().length,
                Math.abs(written.getSamples().length - 8000) <= 4);
    }

    @Test
    public void offProfilePreservesTheSourceFormat() throws Exception {
        // "Off" applies no DSP and no forced conversion: a 48 kHz stereo recording must stay 48 kHz stereo
        // so the audio reaches the model unaltered (only an explicit resampler/channel block may change it).
        SpeechAudioNormalizer offNormalizer = new SpeechAudioNormalizer(AudioProcessingProfiles.off());
        File raw = writeRawWav(new PcmAudioFormat(48000, 2, 16), 24000, 8000);
        File target = File.createTempFile("askai-norm-off-", ".wav");
        target.deleteOnExit();

        NormalizationResult result = offNormalizer.normalize(raw, target);

        assertEquals(48000, result.getTargetFormat().getSampleRateHz());
        assertEquals(2, result.getTargetFormat().getChannels());
        WavFileReader.WavData written = WavFileReader.read(target);
        assertEquals(48000, written.getFormat().getSampleRateHz());
        assertEquals(2, written.getFormat().getChannels());
    }

    @Test
    public void silenceYieldsNoSignalVerdict() throws Exception {
        File raw = writeRawWav(new PcmAudioFormat(44100, 1, 16), 22050, 0);       // 0.5 s silence
        File target = File.createTempFile("askai-norm-", ".wav");
        target.deleteOnExit();

        NormalizationResult result = normalizer.normalize(raw, target);
        RecordingQuality quality = RecordingQualityAnalyzer.withDefaults().analyze(
                result.getDurationMillis(), result.getOverallRms(), result.getPeak(),
                result.getClippedSamples(), result.getTotalSamples(), 0L);
        assertEquals(RecordingQuality.NO_SIGNAL, quality);
    }

    @Test
    public void clippingFractionUsesTotalSamplesNotFrames() throws Exception {
        // Every stereo sample at the 16-bit limit → all samples clipped. Total-samples and clipped-samples
        // must be in the same (interleaved) domain, so the fraction is 1.0, not 2.0. Before the fix,
        // getTotalSamples() returned the frame count (1000), not the interleaved count (2000).
        int frames = 16000; // ~333 ms at 48 kHz, above the min-duration gate
        File raw = writeConstantWav(new PcmAudioFormat(48000, 2, 16), frames, (short) 32767);
        File target = File.createTempFile("askai-norm-", ".wav");
        target.deleteOnExit();

        NormalizationResult result = normalizer.normalize(raw, target);
        assertEquals(frames * 2L, result.getTotalSamples());    // frames * 2 channels (interleaved)
        assertEquals(result.getTotalSamples(), result.getClippedSamples());
        assertEquals(RecordingQuality.CLIPPED, RecordingQualityAnalyzer.withDefaults().analyze(
                result.getDurationMillis(), result.getOverallRms(), result.getPeak(),
                result.getClippedSamples(), result.getTotalSamples(), 0L));
    }

    /** Writes a constant value on every sample (used to force full clipping). */
    private static File writeConstantWav(PcmAudioFormat format, int frames, short value) throws Exception {
        File file = File.createTempFile("askai-raw-", ".wav");
        file.deleteOnExit();
        int channels = format.getChannels();
        short[] samples = new short[frames * channels];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = value;
        }
        WavFileAudioSink sink = new WavFileAudioSink(file);
        sink.open(format);
        sink.write(samples, samples.length);
        sink.close();
        return file;
    }

    /** Writes a sine (amplitude on every channel) as a raw WAV in the given format. */
    private static File writeRawWav(PcmAudioFormat format, int frames, int amplitude) throws Exception {
        File file = File.createTempFile("askai-raw-", ".wav");
        file.deleteOnExit();
        int channels = format.getChannels();
        short[] samples = new short[frames * channels];
        for (int f = 0; f < frames; f++) {
            short value = (short) Math.round(amplitude * Math.sin(2 * Math.PI * 220 * f / format.getSampleRateHz()));
            for (int c = 0; c < channels; c++) {
                samples[f * channels + c] = value;
            }
        }
        WavFileAudioSink sink = new WavFileAudioSink(file);
        sink.open(format);
        sink.write(samples, samples.length);
        sink.close();
        return file;
    }
}

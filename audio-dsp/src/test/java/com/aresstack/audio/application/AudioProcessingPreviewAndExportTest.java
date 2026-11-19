package com.aresstack.audio.application;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.pipeline.AudioProcessingProfiles;
import com.aresstack.audio.profile.AudioProcessingProfile;
import com.aresstack.audio.infrastructure.WavFileReader;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

/** Preview uses the productive processor, leaves the source untouched, and export round-trips as WAV. */
public class AudioProcessingPreviewAndExportTest {

    @Rule
    public final TemporaryFolder folder = new TemporaryFolder();

    private final AudioProcessingPreviewService preview = new DefaultAudioProcessingPreviewService();

    @Test
    public void previewRunsTheDefaultPipelineAndLeavesTheSourceUnchanged() {
        short[] stereo = tone(48000 * 2);
        short[] original = stereo.clone();
        AudioProcessingProfile profile = AudioProcessingProfiles.defaultSpeech();

        ProcessedAudioPreview result = preview.process(
                new AudioBuffer(stereo, new PcmAudioFormat(48000, 2, 16)), profile, "src-1");

        assertEquals(16000, result.getFormat().getSampleRateHz());
        assertEquals(1, result.getFormat().getChannels());
        assertTrue(result.getDurationMillis() > 0);
        assertEquals("src-1", result.getSourceId());
        assertEquals(AudioProfileSignature.of(profile), result.getPipelineSignature());
        // The raw source buffer must not be mutated by the preview.
        assertTrue("source unchanged", Arrays.equals(original, stereo));
    }

    @Test
    public void repeatedProcessingWithFreshStateIsReproducible() {
        short[] mono = tone(16000);
        AudioProcessingProfile profile = AudioProcessingProfiles.defaultSpeech();
        short[] a = preview.process(new AudioBuffer(mono.clone(), new PcmAudioFormat(16000, 1, 16)),
                profile, "s").getSamples();
        short[] b = preview.process(new AudioBuffer(mono.clone(), new PcmAudioFormat(16000, 1, 16)),
                profile, "s").getSamples();
        assertTrue("adaptive state reset per run → identical result", Arrays.equals(a, b));
    }

    @Test
    public void signatureChangesWhenThePipelineChanges() {
        AudioProcessingProfile base = AudioProcessingProfiles.defaultSpeech();
        String baseSig = AudioProfileSignature.of(base);
        // Disable the first block → different pipeline → different signature.
        java.util.List<com.aresstack.audio.profile.AudioBlockDefinition> blocks =
                new java.util.ArrayList<com.aresstack.audio.profile.AudioBlockDefinition>(base.getBlocks());
        blocks.set(0, blocks.get(0).withEnabled(false));
        String changedSig = AudioProfileSignature.of(base.withBlocks(blocks));
        assertNotEquals(baseSig, changedSig);
    }

    @Test
    public void exportWritesAReadablePcm16WavWithTheResultFormat() throws Exception {
        ProcessedAudioPreview result = preview.process(
                new AudioBuffer(tone(32000), new PcmAudioFormat(16000, 1, 16)),
                AudioProcessingProfiles.defaultSpeech(), "s");
        File target = new File(folder.getRoot(), "processed.wav");
        new DefaultProcessedWaveExportService().export(result, target);

        assertTrue(target.isFile());
        WavFileReader.WavData reread = WavFileReader.read(target);
        assertEquals(result.getFormat().getSampleRateHz(), reread.getFormat().getSampleRateHz());
        assertEquals(result.getFormat().getChannels(), reread.getFormat().getChannels());
        assertEquals(result.getSamples().length, reread.getSamples().length);
    }

    @Test
    public void exportReplacesAnExistingTarget() throws Exception {
        File target = folder.newFile("out.wav");
        assertTrue(target.exists());
        ProcessedAudioPreview result = preview.process(
                new AudioBuffer(tone(8000), new PcmAudioFormat(16000, 1, 16)),
                AudioProcessingProfiles.defaultSpeech(), "s");
        new DefaultProcessedWaveExportService().export(result, target);
        assertFalse("no leftover temp files", hasTempSibling(target));
        assertTrue(WavFileReader.read(target).getSamples().length > 0);
    }

    private static boolean hasTempSibling(File target) {
        File[] siblings = target.getParentFile().listFiles();
        if (siblings == null) {
            return false;
        }
        for (File sibling : siblings) {
            if (sibling.getName().endsWith(".wav.tmp")) {
                return true;
            }
        }
        return false;
    }

    private static short[] tone(int length) {
        short[] samples = new short[length];
        for (int i = 0; i < length; i++) {
            samples[i] = (short) Math.round(6000.0d * Math.sin(2.0d * Math.PI * 200.0d * i / 16000.0d));
        }
        return samples;
    }
}

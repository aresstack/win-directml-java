package com.aresstack.audio.infrastructure;

import com.aresstack.audio.domain.PcmAudioFormat;

import org.junit.Test;

import java.io.File;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** The WAV reader round-trips what the sink writes: same format and samples. */
public class WavFileReaderTest {

    @Test
    public void roundTripsFormatAndSamples() throws Exception {
        File file = File.createTempFile("askai-wavreader-", ".wav");
        file.deleteOnExit();
        PcmAudioFormat format = new PcmAudioFormat(48000, 2, 16);
        short[] samples = {0, 1, -1, 32767, -32768, 12345, -12345, 7};

        WavFileAudioSink sink = new WavFileAudioSink(file);
        sink.open(format);
        sink.write(samples, samples.length);
        sink.close();

        WavFileReader.WavData data = WavFileReader.read(file);
        assertEquals(48000, data.getFormat().getSampleRateHz());
        assertEquals(2, data.getFormat().getChannels());
        assertEquals(16, data.getFormat().getBitsPerSample());
        assertArrayEquals(samples, data.getSamples());
    }
}

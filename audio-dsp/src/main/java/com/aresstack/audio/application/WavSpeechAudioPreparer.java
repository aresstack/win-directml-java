package com.aresstack.audio.application;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.infrastructure.WavFileAudioSink;

import java.io.File;
import java.io.IOException;

/**
 * Write the (already DSP-processed) audio to the STT transport WAV <b>preserving its format</b>: the
 * source sample rate and channel count are kept, only wrapped as signed 16-bit PCM WAV. No down-mix and no
 * resampling happen here — the audio reaches the model as unaltered as the pipeline left it.
 *
 * <p>Any rate/channel change must be an explicit DSP block (e.g. a resampler or channel mixer in the
 * selected profile), never a forced final conversion. A pass-through "Off" profile therefore yields a WAV
 * in the original recording format.</p>
 */
public final class WavSpeechAudioPreparer implements SpeechToTextAudioPreparer {

    public File prepare(AudioBuffer source, File targetWav) throws IOException {
        if (source == null) {
            throw new IllegalArgumentException("Source buffer must not be null.");
        }
        if (targetWav == null) {
            throw new IllegalArgumentException("Target WAV must not be null.");
        }
        WavFileAudioSink sink = new WavFileAudioSink(targetWav);
        sink.open(source.getFormat());
        try {
            sink.write(source.getSamples(), source.getSamples().length);
        } finally {
            sink.close();
        }
        return targetWav;
    }
}

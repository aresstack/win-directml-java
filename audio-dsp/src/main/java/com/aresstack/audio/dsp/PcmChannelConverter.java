package com.aresstack.audio.dsp;

/**
 * Convert interleaved multi-channel 16-bit PCM to mono by averaging the channels of each sample
 * frame. Pure and stateless. A trailing partial frame (fewer than {@code channels} samples, which
 * cannot form a complete frame) is dropped — that is at most {@code channels-1} sub-samples and
 * carries no usable audio.
 */
public final class PcmChannelConverter {

    private PcmChannelConverter() {
    }

    /**
     * @param interleaved interleaved samples (frame = one sample per channel, in channel order)
     * @param count       number of valid samples in {@code interleaved}
     * @param channels    channel count (>= 1); when 1 the input is copied through unchanged
     * @return a mono buffer with one averaged sample per complete input frame
     */
    public static short[] downmixToMono(short[] interleaved, int count, int channels) {
        if (channels <= 1) {
            short[] mono = new short[count];
            System.arraycopy(interleaved, 0, mono, 0, count);
            return mono;
        }
        int frames = count / channels;
        short[] mono = new short[frames];
        for (int frame = 0; frame < frames; frame++) {
            int base = frame * channels;
            int sum = 0;
            for (int channel = 0; channel < channels; channel++) {
                sum += interleaved[base + channel];
            }
            mono[frame] = (short) Math.round((double) sum / channels);
        }
        return mono;
    }
}

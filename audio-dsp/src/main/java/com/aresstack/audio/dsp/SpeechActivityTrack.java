package com.aresstack.audio.dsp;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The per-run sequence of {@link SpeechActivityMetadata}, one entry per analyzed frame, in processing
 * order. Produced by the voice-activity block and readable by later blocks through the processing context.
 * It is transient runtime state for a single pipeline run and is never persisted into a profile.
 */
public final class SpeechActivityTrack {

    private final List<SpeechActivityMetadata> frames = new ArrayList<SpeechActivityMetadata>();
    private final int sampleRateHz;
    private final int channels;
    private final int frameSampleCountPerChannel;

    public SpeechActivityTrack() {
        this(0, 0, 0);
    }

    /**
     * Create a track that also records the time base it was measured on, so later blocks can map a sample
     * position to its frame without guessing the framing.
     */
    public SpeechActivityTrack(int sampleRateHz, int channels, int frameSampleCountPerChannel) {
        this.sampleRateHz = sampleRateHz;
        this.channels = channels;
        this.frameSampleCountPerChannel = frameSampleCountPerChannel;
    }

    public int getSampleRateHz() {
        return sampleRateHz;
    }

    public int getChannels() {
        return channels;
    }

    public int getFrameSampleCountPerChannel() {
        return frameSampleCountPerChannel;
    }

    /** @return the number of interleaved samples per frame, or 0 when the time base is unknown. */
    public int getFrameSampleCountInterleaved() {
        return frameSampleCountPerChannel * channels;
    }

    /** @return the metadata for the frame that contains the given interleaved sample index, or null. */
    public SpeechActivityMetadata frameForInterleavedIndex(int interleavedIndex) {
        int frameInterleaved = getFrameSampleCountInterleaved();
        if (frameInterleaved <= 0 || frames.isEmpty()) {
            return null;
        }
        int frame = interleavedIndex / frameInterleaved;
        if (frame < 0) {
            frame = 0;
        }
        if (frame >= frames.size()) {
            frame = frames.size() - 1;
        }
        return frames.get(frame);
    }

    public void add(SpeechActivityMetadata metadata) {
        if (metadata != null) {
            frames.add(metadata);
        }
    }

    public List<SpeechActivityMetadata> getFrames() {
        return Collections.unmodifiableList(frames);
    }

    public int size() {
        return frames.size();
    }

    public boolean isEmpty() {
        return frames.isEmpty();
    }

    public SpeechActivityMetadata getLatest() {
        return frames.isEmpty() ? null : frames.get(frames.size() - 1);
    }

    public int activeFrameCount() {
        int count = 0;
        for (SpeechActivityMetadata frame : frames) {
            if (frame.isSpeechActive()) {
                count++;
            }
        }
        return count;
    }
}

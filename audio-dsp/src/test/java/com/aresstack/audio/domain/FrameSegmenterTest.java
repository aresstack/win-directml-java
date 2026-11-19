package com.aresstack.audio.domain;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class FrameSegmenterTest {

    @Test
    public void cutsArbitraryChunksIntoFixedFramesAndFlushesRemainder() {
        final List<short[]> frames = new ArrayList<short[]>();
        FrameSegmenter.FrameConsumer collector = new FrameSegmenter.FrameConsumer() {
            @Override
            public void onFrame(short[] samples, int count) {
                short[] copy = new short[count];
                System.arraycopy(samples, 0, copy, 0, count);
                frames.add(copy);
            }
        };

        FrameSegmenter segmenter = new FrameSegmenter(160);
        // Push 500 samples in odd chunk sizes: 3 full frames (480) plus a 20-sample remainder.
        short value = 0;
        int[] chunkSizes = {70, 250, 130, 50};
        for (int c = 0; c < chunkSizes.length; c++) {
            short[] chunk = new short[chunkSizes[c]];
            for (int i = 0; i < chunk.length; i++) {
                chunk[i] = value++;
            }
            segmenter.push(chunk, chunk.length, collector);
        }
        segmenter.flush(collector);

        assertEquals(4, frames.size());
        assertEquals(160, frames.get(0).length);
        assertEquals(160, frames.get(1).length);
        assertEquals(160, frames.get(2).length);
        assertEquals(20, frames.get(3).length);
        // Verify the sample order survived the segmentation.
        short expected = 0;
        for (int f = 0; f < frames.size(); f++) {
            short[] frame = frames.get(f);
            for (int i = 0; i < frame.length; i++) {
                assertEquals(expected++, frame[i]);
            }
        }
    }
}

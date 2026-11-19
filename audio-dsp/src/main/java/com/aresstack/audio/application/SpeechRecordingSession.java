package com.aresstack.audio.application;

import com.aresstack.audio.domain.FrameSegmenter;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.dsp.AudioProcessingPipeline;

import java.io.IOException;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Run one start/stop-controlled recording: capture from the source, cut fixed frames, process
 * them through the pipeline on a writer thread and hand them to the sink. Unlike
 * {@link AudioCaptureService} the caller decides when to stop (push-to-talk / toggle recording).
 *
 * <p>The capture thread only cuts frames and offers copies into a bounded queue; the pipeline and
 * the (possibly slow) sink run on the writer thread, so recording never stalls. Frames that do
 * not fit into the queue are dropped and counted.</p>
 */
public final class SpeechRecordingSession {

    private static final int QUEUE_CAPACITY_FRAMES = 128;
    /** Poison pill telling the writer thread that capture has finished. */
    private static final short[] END_OF_STREAM = new short[0];

    private final AudioSource source;
    private final AudioSink sink;
    private final AudioProcessingPipeline pipeline;
    private final int frameDurationMillis;

    private final AtomicLong droppedFrames = new AtomicLong();
    private BlockingQueue<short[]> queue;
    private FrameSegmenter segmenter;
    private FrameSegmenter.FrameConsumer enqueue;
    private Thread writer;
    private CountDownLatch writerDone;
    private final IOException[] writerError = new IOException[1];
    private boolean running;

    public SpeechRecordingSession(AudioSource source, AudioSink sink,
                                  AudioProcessingPipeline pipeline, int frameDurationMillis) {
        this.source = source;
        this.sink = sink;
        this.pipeline = pipeline;
        this.frameDurationMillis = frameDurationMillis;
    }

    /** Open the sink, start the writer thread and begin capturing. */
    public synchronized void start() throws AudioCaptureException, IOException {
        if (running) {
            throw new AudioCaptureException("Recording is already running.");
        }
        final PcmAudioFormat format = source.getFormat();
        segmenter = new FrameSegmenter(format.samplesForMillis(frameDurationMillis));
        queue = new ArrayBlockingQueue<short[]>(QUEUE_CAPACITY_FRAMES);
        droppedFrames.set(0);
        writerError[0] = null;

        sink.open(format);
        writerDone = new CountDownLatch(1);
        writer = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    while (true) {
                        short[] frame = queue.take();
                        if (frame == END_OF_STREAM) {
                            return;
                        }
                        pipeline.process(frame, frame.length, format);
                        sink.write(frame, frame.length);
                    }
                } catch (InterruptedException ex) {
                    Thread.currentThread().interrupt();
                } catch (IOException ex) {
                    writerError[0] = ex;
                } finally {
                    writerDone.countDown();
                }
            }
        }, "audio-dsp-session-writer");
        writer.setDaemon(true);
        writer.start();

        enqueue = new FrameSegmenter.FrameConsumer() {
            @Override
            public void onFrame(short[] samples, int count) {
                // Copy: the segmenter reuses its buffer, and the queue hands off to another thread.
                short[] copy = new short[count];
                System.arraycopy(samples, 0, copy, 0, count);
                if (!queue.offer(copy)) {
                    droppedFrames.incrementAndGet();
                }
            }
        };

        try {
            source.start(new AudioSource.SampleListener() {
                @Override
                public void onSamples(short[] samples, int count) {
                    segmenter.push(samples, count, enqueue);
                }
            });
        } catch (AudioCaptureException ex) {
            // Capture never started: unwind the writer thread and the sink.
            abortWriter();
            closeSinkQuietly();
            running = false;
            throw ex;
        }
        running = true;
    }

    /**
     * Stop capturing, drain the remaining frames and close the sink.
     *
     * @return the number of frames dropped because the sink could not keep up
     * @throws IOException when the writer failed to persist frames
     */
    public synchronized long stop() throws IOException {
        if (!running) {
            return droppedFrames.get();
        }
        running = false;
        source.stop();
        segmenter.flush(enqueue);
        enqueueEndOfStream();
        awaitWriter();
        sink.close();
        if (writerError[0] != null) {
            throw writerError[0];
        }
        return droppedFrames.get();
    }

    public boolean isRunning() {
        return running;
    }

    private void abortWriter() {
        enqueueEndOfStream();
        awaitWriter();
    }

    private void enqueueEndOfStream() {
        try {
            queue.put(END_OF_STREAM);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
    }

    private void awaitWriter() {
        try {
            writerDone.await();
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
    }

    private void closeSinkQuietly() {
        try {
            sink.close();
        } catch (IOException ignored) {
            // The start failure is the error worth reporting.
        }
    }
}

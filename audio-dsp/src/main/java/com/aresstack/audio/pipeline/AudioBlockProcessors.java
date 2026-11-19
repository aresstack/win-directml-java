package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.enhance.SpeechEnhancementBackend;
import com.aresstack.audio.enhance.SpeechEnhancementBackends;
import com.aresstack.audio.dsp.ButterworthFilterProcessor;
import com.aresstack.audio.dsp.CompressorProcessor;
import com.aresstack.audio.dsp.DcOffsetRemovalProcessor;
import com.aresstack.audio.dsp.FirLowPassProcessor;
import com.aresstack.audio.dsp.GainProcessor;
import com.aresstack.audio.dsp.HighPassFilterProcessor;
import com.aresstack.audio.dsp.HighShelfEqualizerProcessor;
import com.aresstack.audio.dsp.LimiterProcessor;
import com.aresstack.audio.dsp.LowShelfEqualizerProcessor;
import com.aresstack.audio.dsp.MasterEqualizerProcessor;
import com.aresstack.audio.dsp.ChannelAligner;
import com.aresstack.audio.dsp.ChannelDiagnostics;
import com.aresstack.audio.dsp.DelayAndSumBeamformer;
import com.aresstack.audio.dsp.DirectionEstimate;
import com.aresstack.audio.dsp.DirectionOfArrivalEstimator;
import com.aresstack.audio.dsp.DirectionTracker;
import com.aresstack.audio.dsp.MicrophoneArrayProfile;
import com.aresstack.audio.dsp.MultichannelOps;
import com.aresstack.audio.dsp.ParametricEqualizerProcessor;
import com.aresstack.audio.dsp.Pcm16Processor;
import com.aresstack.audio.dsp.Pcm16Resampler;
import com.aresstack.audio.dsp.PcmChannelConverter;
import com.aresstack.audio.dsp.ResamplingQuality;
import com.aresstack.audio.dsp.AdaptiveHumRemovalProcessor;
import com.aresstack.audio.dsp.AdaptiveHumRemovalSettings;
import com.aresstack.audio.dsp.BreathReductionProcessor;
import com.aresstack.audio.dsp.BreathReductionSettings;
import com.aresstack.audio.dsp.DeEsserProcessor;
import com.aresstack.audio.dsp.DeEsserSettings;
import com.aresstack.audio.dsp.NoiseProfile;
import com.aresstack.audio.dsp.NoiseProfileEstimator;
import com.aresstack.audio.dsp.NoiseSuppressionSettings;
import com.aresstack.audio.dsp.SpectralNoiseSuppressor;
import com.aresstack.audio.dsp.SpeechLevelerProcessor;
import com.aresstack.audio.dsp.SpeechLevelerSettings;
import com.aresstack.audio.dsp.SpeechLevelerState;
import com.aresstack.audio.dsp.StereoOps;
import com.aresstack.audio.dsp.WpeDereverberation;
import com.aresstack.audio.dsp.WpeDereverberationSettings;
import com.aresstack.audio.dsp.ExpanderProcessor;
import com.aresstack.audio.dsp.FinalLoudnessNormalizer;
import com.aresstack.audio.dsp.FinalLoudnessNormalizerSettings;
import com.aresstack.audio.dsp.ExpanderSettings;
import com.aresstack.audio.dsp.ExpanderState;
import com.aresstack.audio.dsp.PlosiveReductionProcessor;
import com.aresstack.audio.dsp.PlosiveReductionSettings;
import com.aresstack.audio.dsp.RoomProfile;
import com.aresstack.audio.dsp.RoomReverbAnalyzer;
import com.aresstack.audio.dsp.SpectralBlockRunner;
import com.aresstack.audio.dsp.SpectralBreathReduction;
import com.aresstack.audio.dsp.SpectralDeEsser;
import com.aresstack.audio.dsp.SpectralHumRemoval;
import com.aresstack.audio.dsp.SpectralModifier;
import com.aresstack.audio.dsp.SpectralPlosiveReduction;
import com.aresstack.audio.dsp.SpeechGate;
import com.aresstack.audio.dsp.SilenceTrimNoSpeechBehavior;
import com.aresstack.audio.dsp.SilenceTrimmer;
import com.aresstack.audio.dsp.SilenceTrimmerSettings;
import com.aresstack.audio.dsp.SoftNoiseGateProcessor;
import com.aresstack.audio.dsp.SpeechActivityGate;
import com.aresstack.audio.dsp.SpeechActivityTrack;
import com.aresstack.audio.dsp.VoiceActivityDetector;
import com.aresstack.audio.dsp.VoiceActivityDetectorSettings;
import com.aresstack.audio.dsp.VoiceActivityDetectorState;
import com.aresstack.audio.profile.AudioBlockDefinition;

/**
 * Buffer-level {@link AudioBlockProcessor} factories for the built-in blocks. Each factory reproduces the
 * exact behavior the former {@code AudioProfileProcessor} switch had, so a profile's audio output is
 * unchanged: structural blocks (channel mixer, resampler) return a re-formatted buffer, and filter/dynamics
 * blocks wrap a freshly created {@link Pcm16Processor} and process in place.
 */
final class AudioBlockProcessors {

    /** Creates the {@link Pcm16Processor} for one block from its parameters. */
    private interface Pcm16Factory {
        Pcm16Processor create(AudioBlockDefinition block);
    }

    private AudioBlockProcessors() {
    }

    /** Neutral placeholder that leaves the audio unchanged until the user picks a real function. */
    static AudioBlockProcessor passthrough() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                return input;
            }
        };
    }

    static AudioBlockProcessor channelMixer() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                int target = block.getIntParameter("channels", 1);
                if (target != 1) {
                    throw new IllegalArgumentException("Only mono channel mixing is supported at present.");
                }
                if (input.getFormat().getChannels() == 1) {
                    return input;
                }
                short[] mono = PcmChannelConverter.downmixToMono(input.getSamples(),
                        input.getSamples().length, input.getFormat().getChannels());
                PcmAudioFormat format = new PcmAudioFormat(input.getFormat().getSampleRateHz(), 1,
                        input.getFormat().getBitsPerSample());
                return new AudioBuffer(mono, format);
            }
        };
    }

    static AudioBlockProcessor resampler() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                if (input.getFormat().getChannels() != 1) {
                    throw new IllegalArgumentException("Place a channel mixer before the resampler.");
                }
                int targetRate = block.getIntParameter("targetRateHz", 16000);
                ResamplingQuality quality = ResamplingQuality.parse(block.getParameter("quality", "BALANCED"));
                boolean hiddenAntiAliasing = block.getBooleanParameter("hiddenAntiAliasing", false);
                short[] samples = Pcm16Resampler.resample(input.getSamples(),
                        input.getFormat().getSampleRateHz(), targetRate, quality, hiddenAntiAliasing);
                PcmAudioFormat format = new PcmAudioFormat(targetRate, input.getFormat().getChannels(),
                        input.getFormat().getBitsPerSample());
                return new AudioBuffer(samples, format);
            }
        };
    }

    static AudioBlockProcessor lowPass() {
        return inPlace(new Pcm16Factory() {
            public Pcm16Processor create(AudioBlockDefinition block) {
                double cutoffHz = block.getDoubleParameter("cutoffHz", 7200.0d);
                if ("BUTTERWORTH".equalsIgnoreCase(block.getParameter("implementation", "FIR_65"))) {
                    return ButterworthFilterProcessor.lowPass(block.getIntParameter("order", 4), cutoffHz);
                }
                return new FirLowPassProcessor(cutoffHz);
            }
        });
    }

    static AudioBlockProcessor highPass() {
        return inPlace(new Pcm16Factory() {
            public Pcm16Processor create(AudioBlockDefinition block) {
                double cutoffHz = block.getDoubleParameter("cutoffHz", 80.0d);
                if ("BUTTERWORTH".equalsIgnoreCase(block.getParameter("implementation", "LEGACY_IIR"))) {
                    return ButterworthFilterProcessor.highPass(block.getIntParameter("order", 2), cutoffHz);
                }
                return new HighPassFilterProcessor(cutoffHz);
            }
        });
    }

    static AudioBlockProcessor bandPass() {
        return inPlace(new Pcm16Factory() {
            public Pcm16Processor create(AudioBlockDefinition block) {
                return ButterworthFilterProcessor.bandPass(block.getIntParameter("order", 2),
                        block.getDoubleParameter("centerHz", 1000.0d),
                        block.getDoubleParameter("widthHz", 500.0d));
            }
        });
    }

    static AudioBlockProcessor bandStop() {
        return inPlace(new Pcm16Factory() {
            public Pcm16Processor create(AudioBlockDefinition block) {
                return ButterworthFilterProcessor.bandStop(block.getIntParameter("order", 2),
                        block.getDoubleParameter("centerHz", 1000.0d),
                        block.getDoubleParameter("widthHz", 500.0d));
            }
        });
    }

    static AudioBlockProcessor dcOffsetRemoval() {
        return inPlace(new Pcm16Factory() {
            public Pcm16Processor create(AudioBlockDefinition block) {
                return new DcOffsetRemovalProcessor();
            }
        });
    }

    static AudioBlockProcessor noiseGate() {
        return inPlace(new Pcm16Factory() {
            public Pcm16Processor create(AudioBlockDefinition block) {
                return new SoftNoiseGateProcessor(
                        block.getDoubleParameter("threshold", 300.0d),
                        block.getDoubleParameter("closedGain", 0.3d),
                        block.getDoubleParameter("attackMillis", 5.0d),
                        block.getDoubleParameter("releaseMillis", 150.0d));
            }
        });
    }

    static AudioBlockProcessor compressor() {
        return inPlace(new Pcm16Factory() {
            public Pcm16Processor create(AudioBlockDefinition block) {
                return new CompressorProcessor(
                        block.getDoubleParameter("threshold", 12000.0d),
                        block.getDoubleParameter("ratio", 3.0d),
                        block.getDoubleParameter("attackMillis", 5.0d),
                        block.getDoubleParameter("releaseMillis", 100.0d));
            }
        });
    }

    static AudioBlockProcessor limiter() {
        return inPlace(new Pcm16Factory() {
            public Pcm16Processor create(AudioBlockDefinition block) {
                return new LimiterProcessor(block.getIntParameter("ceiling", 30000));
            }
        });
    }

    static AudioBlockProcessor gain() {
        return inPlace(new Pcm16Factory() {
            public Pcm16Processor create(AudioBlockDefinition block) {
                return new GainProcessor(block.getDoubleParameter("gainDb", 0.0d));
            }
        });
    }

    /** Channel-strip equalizer: 3-band tone shaping + master gain + optional loudness soft-saturation. */
    static AudioBlockProcessor equalizer() {
        return inPlace(new Pcm16Factory() {
            public Pcm16Processor create(AudioBlockDefinition block) {
                return new MasterEqualizerProcessor(
                        block.getDoubleParameter("lowShelfHz", 120.0d),
                        block.getDoubleParameter("lowShelfGainDb", 0.0d),
                        block.getDoubleParameter("midHz", 1000.0d),
                        block.getDoubleParameter("midGainDb", 0.0d),
                        block.getDoubleParameter("midQ", 1.0d),
                        block.getDoubleParameter("highShelfHz", 6000.0d),
                        block.getDoubleParameter("highShelfGainDb", 0.0d),
                        block.getDoubleParameter("gainDb", 0.0d),
                        block.getBooleanParameter("loudness", false),
                        block.getDoubleParameter("loudnessDriveDb", 0.0d),
                        block.getDoubleParameter("peakCeilingDb", -0.5d));
            }
        });
    }

    static AudioBlockProcessor parametricEqualizer() {
        return inPlace(new Pcm16Factory() {
            public Pcm16Processor create(AudioBlockDefinition block) {
                return new ParametricEqualizerProcessor(
                        block.getDoubleParameter("centerHz", 1000.0d),
                        block.getDoubleParameter("gainDb", 0.0d),
                        block.getDoubleParameter("q", 1.0d));
            }
        });
    }

    static AudioBlockProcessor lowShelfEqualizer() {
        return inPlace(new Pcm16Factory() {
            public Pcm16Processor create(AudioBlockDefinition block) {
                return new LowShelfEqualizerProcessor(
                        block.getDoubleParameter("cutoffHz", 200.0d),
                        block.getDoubleParameter("gainDb", 0.0d),
                        block.getDoubleParameter("slope", 1.0d));
            }
        });
    }

    static AudioBlockProcessor highShelfEqualizer() {
        return inPlace(new Pcm16Factory() {
            public Pcm16Processor create(AudioBlockDefinition block) {
                return new HighShelfEqualizerProcessor(
                        block.getDoubleParameter("cutoffHz", 6000.0d),
                        block.getDoubleParameter("gainDb", 0.0d),
                        block.getDoubleParameter("slope", 1.0d));
            }
        });
    }

    /**
     * Voice-activity detection: an analysis block that never changes the audio. It frames the buffer
     * (non-overlapping, frame size derived from the sample rate and configured duration, in order), runs the
     * adaptive detector with fresh state, publishes a {@link SpeechActivityTrack} into the context for later
     * blocks, and returns the input buffer unchanged (bit-identical PCM).
     */
    static AudioBlockProcessor voiceActivityDetection() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                PcmAudioFormat format = input.getFormat();
                int channels = Math.max(1, format.getChannels());
                VoiceActivityDetectorSettings settings = new VoiceActivityDetectorSettings(
                        block.getDoubleParameter("sensitivity", 0.5d),
                        block.getDoubleParameter("minSpeechProbability", 0.5d),
                        block.getIntParameter("frameDurationMs", 20),
                        block.getDoubleParameter("attackMs", 50.0d),
                        block.getDoubleParameter("releaseMs", 300.0d),
                        block.getDoubleParameter("hangoverMs", 200.0d),
                        block.getDoubleParameter("minSpeechMs", 80.0d),
                        block.getDoubleParameter("minSilenceMs", 150.0d),
                        block.getDoubleParameter("noiseAdaptationSpeed", 0.05d),
                        block.getBooleanParameter("adaptNoiseDuringSpeech", false));
                int framePerChannel = Math.max(1,
                        (int) Math.round(format.getSampleRateHz() * settings.getFrameDurationMs() / 1000.0d));
                int frameInterleaved = framePerChannel * channels;

                VoiceActivityDetector detector = new VoiceActivityDetector();
                VoiceActivityDetectorState state = new VoiceActivityDetectorState();
                SpeechActivityTrack track = new SpeechActivityTrack(
                        format.getSampleRateHz(), channels, framePerChannel);
                short[] samples = input.getSamples();
                for (int start = 0; start < samples.length; start += frameInterleaved) {
                    int count = Math.min(frameInterleaved, samples.length - start);
                    track.add(detector.analyzeFrame(samples, start, count, channels,
                            format.getSampleRateHz(), settings, state));
                }
                context.setSpeechActivity(track);
                return input; // analysis only — audio is passed through untouched
            }
        };
    }

    /**
     * Speech Gate: mute every non-speech region to exact digital silence using the upstream speech-activity
     * track only (no own detector). Format-preserving; without a usable track (or when the track's time base
     * no longer matches the audio) it passes the audio through unchanged — the validator flags that case.
     */
    static AudioBlockProcessor speechGate() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                SpeechActivityTrack track = context.getSpeechActivity();
                PcmAudioFormat format = input.getFormat();
                if (track == null || track.isEmpty()
                        || track.getSampleRateHz() != format.getSampleRateHz()
                        || track.getChannels() != Math.max(1, format.getChannels())
                        || track.getFrameSampleCountPerChannel() <= 0) {
                    return input; // no usable track: never fall back to a hidden energy detector
                }
                SpeechActivityGate.apply(input.getSamples(), input.getSamples().length, format, track,
                        block.getDoubleParameter("minSpeechProbability", 0.5d),
                        block.getDoubleParameter("preRollMs", 100.0d),
                        block.getDoubleParameter("postRollMs", 250.0d),
                        block.getDoubleParameter("attackMs", 5.0d),
                        block.getDoubleParameter("releaseMs", 50.0d));
                return input;
            }
        };
    }

    /**
     * Soft downward expander. Format-preserving; reads the optional upstream speech-activity track from the
     * context for speech protection (never duplicating a detector), and runs level-based when none is present.
     */
    static AudioBlockProcessor expander() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                ExpanderSettings settings = new ExpanderSettings(
                        block.getDoubleParameter("thresholdDb", -45.0d),
                        block.getDoubleParameter("ratio", 2.0d),
                        block.getDoubleParameter("kneeDb", 6.0d),
                        block.getDoubleParameter("attackMs", 10.0d),
                        block.getDoubleParameter("releaseMs", 200.0d),
                        block.getDoubleParameter("holdMs", 50.0d),
                        block.getDoubleParameter("maxAttenuationDb", 18.0d),
                        block.getDoubleParameter("detectorWindowMs", 20.0d),
                        block.getBooleanParameter("speechProtection", false),
                        block.getDoubleParameter("minSpeechProbability", 0.5d));
                SpeechActivityTrack track = settings.isSpeechProtection() ? context.getSpeechActivity() : null;
                new ExpanderProcessor(settings).process(input.getSamples(), input.getSamples().length,
                        input.getFormat(), new ExpanderState(), track);
                return input;
            }
        };
    }

    /**
     * Leading/trailing silence trimmer. Uses the upstream speech-activity track only; without one it passes
     * the audio through unchanged (the validator blocks that configuration). Trimming changes the sample
     * count, so a new buffer is returned and the now-stale speech track is invalidated for later blocks.
     */
    static AudioBlockProcessor silenceTrimmer() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                SpeechActivityTrack track = context.getSpeechActivity();
                if (track == null) {
                    return input; // no usable track: never fall back to a hidden energy detector
                }
                SilenceTrimmerSettings settings = new SilenceTrimmerSettings(
                        block.getBooleanParameter("trimLeading", true),
                        block.getBooleanParameter("trimTrailing", true),
                        block.getDoubleParameter("minSpeechProbability", 0.5d),
                        block.getDoubleParameter("preRollMs", 200.0d),
                        block.getDoubleParameter("postRollMs", 350.0d),
                        block.getDoubleParameter("minRetainedMs", 400.0d),
                        parseNoSpeech(block.getParameter("noSpeechBehavior", "KEEP_ORIGINAL")),
                        block.getBooleanParameter("zeroCrossingAlignment", true),
                        block.getDoubleParameter("zeroCrossingSearchMs", 5.0d));
                SilenceTrimmer.TrimBounds bounds = new SilenceTrimmer()
                        .computeBounds(input.getSamples(), input.getFormat(), track, settings);
                if (bounds.noSpeech) {
                    if (settings.getNoSpeechBehavior() == SilenceTrimNoSpeechBehavior.FAIL) {
                        throw new IllegalStateException(
                                "Silence Trimmer: no speech detected in the recording.");
                    }
                    return input; // KEEP_ORIGINAL
                }
                if (!bounds.trimmed) {
                    return input;
                }
                int length = bounds.endInterleaved - bounds.startInterleaved;
                short[] trimmed = new short[length];
                System.arraycopy(input.getSamples(), bounds.startInterleaved, trimmed, 0, length);
                context.setSpeechActivity(null); // the track no longer matches the trimmed time base
                return new AudioBuffer(trimmed, input.getFormat());
            }
        };
    }

    /** De-esser: dynamic reduction of an over-emphasized sibilance band. Format-preserving, no track needed. */
    static AudioBlockProcessor deEsser() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                DeEsserSettings settings = new DeEsserSettings(
                        block.getDoubleParameter("targetFrequencyHz", 6500.0d),
                        block.getDoubleParameter("bandwidthHz", 2500.0d),
                        block.getDoubleParameter("thresholdDb", -30.0d),
                        block.getDoubleParameter("reductionDb", 8.0d),
                        block.getDoubleParameter("attackMs", 2.0d),
                        block.getDoubleParameter("releaseMs", 60.0d));
                new DeEsserProcessor(settings).process(input.getSamples(), input.getSamples().length,
                        input.getFormat());
                return input;
            }
        };
    }

    /** Adaptive hum removal: tracks a drifting mains fundamental and notches it and its harmonics. */
    static AudioBlockProcessor adaptiveHumRemoval() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                AdaptiveHumRemovalSettings settings = new AdaptiveHumRemovalSettings(
                        block.getDoubleParameter("baseFrequencyHz", 50.0d),
                        block.getDoubleParameter("searchRangeHz", 3.0d),
                        block.getDoubleParameter("adaptationSpeed", 0.1d),
                        block.getIntParameter("harmonics", 3),
                        block.getDoubleParameter("maxAttenuationDb", 24.0d),
                        block.getBooleanParameter("speechProtection", false));
                SpeechActivityTrack track = settings.isSpeechProtection() ? context.getSpeechActivity() : null;
                new AdaptiveHumRemovalProcessor(settings).process(input.getSamples(),
                        input.getSamples().length, input.getFormat(), track);
                return input;
            }
        };
    }

    /** Plosive reduction: ducks low-frequency transient bursts (P/B pops). Format-preserving, no track needed. */
    static AudioBlockProcessor plosiveReduction() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                PlosiveReductionSettings settings = new PlosiveReductionSettings(
                        block.getDoubleParameter("strength", 0.6d),
                        block.getDoubleParameter("targetFrequencyHz", 120.0d),
                        block.getDoubleParameter("attackMs", 3.0d),
                        block.getDoubleParameter("releaseMs", 80.0d));
                new PlosiveReductionProcessor(settings).process(input.getSamples(),
                        input.getSamples().length, input.getFormat());
                return input;
            }
        };
    }

    /** Breath reduction: attenuates audible non-speech using the upstream track; passes through without one. */
    static AudioBlockProcessor breathReduction() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                BreathReductionSettings settings = new BreathReductionSettings(
                        block.getDoubleParameter("sensitivity", 0.5d),
                        block.getDoubleParameter("maxAttenuationDb", 12.0d),
                        block.getBooleanParameter("speechProtection", true),
                        block.getDoubleParameter("attackMs", 5.0d),
                        block.getDoubleParameter("releaseMs", 120.0d));
                new BreathReductionProcessor(settings).process(input.getSamples(),
                        input.getSamples().length, input.getFormat(), context.getSpeechActivity());
                return input;
            }
        };
    }

    private static final int FFT_SIZE = 1024;
    private static final int FFT_HOP = 512;

    /** De-Esser (FFT): attenuate the sibilance band in the STFT domain. */
    static AudioBlockProcessor deEsserFft() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                final DeEsserSettings settings = new DeEsserSettings(
                        block.getDoubleParameter("targetFrequencyHz", 6500.0d),
                        block.getDoubleParameter("bandwidthHz", 3000.0d),
                        block.getDoubleParameter("thresholdDb", -30.0d),
                        block.getDoubleParameter("reductionDb", 8.0d),
                        block.getDoubleParameter("attackMs", 5.0d),
                        block.getDoubleParameter("releaseMs", 60.0d));
                SpectralBlockRunner.apply(input.getSamples(), input.getSamples().length, input.getFormat(),
                        FFT_SIZE, FFT_HOP, new SpectralBlockRunner.ModifierFactory() {
                            public SpectralModifier create() {
                                return new SpectralDeEsser(settings, FFT_HOP);
                            }
                        });
                return input;
            }
        };
    }

    /** Adaptive Hum Removal (FFT): spectral peak tracking of the mains fundamental and harmonic bin notches. */
    static AudioBlockProcessor adaptiveHumRemovalFft() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                final AdaptiveHumRemovalSettings settings = new AdaptiveHumRemovalSettings(
                        block.getDoubleParameter("baseFrequencyHz", 50.0d),
                        block.getDoubleParameter("searchRangeHz", 3.0d),
                        block.getDoubleParameter("adaptationSpeed", 0.1d),
                        block.getIntParameter("harmonics", 3),
                        block.getDoubleParameter("maxAttenuationDb", 24.0d),
                        block.getBooleanParameter("speechProtection", false));
                final SpeechGate gate = speechGate(settings.isSpeechProtection() ? context.getSpeechActivity()
                        : null, input.getFormat().getChannels());
                SpectralBlockRunner.apply(input.getSamples(), input.getSamples().length, input.getFormat(),
                        FFT_SIZE, FFT_HOP, new SpectralBlockRunner.ModifierFactory() {
                            public SpectralModifier create() {
                                return new SpectralHumRemoval(settings, gate);
                            }
                        });
                return input;
            }
        };
    }

    /** Plosive Reduction (FFT): duck the low band on a spectral low-frequency transient. */
    static AudioBlockProcessor plosiveReductionFft() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                final PlosiveReductionSettings settings = new PlosiveReductionSettings(
                        block.getDoubleParameter("strength", 0.6d),
                        block.getDoubleParameter("targetFrequencyHz", 120.0d),
                        block.getDoubleParameter("attackMs", 5.0d),
                        block.getDoubleParameter("releaseMs", 80.0d));
                SpectralBlockRunner.apply(input.getSamples(), input.getSamples().length, input.getFormat(),
                        FFT_SIZE, FFT_HOP, new SpectralBlockRunner.ModifierFactory() {
                            public SpectralModifier create() {
                                return new SpectralPlosiveReduction(settings, FFT_HOP);
                            }
                        });
                return input;
            }
        };
    }

    /** Breath Reduction (FFT): attenuate noise-like non-speech frames judged by spectral flatness + VAD. */
    static AudioBlockProcessor breathReductionFft() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                final BreathReductionSettings settings = new BreathReductionSettings(
                        block.getDoubleParameter("sensitivity", 0.5d),
                        block.getDoubleParameter("maxAttenuationDb", 12.0d),
                        block.getBooleanParameter("speechProtection", true),
                        block.getDoubleParameter("attackMs", 5.0d),
                        block.getDoubleParameter("releaseMs", 120.0d));
                final SpeechGate gate = speechGate(context.getSpeechActivity(),
                        input.getFormat().getChannels());
                SpectralBlockRunner.apply(input.getSamples(), input.getSamples().length, input.getFormat(),
                        FFT_SIZE, FFT_HOP, new SpectralBlockRunner.ModifierFactory() {
                            public SpectralModifier create() {
                                return new SpectralBreathReduction(settings, FFT_HOP, gate);
                            }
                        });
                return input;
            }
        };
    }

    /**
     * Noise Profiler: an analysis block that never changes the audio. It estimates a background-noise
     * magnitude spectrum from the frames treated as noise (non-speech frames when an upstream speech track
     * is available, otherwise the whole signal), publishes the {@link NoiseProfile} into the context for a
     * later Adaptive Noise Suppression block, and returns the input buffer unchanged.
     */
    static AudioBlockProcessor noiseProfiler() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                String mode = block.getParameter("mode", "AUTOMATIC");
                if ("USE_EXISTING".equals(mode) && context.getNoiseProfile() != null) {
                    return input; // keep a previously supplied/learned profile
                }
                SpeechActivityTrack track = context.getSpeechActivity();
                SpeechGate gate = "LEARN_FROM_SILENCE".equals(mode)
                        ? null // treat the whole recording as noise
                        : (track == null ? null : speechGate(track, input.getFormat().getChannels()));
                int rate = input.getFormat().getSampleRateHz();
                int maxFrames = 0;
                double learnTimeMs = block.getDoubleParameter("learnTimeMs", 0.0d);
                if (learnTimeMs > 0.0d) {
                    maxFrames = (int) Math.max(1, learnTimeMs * rate / 1000.0d / FFT_HOP);
                }
                NoiseProfile profile = new NoiseProfileEstimator(FFT_SIZE, FFT_HOP)
                        .estimate(input.getSamples(), input.getSamples().length, input.getFormat(),
                                gate, maxFrames);
                if (profile != null) {
                    context.setNoiseProfile(profile);
                }
                return input; // analysis only — audio is passed through untouched
            }
        };
    }

    /**
     * Adaptive Noise Suppression: frequency-dependent reduction of stationary/slowly drifting background
     * noise in the STFT domain. In "use fixed profile" mode it uses the noise model published by an upstream
     * Noise Profiler; otherwise it tracks the noise floor per bin itself. Speech protection reads the
     * upstream speech-activity track. Format-preserving.
     */
    static AudioBlockProcessor adaptiveNoiseSuppression() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                final NoiseSuppressionSettings settings = new NoiseSuppressionSettings(
                        parseSuppressionMode(block.getParameter("mode", "AUTOMATIC")),
                        block.getDoubleParameter("maxAttenuationDb", 12.0d),
                        block.getDoubleParameter("adaptationSpeed", 0.1d),
                        block.getDoubleParameter("noiseFloorDb", -60.0d),
                        block.getBooleanParameter("speechProtection", true),
                        block.getDoubleParameter("minSpeechProbability", 0.5d),
                        block.getBooleanParameter("adaptDuringSpeech", false),
                        block.getBooleanParameter("freezeProfile", false),
                        block.getDoubleParameter("artifactProtection", 0.4d),
                        block.getDoubleParameter("attackMs", 15.0d),
                        block.getDoubleParameter("releaseMs", 120.0d));
                final NoiseProfile fixed = settings.getMode()
                        == NoiseSuppressionSettings.Mode.USE_FIXED_PROFILE ? context.getNoiseProfile() : null;
                final SpeechGate gate = speechGate(settings.isSpeechProtection() ? context.getSpeechActivity()
                        : null, input.getFormat().getChannels());
                SpectralBlockRunner.apply(input.getSamples(), input.getSamples().length, input.getFormat(),
                        FFT_SIZE, FFT_HOP, new SpectralBlockRunner.ModifierFactory() {
                            public SpectralModifier create() {
                                return new SpectralNoiseSuppressor(settings, fixed, gate);
                            }
                        });
                return input;
            }
        };
    }

    /**
     * Speech Leveler: speech-aware dynamic level control. Format-preserving; reads the upstream
     * speech-activity track for speech-driven gain and silence protection (level-based fallback without one).
     * A fresh {@link SpeechLevelerState} per run keeps results reproducible.
     */
    static AudioBlockProcessor speechLeveler() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                SpeechLevelerSettings settings = new SpeechLevelerSettings(
                        block.getDoubleParameter("targetSpeechLevelDb", -20.0d),
                        block.getDoubleParameter("maxGainDb", 18.0d),
                        block.getDoubleParameter("maxAttenuationDb", 12.0d),
                        block.getDoubleParameter("attackMs", 120.0d),
                        block.getDoubleParameter("releaseMs", 1000.0d),
                        block.getDoubleParameter("holdMs", 300.0d),
                        block.getDoubleParameter("maxGainChangePerSecond", 9.0d),
                        block.getDoubleParameter("minSpeechProbability", 0.5d),
                        block.getDoubleParameter("silenceGainLimitDb", 6.0d),
                        block.getBooleanParameter("noiseProtection", true),
                        block.getBooleanParameter("clippingProtection", true));
                new SpeechLevelerProcessor(settings).process(input.getSamples(), input.getSamples().length,
                        input.getFormat(), new SpeechLevelerState(), context.getSpeechActivity());
                return input;
            }
        };
    }

    /**
     * Final Loudness Normalizer: offline, whole-signal normalization to a target RMS or peak level with one
     * constant gain (no dynamic pumping), bounded by the maximum total boost/cut and the peak ceiling.
     */
    static AudioBlockProcessor finalLoudnessNormalizer() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                FinalLoudnessNormalizerSettings settings = new FinalLoudnessNormalizerSettings(
                        parseLoudnessMode(block.getParameter("mode", "TARGET_RMS")),
                        block.getDoubleParameter("targetLevelDb", -20.0d),
                        block.getDoubleParameter("maxTotalGainDb", 24.0d),
                        block.getDoubleParameter("maxTotalAttenuationDb", 24.0d),
                        block.getDoubleParameter("peakCeilingDb", -1.0d),
                        block.getBooleanParameter("clippingProtection", true),
                        block.getBooleanParameter("allowAmplification", true),
                        block.getBooleanParameter("allowAttenuation", true));
                new FinalLoudnessNormalizer(settings).process(input.getSamples(), input.getSamples().length);
                return input;
            }
        };
    }

    /**
     * Room/Reverb Analyzer: an analysis block that estimates reverberation time and strength from the decay
     * of the signal energy, publishes a {@link RoomProfile} into the context for a later Dereverberation
     * block, and returns the input buffer unchanged.
     */
    static AudioBlockProcessor roomReverbAnalyzer() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                RoomProfile profile = new RoomReverbAnalyzer(
                        block.getDoubleParameter("frameDurationMs", 20.0d),
                        block.getDoubleParameter("minDecayDb", 6.0d),
                        block.getDoubleParameter("maxReverbSeconds", 3.0d))
                        .analyze(input.getSamples(), input.getSamples().length, input.getFormat());
                context.setRoomProfile(profile);
                return input; // analysis only — audio is passed through untouched
            }
        };
    }

    /**
     * Speech Enhancer: a general block that runs a selectable speech-enhancement backend. The pure-Java
     * backend always works; optional native/model backends run only when available. When the selected
     * backend is missing (or its runtime/model is absent), the audio passes through unchanged so the profile
     * stays editable and never crashes — the validator surfaces the status.
     */
    static AudioBlockProcessor speechEnhancer() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                SpeechEnhancementBackend backend = SpeechEnhancementBackends.resolve(
                        block.getParameter("backend", "PURE_JAVA_DSP"));
                if (backend == null || !backend.availability(input.getFormat()).isRunnable()) {
                    return input; // missing/uninstalled backend: pass through, never fail
                }
                backend.enhance(input.getSamples(), input.getSamples().length, input.getFormat(),
                        block.getDoubleParameter("strength", 0.6d),
                        block.getBooleanParameter("speechProtection", true),
                        block.getDoubleParameter("artifactProtection", 0.4d));
                return input;
            }
        };
    }

    /**
     * Voice Isolation: keep the dominant voice and reduce other voices/music. Distinct from noise
     * suppression. The pure-Java center backend is a limited stereo approximation (emphasize centred speech,
     * reduce lateral content); a neural backend is optional and not bundled. When the backend is unavailable
     * or the source is unsuitable (mono for the center backend) the audio passes through unchanged.
     */
    static AudioBlockProcessor voiceIsolation() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                String backend = block.getParameter("backend", "PURE_JAVA_CENTER");
                if (!"PURE_JAVA_CENTER".equals(backend)) {
                    return input; // neural backend not installed: pass through, stays editable
                }
                if (input.getFormat().getChannels() != 2) {
                    return input; // the center approximation needs a stereo source
                }
                double strength = clamp01(block.getDoubleParameter("strength", 0.7d));
                StereoOps.applyMidSide(input.getSamples(), input.getSamples().length, 1.0d, 1.0d - strength, true);
                return input;
            }
        };
    }

    /** Channel Selector: output a mono buffer consisting of one chosen input channel (with a fallback). */
    static AudioBlockProcessor channelSelector() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                PcmAudioFormat format = input.getFormat();
                int channels = format.getChannels();
                if (channels == 1) {
                    return input;
                }
                int index = block.getIntParameter("channelIndex", 0);
                if (index < 0 || index >= channels) {
                    index = block.getIntParameter("fallbackChannel", 0);
                }
                short[] mono = MultichannelOps.selectChannel(input.getSamples(),
                        input.getSamples().length, channels, index);
                return new AudioBuffer(mono, new PcmAudioFormat(format.getSampleRateHz(), 1,
                        format.getBitsPerSample()));
            }
        };
    }

    /** Matrix Mixer / configurable downmix: a weighted combination of the input channels into mono. */
    static AudioBlockProcessor matrixMixer() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                PcmAudioFormat format = input.getFormat();
                int channels = format.getChannels();
                if (channels == 1) {
                    return input;
                }
                double[] weights = parseDoubles(block.getParameter("weights", ""));
                boolean normalize = block.getBooleanParameter("normalize", true);
                short[] mono = MultichannelOps.downmixToMono(input.getSamples(),
                        input.getSamples().length, channels, weights, normalize);
                return new AudioBuffer(mono, new PcmAudioFormat(format.getSampleRateHz(), 1,
                        format.getBitsPerSample()));
            }
        };
    }

    /** Channel Gain and Polarity: per-channel gain and optional polarity inversion (channel count kept). */
    static AudioBlockProcessor channelGainPolarity() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                int channels = input.getFormat().getChannels();
                double[] gains = parseDoubles(block.getParameter("gainsDb", ""));
                for (int i = 0; i < gains.length; i++) {
                    gains[i] = Math.pow(10.0d, gains[i] / 20.0d);
                }
                boolean[] invert = parseBooleans(block.getParameter("polarityInvert", ""), channels);
                MultichannelOps.applyGainPolarity(input.getSamples(), input.getSamples().length,
                        channels, gains, invert);
                return input;
            }
        };
    }

    /** Phase and Correlation Analyzer: measure channel 0/1 correlation and publish it. Audio unchanged. */
    static AudioBlockProcessor phaseCorrelationAnalyzer() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                if (input.getFormat().getChannels() >= 2) {
                    double corr = MultichannelOps.correlation(input.getSamples(),
                            input.getSamples().length, input.getFormat().getChannels(), 0, 1);
                    context.setChannelCorrelation(corr);
                }
                return input;
            }
        };
    }

    /**
     * Delay-and-Sum Beamformer: with a valid microphone-array geometry, aligns and sums the channels toward
     * a target direction, producing a mono output. Without a valid geometry (or on a channel-count mismatch)
     * it passes the input through unchanged — the validator surfaces the error — and never invents positions.
     */
    static AudioBlockProcessor delayAndSumBeamformer() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                PcmAudioFormat format = input.getFormat();
                int channels = format.getChannels();
                MicrophoneArrayProfile array = MicrophoneArrayProfile.parse(block.getId(), "inline",
                        block.getParameter("micPositionsMm", ""));
                if (array == null || channels < 2 || array.getMicrophoneCount() != channels) {
                    return input; // invalid/absent geometry: never fabricate positions
                }
                double[] weights = parseDoubles(block.getParameter("channelWeights", ""));
                double gain = Math.pow(10.0d, block.getDoubleParameter("outputGainDb", 0.0d) / 20.0d);
                double elevation = block.getDoubleParameter("targetElevationDeg", 0.0d);
                double c = block.getDoubleParameter("speedOfSoundMmPerS", 343000.0d);
                int rate = format.getSampleRateHz();
                int frames = input.getSamples().length / channels;
                if (block.getBooleanParameter("tracking", false)) {
                    short[] mono = beamformTracking(input.getSamples(), frames, channels, format, array,
                            block, weights, gain, elevation, c, rate);
                    return new AudioBuffer(mono, new PcmAudioFormat(rate, 1, format.getBitsPerSample()));
                }
                short[] mono = DelayAndSumBeamformer.beamform(input.getSamples(), input.getSamples().length,
                        format, array, block.getDoubleParameter("targetAzimuthDeg", 90.0d),
                        elevation, c, weights, gain);
                return new AudioBuffer(mono, new PcmAudioFormat(rate, 1, format.getBitsPerSample()));
            }
        };
    }

    /** Beamform per block toward a tracked direction so the beam follows a moving speaker. */
    private static short[] beamformTracking(short[] samples, int frames, int channels, PcmAudioFormat format,
                                            MicrophoneArrayProfile array, AudioBlockDefinition block,
                                            double[] weights, double gain, double elevation, double c, int rate) {
        int blockFrames = Math.max(64, block.getIntParameter("trackingBlockFrames", 512));
        int maxLag = Math.max(1, block.getIntParameter("maxLagSamples", 32));
        DirectionTracker tracker = new DirectionTracker(
                block.getDoubleParameter("directionSmoothing", 0.7d),
                block.getDoubleParameter("maxAngularSpeedDegPerBlock", 15.0d),
                block.getDoubleParameter("minConfidence", 0.2d),
                block.getIntParameter("holdBlocks", 3),
                block.getDoubleParameter("fallbackAzimuthDeg", 90.0d),
                block.getBooleanParameter("updateDuringSilence", false));
        short[] mono = new short[frames];
        for (int start = 0; start < frames; start += blockFrames) {
            int end = Math.min(frames, start + blockFrames);
            DirectionEstimate est = DirectionOfArrivalEstimator.estimate(samples, start, end, channels,
                    array, rate, c, maxLag);
            double azimuth = tracker.update(est);
            int blkLen = (end - start) * channels;
            short[] blk = new short[blkLen];
            System.arraycopy(samples, start * channels, blk, 0, blkLen);
            short[] monoBlk = DelayAndSumBeamformer.beamform(blk, blkLen, format, array, azimuth, elevation,
                    c, weights, gain);
            System.arraycopy(monoBlk, 0, mono, start, end - start);
        }
        return mono;
    }

    /**
     * Direction of Arrival Analyzer: estimate the source azimuth from the array geometry and publish a
     * {@link DirectionEstimate} into the context. Analysis only — the audio is unchanged.
     */
    static AudioBlockProcessor directionOfArrivalAnalyzer() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                int channels = input.getFormat().getChannels();
                MicrophoneArrayProfile array = MicrophoneArrayProfile.parse(block.getId(), "inline",
                        block.getParameter("micPositionsMm", ""));
                if (array == null || channels < 2 || array.getMicrophoneCount() != channels) {
                    return input;
                }
                int frames = input.getSamples().length / channels;
                DirectionEstimate estimate = DirectionOfArrivalEstimator.estimate(input.getSamples(), 0, frames,
                        channels, array, input.getFormat().getSampleRateHz(),
                        block.getDoubleParameter("speedOfSoundMmPerS", 343000.0d),
                        Math.max(1, block.getIntParameter("maxLagSamples", 32)));
                context.setDirectionEstimate(estimate);
                return input;
            }
        };
    }

    /** Mid/Side Processor: independent mid/side gains with mono-compatibility protection. Stereo only. */
    static AudioBlockProcessor midSideProcessor() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                if (input.getFormat().getChannels() != 2) {
                    return input;
                }
                double midGain = Math.pow(10.0d, block.getDoubleParameter("midGainDb", 0.0d) / 20.0d);
                double sideGain = Math.pow(10.0d, block.getDoubleParameter("sideGainDb", 0.0d) / 20.0d);
                double sideReduction = block.getDoubleParameter("sideReduction", 0.0d);
                sideGain *= Math.max(0.0d, 1.0d - sideReduction);
                StereoOps.applyMidSide(input.getSamples(), input.getSamples().length, midGain, sideGain,
                        block.getBooleanParameter("monoCompatibilityProtection", true));
                return input;
            }
        };
    }

    /** Center Speech Extractor: emphasize centred (mid) speech and reduce lateral (side) content. Stereo only. */
    static AudioBlockProcessor centerSpeechExtractor() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                if (input.getFormat().getChannels() != 2) {
                    return input;
                }
                double centerAmount = clamp01(block.getDoubleParameter("centerAmount", 0.6d));
                double midGain = Math.pow(10.0d, block.getDoubleParameter("centerBoostDb", 0.0d) / 20.0d);
                StereoOps.applyMidSide(input.getSamples(), input.getSamples().length, midGain,
                        1.0d - centerAmount, true);
                return input;
            }
        };
    }

    /** Stereo Width Control: scale the side signal to narrow/widen, keeping the centre. Stereo only. */
    static AudioBlockProcessor stereoWidthControl() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                if (input.getFormat().getChannels() != 2) {
                    return input;
                }
                double width = block.getDoubleParameter("width", 1.0d);
                StereoOps.applyMidSide(input.getSamples(), input.getSamples().length, 1.0d, width,
                        block.getBooleanParameter("monoCompatibilityProtection", true));
                return input;
            }
        };
    }

    private static double clamp01(double value) {
        if (Double.isNaN(value) || value < 0.0d) {
            return 0.0d;
        }
        return value > 1.0d ? 1.0d : value;
    }

    /** Channel Delay Alignment: time-align channels to a reference (auto cross-correlation or manual). */
    static AudioBlockProcessor channelDelayAlignment() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                int channels = input.getFormat().getChannels();
                if (channels < 2) {
                    return input;
                }
                int reference = clampIndex(block.getIntParameter("referenceChannel", 0), channels);
                int maxCorrection = Math.max(1, block.getIntParameter("maxCorrectionSamples", 64));
                boolean fractional = block.getBooleanParameter("fractionalDelay", true);
                double[] delays = new double[channels];
                if ("MANUAL".equals(block.getParameter("mode", "AUTO"))) {
                    double[] manual = parseDoubles(block.getParameter("delaysSamples", ""));
                    for (int c = 0; c < channels && c < manual.length; c++) {
                        delays[c] = clampDelay(manual[c], maxCorrection);
                    }
                } else {
                    for (int c = 0; c < channels; c++) {
                        if (c == reference) {
                            continue;
                        }
                        double lag = ChannelAligner.estimateDelay(input.getSamples(),
                                input.getSamples().length, channels, reference, c, maxCorrection);
                        if (!fractional) {
                            lag = Math.rint(lag);
                        }
                        delays[c] = clampDelay(-lag, maxCorrection); // shift the channel to line up with the reference
                    }
                }
                ChannelAligner.applyDelays(input.getSamples(), input.getSamples().length, channels, delays);
                return input;
            }
        };
    }

    /** Best Channel Selector: choose the highest-quality channel and output it as mono. */
    static AudioBlockProcessor bestChannelSelector() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                PcmAudioFormat format = input.getFormat();
                int channels = format.getChannels();
                if (channels == 1) {
                    return input;
                }
                int preferred = block.getIntParameter("preferredChannel", -1);
                int best = ChannelDiagnostics.bestChannelIndex(input.getSamples(),
                        input.getSamples().length, channels, preferred);
                short[] mono = MultichannelOps.selectChannel(input.getSamples(),
                        input.getSamples().length, channels, best);
                return new AudioBuffer(mono, new PcmAudioFormat(format.getSampleRateHz(), 1,
                        format.getBitsPerSample()));
            }
        };
    }

    /** Channel Health Analyzer: report silent/clipping/DC/quiet channels. Audio unchanged. */
    static AudioBlockProcessor channelHealthAnalyzer() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                context.setChannelHealthSummary(ChannelDiagnostics.describeHealth(input.getSamples(),
                        input.getSamples().length, input.getFormat().getChannels()));
                return input;
            }
        };
    }

    private static int clampIndex(int index, int channels) {
        return index < 0 ? 0 : (index >= channels ? channels - 1 : index);
    }

    private static double clampDelay(double delay, int max) {
        if (Double.isNaN(delay)) {
            return 0.0d;
        }
        return delay < -max ? -max : (delay > max ? max : delay);
    }

    private static double[] parseDoubles(String csv) {
        if (csv == null || csv.trim().length() == 0) {
            return new double[0];
        }
        String[] parts = csv.split(",");
        double[] out = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                out[i] = Double.parseDouble(parts[i].trim());
            } catch (NumberFormatException ex) {
                out[i] = 0.0d;
            }
        }
        return out;
    }

    private static boolean[] parseBooleans(String csv, int channels) {
        boolean[] out = new boolean[Math.max(0, channels)];
        if (csv == null || csv.trim().length() == 0) {
            return out;
        }
        String[] parts = csv.split(",");
        for (int i = 0; i < parts.length && i < out.length; i++) {
            String v = parts[i].trim();
            out[i] = "1".equals(v) || "true".equalsIgnoreCase(v) || "-".equals(v);
        }
        return out;
    }

    /**
     * Dereverberation: pure-Java single-channel WPE. Format-preserving; reads the upstream speech-activity
     * track for speech protection. A fresh processor per run keeps adaptive estimates from leaking.
     */
    static AudioBlockProcessor dereverberation() {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                WpeDereverberationSettings settings = new WpeDereverberationSettings(
                        parseWpeMode(block.getParameter("mode", "OFFLINE")),
                        block.getDoubleParameter("strength", 0.7d),
                        block.getIntParameter("predictionDelay", 2),
                        block.getIntParameter("filterLength", 8),
                        block.getIntParameter("iterations", 3),
                        block.getDoubleParameter("earlyReflectionPreservation", 0.5d),
                        block.getBooleanParameter("speechProtection", false),
                        block.getDoubleParameter("artifactProtection", 0.2d),
                        block.getDoubleParameter("adaptationSpeed", 0.3d),
                        block.getIntParameter("blockSizeFrames", 64));
                SpeechGate gate = speechGate(settings.isSpeechProtection() ? context.getSpeechActivity() : null,
                        input.getFormat().getChannels());
                new WpeDereverberation(settings).process(input.getSamples(), input.getSamples().length,
                        input.getFormat(), gate);
                return input;
            }
        };
    }

    private static WpeDereverberationSettings.Mode parseWpeMode(String value) {
        try {
            return WpeDereverberationSettings.Mode.valueOf(value);
        } catch (RuntimeException ex) {
            return WpeDereverberationSettings.Mode.OFFLINE;
        }
    }

    private static FinalLoudnessNormalizerSettings.Mode parseLoudnessMode(String value) {
        try {
            return FinalLoudnessNormalizerSettings.Mode.valueOf(value);
        } catch (RuntimeException ex) {
            return FinalLoudnessNormalizerSettings.Mode.TARGET_RMS;
        }
    }

    private static NoiseSuppressionSettings.Mode parseSuppressionMode(String value) {
        try {
            return NoiseSuppressionSettings.Mode.valueOf(value);
        } catch (RuntimeException ex) {
            return NoiseSuppressionSettings.Mode.AUTOMATIC;
        }
    }

    /** Bridge a mono STFT sample index to the interleaved speech-activity track, or NEVER without a track. */
    private static SpeechGate speechGate(final SpeechActivityTrack track, final int channels) {
        if (track == null) {
            return SpeechGate.NEVER;
        }
        final int ch = Math.max(1, channels);
        return new SpeechGate() {
            public boolean isSpeech(int monoSampleIndex) {
                if (monoSampleIndex < 0) {
                    return false;
                }
                com.aresstack.audio.dsp.SpeechActivityMetadata frame =
                        track.frameForInterleavedIndex(monoSampleIndex * ch);
                return frame != null && frame.isSpeechActive();
            }
        };
    }

    private static SilenceTrimNoSpeechBehavior parseNoSpeech(String value) {
        try {
            return SilenceTrimNoSpeechBehavior.valueOf(value);
        } catch (RuntimeException ex) {
            return SilenceTrimNoSpeechBehavior.KEEP_ORIGINAL;
        }
    }

    /** Wrap a per-block {@link Pcm16Processor} as an in-place, format-preserving buffer processor. */
    private static AudioBlockProcessor inPlace(final Pcm16Factory factory) {
        return new AudioBlockProcessor() {
            public AudioBuffer process(AudioBuffer input, AudioBlockDefinition block, AudioProcessingContext context) {
                Pcm16Processor processor = factory.create(block);
                processor.process(input.getSamples(), input.getSamples().length, input.getFormat());
                return input;
            }
        };
    }
}

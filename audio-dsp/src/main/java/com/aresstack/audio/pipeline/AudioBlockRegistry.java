package com.aresstack.audio.pipeline;

import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The central registry of {@link AudioBlockDescriptor}s. It is the single place that knows every block
 * type: its parameters/defaults, capabilities, category, processor factory and canvas summary. The former
 * per-type switches in the profile processor, the inspector and the canvas all resolve through here, and a
 * new block is added by registering one descriptor.
 */
public final class AudioBlockRegistry {

    private static final AudioBlockRegistry INSTANCE = new AudioBlockRegistry();

    private final Map<AudioBlockType, AudioBlockDescriptor> descriptors =
            new LinkedHashMap<AudioBlockType, AudioBlockDescriptor>();

    private AudioBlockRegistry() {
        register(passthrough());
        register(channelMixer());
        register(lowPass());
        register(highPass());
        register(bandPass());
        register(bandStop());
        register(resampler());
        register(dcOffsetRemoval());
        register(noiseGate());
        register(compressor());
        register(limiter());
        register(gain());
        register(parametricEqualizer());
        register(lowShelfEqualizer());
        register(highShelfEqualizer());
        register(equalizer());
        register(voiceActivityDetection());
        register(speechGate());
        register(expander());
        register(silenceTrimmer());
        register(deEsser());
        register(adaptiveHumRemoval());
        register(plosiveReduction());
        register(breathReduction());
        register(deEsserFft());
        register(adaptiveHumRemovalFft());
        register(plosiveReductionFft());
        register(breathReductionFft());
        register(noiseProfiler());
        register(adaptiveNoiseSuppression());
        register(speechLeveler());
        register(finalLoudnessNormalizer());
        register(roomReverbAnalyzer());
        register(dereverberation());
        register(channelSelector());
        register(matrixMixer());
        register(channelGainPolarity());
        register(phaseCorrelationAnalyzer());
        register(channelDelayAlignment());
        register(bestChannelSelector());
        register(channelHealthAnalyzer());
        register(midSideProcessor());
        register(centerSpeechExtractor());
        register(stereoWidthControl());
        register(delayAndSumBeamformer());
        register(directionOfArrivalAnalyzer());
        register(speechEnhancer());
        register(voiceIsolation());
    }

    public static AudioBlockRegistry getInstance() {
        return INSTANCE;
    }

    private void register(AudioBlockDescriptor descriptor) {
        descriptors.put(descriptor.getType(), descriptor);
    }

    /** @return the descriptor for a type, never null (every {@link AudioBlockType} is registered). */
    public AudioBlockDescriptor descriptor(AudioBlockType type) {
        AudioBlockDescriptor descriptor = descriptors.get(type);
        if (descriptor == null) {
            throw new IllegalArgumentException("No descriptor registered for block type: " + type);
        }
        return descriptor;
    }

    /** @return the descriptor for a persisted type id, or null when the id is unknown (diagnosable). */
    public AudioBlockDescriptor descriptorForId(String typeId) {
        if (typeId == null) {
            return null;
        }
        for (AudioBlockDescriptor descriptor : descriptors.values()) {
            if (descriptor.getTypeId().equals(typeId.trim())) {
                return descriptor;
            }
        }
        return null;
    }

    public List<AudioBlockDescriptor> all() {
        return Collections.unmodifiableList(new ArrayList<AudioBlockDescriptor>(descriptors.values()));
    }

    public AudioBlockProcessor createProcessor(AudioBlockType type) {
        return descriptor(type).createProcessor();
    }

    public Map<String, String> defaultParameters(AudioBlockType type) {
        return descriptor(type).defaultParameters();
    }

    public AudioBlockDefinition defaultDefinition(AudioBlockType type, String id) {
        return descriptor(type).createDefaultDefinition(id);
    }

    /** @return a default definition of the type with a freshly generated block id (for "add block"). */
    public AudioBlockDefinition createDefaultDefinition(AudioBlockType type) {
        return descriptor(type).createDefaultDefinition("block-" + java.util.UUID.randomUUID().toString());
    }

    // ------------------------------------------------------------------ descriptor declarations

    private static AudioBlockDescriptor passthrough() {
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .modifiesAudio(false)
                .build();
        return descriptor(AudioBlockType.PASSTHROUGH, AudioBlockCategory.GENERAL,
                Collections.<AudioParameterDescriptor>emptyList(), capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.passthrough();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "No effect — choose a function";
                    }
                });
    }

    private static AudioBlockDescriptor channelMixer() {
        return descriptor(AudioBlockType.CHANNEL_MIXER, AudioBlockCategory.INPUT_CHANNEL,
                Collections.singletonList(AudioParameterDescriptor.integer("channels", "Output channels", 1, 1, 1)),
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.channelMixer();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Output: mono";
                    }
                });
    }

    private static AudioBlockDescriptor lowPass() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.choice("implementation", "Filter design", "FIR_65", Arrays.asList(
                new AudioParameterChoice("FIR_65", "65-tap FIR (existing)"),
                new AudioParameterChoice("BUTTERWORTH", "Butterworth (iirj)"))));
        params.add(AudioParameterDescriptor.decimal("cutoffHz", "Cutoff (Hz)", 7200.0d, 1.0d, 96000.0d, 10.0d));
        params.add(AudioParameterDescriptor.integer("order", "Butterworth order", 4, 1, 12));
        return descriptor(AudioBlockType.LOW_PASS, AudioBlockCategory.FILTERS_EQ, params,
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.lowPass();
                    }
                },
                filterSummary());
    }

    private static AudioBlockDescriptor highPass() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.choice("implementation", "Filter design", "LEGACY_IIR", Arrays.asList(
                new AudioParameterChoice("LEGACY_IIR", "First-order IIR (existing)"),
                new AudioParameterChoice("BUTTERWORTH", "Butterworth (iirj)"))));
        params.add(AudioParameterDescriptor.decimal("cutoffHz", "Cutoff (Hz)", 80.0d, 1.0d, 96000.0d, 10.0d));
        params.add(AudioParameterDescriptor.integer("order", "Butterworth order", 2, 1, 12));
        return descriptor(AudioBlockType.HIGH_PASS, AudioBlockCategory.FILTERS_EQ, params,
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.highPass();
                    }
                },
                filterSummary());
    }

    private static AudioBlockDescriptor bandPass() {
        return bandDescriptor(AudioBlockType.BAND_PASS, new SimpleAudioBlockDescriptor.ProcessorFactory() {
            public AudioBlockProcessor create() {
                return AudioBlockProcessors.bandPass();
            }
        });
    }

    private static AudioBlockDescriptor bandStop() {
        return bandDescriptor(AudioBlockType.BAND_STOP, new SimpleAudioBlockDescriptor.ProcessorFactory() {
            public AudioBlockProcessor create() {
                return AudioBlockProcessors.bandStop();
            }
        });
    }

    private static AudioBlockDescriptor bandDescriptor(AudioBlockType type,
                                                       SimpleAudioBlockDescriptor.ProcessorFactory factory) {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("centerHz", "Center (Hz)", 1000.0d, 1.0d, 96000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("widthHz", "Width (Hz)", 500.0d, 1.0d, 96000.0d, 10.0d));
        params.add(AudioParameterDescriptor.integer("order", "Filter order", 2, 1, 12));
        return descriptor(type, AudioBlockCategory.FILTERS_EQ, params, StaticBlockCapabilities.audioEffect(),
                factory,
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return block.getParameter("centerHz", "") + " Hz · width "
                                + block.getParameter("widthHz", "") + " Hz";
                    }
                });
    }

    private static AudioBlockDescriptor resampler() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.integer("targetRateHz", "Target rate (Hz)", 16000, 4000, 192000));
        params.add(AudioParameterDescriptor.choice("quality", "Quality", "BALANCED", Arrays.asList(
                new AudioParameterChoice("FAST", "FAST"),
                new AudioParameterChoice("BALANCED", "BALANCED"),
                new AudioParameterChoice("HIGH", "HIGH"))));
        params.add(AudioParameterDescriptor.bool("hiddenAntiAliasing", "Hidden anti-alias filter", false));
        return descriptor(AudioBlockType.RESAMPLER, AudioBlockCategory.OUTPUT, params,
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.resampler();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return block.getParameter("targetRateHz", "") + " Hz · "
                                + block.getParameter("quality", "BALANCED");
                    }
                });
    }

    private static AudioBlockDescriptor dcOffsetRemoval() {
        return descriptor(AudioBlockType.DC_OFFSET_REMOVAL, AudioBlockCategory.FILTERS_EQ,
                Collections.<AudioParameterDescriptor>emptyList(), StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.dcOffsetRemoval();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Adaptive offset estimate";
                    }
                });
    }

    private static AudioBlockDescriptor noiseGate() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("threshold", "Threshold", 300.0d, 0.0d, 32767.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("closedGain", "Closed gain", 0.3d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.decimal("attackMillis", "Attack (ms)", 5.0d, 0.0d, 5000.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("releaseMillis", "Release (ms)", 150.0d, 0.0d, 10000.0d, 5.0d));
        return descriptor(AudioBlockType.NOISE_GATE, AudioBlockCategory.DYNAMICS, params,
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.noiseGate();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Threshold " + block.getParameter("threshold", "");
                    }
                });
    }

    private static AudioBlockDescriptor compressor() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("threshold", "Threshold", 12000.0d, 0.0d, 32767.0d, 100.0d));
        params.add(AudioParameterDescriptor.decimal("ratio", "Ratio", 3.0d, 1.0d, 30.0d, 0.5d));
        params.add(AudioParameterDescriptor.decimal("attackMillis", "Attack (ms)", 5.0d, 0.0d, 5000.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("releaseMillis", "Release (ms)", 100.0d, 0.0d, 10000.0d, 5.0d));
        return descriptor(AudioBlockType.COMPRESSOR, AudioBlockCategory.DYNAMICS, params,
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.compressor();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return block.getParameter("ratio", "") + ":1 above " + block.getParameter("threshold", "");
                    }
                });
    }

    private static AudioBlockDescriptor limiter() {
        return descriptor(AudioBlockType.LIMITER, AudioBlockCategory.OUTPUT,
                Collections.singletonList(AudioParameterDescriptor.integer("ceiling", "Ceiling", 30000, 1, 32767)),
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.limiter();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Ceiling " + block.getParameter("ceiling", "");
                    }
                });
    }

    private static AudioBlockDescriptor gain() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("gainDb", "Gain (dB)", 0.0d, -60.0d, 24.0d, 0.5d));
        return descriptor(AudioBlockType.GAIN, AudioBlockCategory.DYNAMICS, params,
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.gain();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return formatDb(block.getDoubleParameter("gainDb", 0.0d));
                    }
                });
    }

    private static AudioBlockDescriptor parametricEqualizer() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("centerHz", "Center (Hz)", 1000.0d, 1.0d, 96000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("gainDb", "Gain (dB)", 0.0d, -24.0d, 24.0d, 0.5d));
        params.add(AudioParameterDescriptor.decimal("q", "Q factor", 1.0d, 0.1d, 24.0d, 0.1d));
        return descriptor(AudioBlockType.PARAMETRIC_EQ, AudioBlockCategory.FILTERS_EQ, params,
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.parametricEqualizer();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return formatHz(block.getDoubleParameter("centerHz", 0.0d)) + " · "
                                + formatDb(block.getDoubleParameter("gainDb", 0.0d)) + " · Q "
                                + formatQ(block.getDoubleParameter("q", 1.0d));
                    }
                });
    }

    private static AudioBlockDescriptor lowShelfEqualizer() {
        return shelfDescriptor(AudioBlockType.LOW_SHELF, 200.0d,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.lowShelfEqualizer();
                    }
                });
    }

    private static AudioBlockDescriptor highShelfEqualizer() {
        return shelfDescriptor(AudioBlockType.HIGH_SHELF, 6000.0d,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.highShelfEqualizer();
                    }
                });
    }

    private static AudioBlockDescriptor equalizer() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("lowShelfHz", "Low shelf (Hz)", 120.0d, 20.0d, 2000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("lowShelfGainDb", "Low shelf gain (dB)",
                0.0d, -24.0d, 24.0d, 0.5d));
        params.add(AudioParameterDescriptor.decimal("midHz", "Mid frequency (Hz)", 1000.0d, 100.0d, 12000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("midGainDb", "Mid gain (dB)", 0.0d, -24.0d, 24.0d, 0.5d));
        params.add(AudioParameterDescriptor.decimal("midQ", "Mid Q", 1.0d, 0.1d, 24.0d, 0.1d));
        params.add(AudioParameterDescriptor.decimal("highShelfHz", "High shelf (Hz)",
                6000.0d, 1000.0d, 20000.0d, 100.0d));
        params.add(AudioParameterDescriptor.decimal("highShelfGainDb", "High shelf gain (dB)",
                0.0d, -24.0d, 24.0d, 0.5d));
        params.add(AudioParameterDescriptor.decimal("gainDb", "Master gain (dB)", 0.0d, -24.0d, 24.0d, 0.5d));
        params.add(AudioParameterDescriptor.bool("loudness", "Loudness (soft saturation)", false));
        params.add(AudioParameterDescriptor.decimal("loudnessDriveDb", "Loudness drive (dB)",
                0.0d, 0.0d, 24.0d, 0.5d));
        params.add(AudioParameterDescriptor.decimal("peakCeilingDb", "Peak ceiling (dBFS)",
                -0.5d, -12.0d, 0.0d, 0.5d));
        return descriptor(AudioBlockType.EQUALIZER, AudioBlockCategory.FILTERS_EQ, params,
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.equalizer();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        String loud = block.getBooleanParameter("loudness", false)
                                ? " · loudness +" + formatQ(block.getDoubleParameter("loudnessDriveDb", 0.0d)) : "";
                        return "3-band · master " + formatDb(block.getDoubleParameter("gainDb", 0.0d)) + loud;
                    }
                });
    }

    private static AudioBlockDescriptor shelfDescriptor(AudioBlockType type, double defaultCutoffHz,
                                                        SimpleAudioBlockDescriptor.ProcessorFactory factory) {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("cutoffHz", "Cutoff (Hz)", defaultCutoffHz, 1.0d, 96000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("gainDb", "Gain (dB)", 0.0d, -24.0d, 24.0d, 0.5d));
        params.add(AudioParameterDescriptor.decimal("slope", "Shelf slope", 1.0d, 0.1d, 2.0d, 0.1d));
        return descriptor(type, AudioBlockCategory.FILTERS_EQ, params, StaticBlockCapabilities.audioEffect(),
                factory,
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return formatHz(block.getDoubleParameter("cutoffHz", 0.0d)) + " · "
                                + formatDb(block.getDoubleParameter("gainDb", 0.0d));
                    }
                });
    }

    private static AudioBlockDescriptor voiceActivityDetection() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("sensitivity", "Sensitivity", 0.5d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.decimal("minSpeechProbability", "Min speech probability",
                0.5d, 0.3d, 0.9d, 0.05d));
        params.add(AudioParameterDescriptor.integer("frameDurationMs", "Frame duration (ms)", 20, 10, 30));
        params.add(AudioParameterDescriptor.decimal("attackMs", "Attack (ms)", 50.0d, 20.0d, 300.0d, 5.0d));
        params.add(AudioParameterDescriptor.decimal("releaseMs", "Release (ms)", 300.0d, 100.0d, 1500.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("hangoverMs", "Hangover (ms)", 200.0d, 0.0d, 1000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("minSpeechMs", "Min speech duration (ms)",
                80.0d, 0.0d, 500.0d, 5.0d));
        params.add(AudioParameterDescriptor.decimal("minSilenceMs", "Min silence duration (ms)",
                150.0d, 0.0d, 1000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("noiseAdaptationSpeed", "Noise adaptation speed",
                0.05d, 0.001d, 0.2d, 0.001d));
        params.add(AudioParameterDescriptor.bool("adaptNoiseDuringSpeech", "Adapt noise during speech", false));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .modifiesAudio(false)
                .producesMetadata(true)
                .framing(320, 0, 0) // requires framing; frame size is derived per rate at run time
                .build();
        return descriptor(AudioBlockType.VOICE_ACTIVITY_DETECTION, AudioBlockCategory.ANALYSIS, params,
                capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.voiceActivityDetection();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Sensitivity "
                                + String.format(java.util.Locale.ROOT, "%.2f",
                                        block.getDoubleParameter("sensitivity", 0.5d))
                                + " · threshold "
                                + String.format(java.util.Locale.ROOT, "%.2f",
                                        block.getDoubleParameter("minSpeechProbability", 0.5d));
                    }
                });
    }

    private static AudioBlockDescriptor speechGate() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("minSpeechProbability", "Min speech probability",
                0.5d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.decimal("preRollMs", "Pre-roll (ms)", 100.0d, 0.0d, 2000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("postRollMs", "Post-roll (ms)", 250.0d, 0.0d, 5000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("attackMs", "Open fade (ms)", 5.0d, 0.0d, 500.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("releaseMs", "Close fade (ms)", 50.0d, 0.0d, 2000.0d, 5.0d));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .framing(320, 0, 0)
                .consumesSpeechMetadata(true)
                .requiresSpeechActivityTrack(true)
                .build();
        return descriptor(AudioBlockType.SPEECH_GATE, AudioBlockCategory.DYNAMICS, params, capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.speechGate();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Mute pauses · pre " + formatQ(block.getDoubleParameter("preRollMs", 100.0d))
                                + " / post " + formatQ(block.getDoubleParameter("postRollMs", 250.0d)) + " ms";
                    }
                });
    }

    private static AudioBlockDescriptor expander() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("thresholdDb", "Threshold (dBFS)", -45.0d, -80.0d, 0.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("ratio", "Ratio", 2.0d, 1.0d, 20.0d, 0.5d));
        params.add(AudioParameterDescriptor.decimal("kneeDb", "Knee (dB)", 6.0d, 0.0d, 24.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("attackMs", "Attack (ms)", 10.0d, 0.0d, 500.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("releaseMs", "Release (ms)", 200.0d, 10.0d, 5000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("holdMs", "Hold (ms)", 50.0d, 0.0d, 2000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("maxAttenuationDb", "Maximum attenuation (dB)",
                18.0d, 0.0d, 80.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("detectorWindowMs", "Detector window (ms)",
                20.0d, 5.0d, 100.0d, 1.0d));
        params.add(AudioParameterDescriptor.bool("speechProtection", "Speech protection", false));
        params.add(AudioParameterDescriptor.decimal("minSpeechProbability", "Min speech probability",
                0.5d, 0.0d, 1.0d, 0.05d));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .framing(320, 0, 0)
                .consumesSpeechMetadata(true)
                .build();
        return descriptor(AudioBlockType.EXPANDER, AudioBlockCategory.DYNAMICS, params, capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.expander();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return formatHz0(block.getDoubleParameter("thresholdDb", 0.0d)) + " dB · "
                                + formatQ(block.getDoubleParameter("ratio", 1.0d)) + ":1 · max -"
                                + formatQ(block.getDoubleParameter("maxAttenuationDb", 0.0d)) + " dB";
                    }
                });
    }

    private static AudioBlockDescriptor silenceTrimmer() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.bool("trimLeading", "Trim leading silence", true));
        params.add(AudioParameterDescriptor.bool("trimTrailing", "Trim trailing silence", true));
        params.add(AudioParameterDescriptor.decimal("minSpeechProbability", "Min speech probability",
                0.5d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.decimal("preRollMs", "Pre-roll (ms)", 200.0d, 0.0d, 5000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("postRollMs", "Post-roll (ms)", 350.0d, 0.0d, 5000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("minRetainedMs", "Minimum retained (ms)",
                400.0d, 0.0d, 60000.0d, 50.0d));
        params.add(AudioParameterDescriptor.choice("noSpeechBehavior", "No-speech behavior", "KEEP_ORIGINAL",
                Arrays.asList(new AudioParameterChoice("KEEP_ORIGINAL", "Keep original"),
                        new AudioParameterChoice("FAIL", "Fail"))));
        params.add(AudioParameterDescriptor.bool("zeroCrossingAlignment", "Zero-crossing alignment", true));
        params.add(AudioParameterDescriptor.decimal("zeroCrossingSearchMs", "Zero-crossing search (ms)",
                5.0d, 0.0d, 100.0d, 1.0d));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .streaming(false)
                .requiresCompleteSignal(true)
                .requiresSpeechActivityTrack(true)
                .consumesSpeechMetadata(true)
                .changesDuration(true)
                .changesSampleCount(true)
                .build();
        return descriptor(AudioBlockType.SILENCE_TRIMMER, AudioBlockCategory.OUTPUT, params, capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.silenceTrimmer();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        boolean lead = block.getBooleanParameter("trimLeading", true);
                        boolean trail = block.getBooleanParameter("trimTrailing", true);
                        String scope = lead && trail ? "Leading + trailing"
                                : lead ? "Leading" : trail ? "Trailing" : "Off";
                        return scope + " · " + formatHz0(block.getDoubleParameter("preRollMs", 0.0d)) + "/"
                                + formatHz0(block.getDoubleParameter("postRollMs", 0.0d)) + " ms";
                    }
                });
    }

    private static AudioBlockDescriptor deEsser() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("targetFrequencyHz", "Target frequency (Hz)",
                6500.0d, 1000.0d, 20000.0d, 100.0d));
        params.add(AudioParameterDescriptor.decimal("bandwidthHz", "Bandwidth (Hz)",
                2500.0d, 100.0d, 12000.0d, 100.0d));
        params.add(AudioParameterDescriptor.decimal("thresholdDb", "Threshold (dBFS)", -30.0d, -80.0d, 0.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("reductionDb", "Reduction (dB)", 8.0d, 0.0d, 40.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("attackMs", "Attack (ms)", 2.0d, 0.0d, 200.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("releaseMs", "Release (ms)", 60.0d, 1.0d, 2000.0d, 5.0d));
        return descriptor(AudioBlockType.DE_ESSER, AudioBlockCategory.SPEECH_ENHANCEMENT, params,
                StaticBlockCapabilities.builder().framing(320, 0, 0).build(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.deEsser();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return formatHz0(block.getDoubleParameter("targetFrequencyHz", 0.0d)) + " Hz · -"
                                + formatQ(block.getDoubleParameter("reductionDb", 0.0d)) + " dB max";
                    }
                });
    }

    private static AudioBlockDescriptor adaptiveHumRemoval() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("baseFrequencyHz", "Base frequency (Hz)",
                50.0d, 20.0d, 500.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("searchRangeHz", "Search range (Hz)", 3.0d, 0.0d, 20.0d, 0.5d));
        params.add(AudioParameterDescriptor.decimal("adaptationSpeed", "Adaptation speed", 0.1d, 0.0d, 1.0d, 0.01d));
        params.add(AudioParameterDescriptor.integer("harmonics", "Harmonics", 3, 1, 12));
        params.add(AudioParameterDescriptor.decimal("maxAttenuationDb", "Maximum attenuation (dB)",
                24.0d, 0.0d, 80.0d, 1.0d));
        params.add(AudioParameterDescriptor.bool("speechProtection", "Speech protection", false));
        return descriptor(AudioBlockType.ADAPTIVE_HUM_REMOVAL, AudioBlockCategory.NOISE_REDUCTION, params,
                StaticBlockCapabilities.builder().framing(320, 0, 0).consumesSpeechMetadata(true).build(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.adaptiveHumRemoval();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return formatHz0(block.getDoubleParameter("baseFrequencyHz", 0.0d)) + " Hz · "
                                + block.getIntParameter("harmonics", 3) + " harmonics";
                    }
                });
    }

    private static AudioBlockDescriptor plosiveReduction() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("strength", "Strength", 0.6d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.decimal("targetFrequencyHz", "Target frequency (Hz)",
                120.0d, 20.0d, 500.0d, 5.0d));
        params.add(AudioParameterDescriptor.decimal("attackMs", "Attack (ms)", 3.0d, 0.0d, 200.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("releaseMs", "Release (ms)", 80.0d, 1.0d, 2000.0d, 5.0d));
        return descriptor(AudioBlockType.PLOSIVE_REDUCTION, AudioBlockCategory.SPEECH_ENHANCEMENT, params,
                StaticBlockCapabilities.builder().framing(320, 0, 0).build(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.plosiveReduction();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "< " + formatHz0(block.getDoubleParameter("targetFrequencyHz", 0.0d))
                                + " Hz · strength " + formatQ(block.getDoubleParameter("strength", 0.0d));
                    }
                });
    }

    private static AudioBlockDescriptor breathReduction() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("sensitivity", "Sensitivity", 0.5d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.decimal("maxAttenuationDb", "Maximum attenuation (dB)",
                12.0d, 0.0d, 80.0d, 1.0d));
        params.add(AudioParameterDescriptor.bool("speechProtection", "Speech protection", true));
        params.add(AudioParameterDescriptor.decimal("attackMs", "Attack (ms)", 5.0d, 0.0d, 500.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("releaseMs", "Release (ms)", 120.0d, 1.0d, 5000.0d, 5.0d));
        return descriptor(AudioBlockType.BREATH_REDUCTION, AudioBlockCategory.SPEECH_ENHANCEMENT, params,
                StaticBlockCapabilities.builder().framing(320, 0, 0).consumesSpeechMetadata(true).build(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.breathReduction();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "sensitivity " + formatQ(block.getDoubleParameter("sensitivity", 0.0d))
                                + " · -" + formatQ(block.getDoubleParameter("maxAttenuationDb", 0.0d)) + " dB max";
                    }
                });
    }

    // ------------------------------------------------------------------ FFT (STFT) variants

    private static AudioBlockDescriptor deEsserFft() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("targetFrequencyHz", "Target frequency (Hz)",
                6500.0d, 1000.0d, 20000.0d, 100.0d));
        params.add(AudioParameterDescriptor.decimal("bandwidthHz", "Bandwidth (Hz)",
                3000.0d, 100.0d, 12000.0d, 100.0d));
        params.add(AudioParameterDescriptor.decimal("thresholdDb", "Threshold (dBFS)", -30.0d, -80.0d, 0.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("reductionDb", "Reduction (dB)", 8.0d, 0.0d, 40.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("attackMs", "Attack (ms)", 5.0d, 0.0d, 200.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("releaseMs", "Release (ms)", 60.0d, 1.0d, 2000.0d, 5.0d));
        return descriptor(AudioBlockType.DE_ESSER_FFT, AudioBlockCategory.SPEECH_ENHANCEMENT, params,
                spectralCapabilities(false),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.deEsserFft();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "FFT · " + formatHz0(block.getDoubleParameter("targetFrequencyHz", 0.0d))
                                + " Hz · -" + formatQ(block.getDoubleParameter("reductionDb", 0.0d)) + " dB max";
                    }
                });
    }

    private static AudioBlockDescriptor adaptiveHumRemovalFft() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("baseFrequencyHz", "Base frequency (Hz)",
                50.0d, 20.0d, 500.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("searchRangeHz", "Search range (Hz)", 3.0d, 0.0d, 20.0d, 0.5d));
        params.add(AudioParameterDescriptor.decimal("adaptationSpeed", "Adaptation speed", 0.1d, 0.0d, 1.0d, 0.01d));
        params.add(AudioParameterDescriptor.integer("harmonics", "Harmonics", 3, 1, 12));
        params.add(AudioParameterDescriptor.decimal("maxAttenuationDb", "Maximum attenuation (dB)",
                24.0d, 0.0d, 80.0d, 1.0d));
        params.add(AudioParameterDescriptor.bool("speechProtection", "Speech protection", false));
        return descriptor(AudioBlockType.ADAPTIVE_HUM_REMOVAL_FFT, AudioBlockCategory.NOISE_REDUCTION, params,
                spectralCapabilities(true),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.adaptiveHumRemovalFft();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "FFT · " + formatHz0(block.getDoubleParameter("baseFrequencyHz", 0.0d))
                                + " Hz · " + block.getIntParameter("harmonics", 3) + " harmonics";
                    }
                });
    }

    private static AudioBlockDescriptor plosiveReductionFft() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("strength", "Strength", 0.6d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.decimal("targetFrequencyHz", "Target frequency (Hz)",
                120.0d, 20.0d, 500.0d, 5.0d));
        params.add(AudioParameterDescriptor.decimal("attackMs", "Attack (ms)", 5.0d, 0.0d, 200.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("releaseMs", "Release (ms)", 80.0d, 1.0d, 2000.0d, 5.0d));
        return descriptor(AudioBlockType.PLOSIVE_REDUCTION_FFT, AudioBlockCategory.SPEECH_ENHANCEMENT, params,
                spectralCapabilities(false),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.plosiveReductionFft();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "FFT · < " + formatHz0(block.getDoubleParameter("targetFrequencyHz", 0.0d))
                                + " Hz · strength " + formatQ(block.getDoubleParameter("strength", 0.0d));
                    }
                });
    }

    private static AudioBlockDescriptor breathReductionFft() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("sensitivity", "Sensitivity", 0.5d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.decimal("maxAttenuationDb", "Maximum attenuation (dB)",
                12.0d, 0.0d, 80.0d, 1.0d));
        params.add(AudioParameterDescriptor.bool("speechProtection", "Speech protection", true));
        params.add(AudioParameterDescriptor.decimal("attackMs", "Attack (ms)", 5.0d, 0.0d, 500.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("releaseMs", "Release (ms)", 120.0d, 1.0d, 5000.0d, 5.0d));
        return descriptor(AudioBlockType.BREATH_REDUCTION_FFT, AudioBlockCategory.SPEECH_ENHANCEMENT, params,
                spectralCapabilities(true),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.breathReductionFft();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "FFT · sensitivity " + formatQ(block.getDoubleParameter("sensitivity", 0.0d))
                                + " · -" + formatQ(block.getDoubleParameter("maxAttenuationDb", 0.0d)) + " dB max";
                    }
                });
    }

    // ------------------------------------------------------------------ noise profiling / suppression

    private static AudioBlockDescriptor noiseProfiler() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.choice("mode", "Mode", "AUTOMATIC", Arrays.asList(
                new AudioParameterChoice("AUTOMATIC", "Learn from detected speech pauses"),
                new AudioParameterChoice("LEARN_FROM_SILENCE", "Treat the whole recording as noise"),
                new AudioParameterChoice("USE_EXISTING", "Keep an existing learned profile"))));
        params.add(AudioParameterDescriptor.decimal("learnTimeMs", "Learn time (ms, 0 = all)",
                0.0d, 0.0d, 600000.0d, 100.0d));
        params.add(AudioParameterDescriptor.decimal("minConfidence", "Minimum confidence",
                0.2d, 0.0d, 1.0d, 0.05d));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .modifiesAudio(false)
                .producesMetadata(true)
                .consumesSpeechMetadata(true)
                .framing(1024, 0, 512)
                .build();
        return descriptor(AudioBlockType.NOISE_PROFILER, AudioBlockCategory.ANALYSIS, params, capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.noiseProfiler();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Learns noise · " + block.getParameter("mode", "AUTOMATIC");
                    }
                });
    }

    private static AudioBlockDescriptor adaptiveNoiseSuppression() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.choice("mode", "Mode", "AUTOMATIC", Arrays.asList(
                new AudioParameterChoice("AUTOMATIC", "Automatic (track the noise floor)"),
                new AudioParameterChoice("LEARN_FROM_SILENCE", "Learn from silence"),
                new AudioParameterChoice("USE_FIXED_PROFILE", "Use a learned noise profile"))));
        params.add(AudioParameterDescriptor.decimal("maxAttenuationDb", "Maximum attenuation (dB)",
                12.0d, 0.0d, 80.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("adaptationSpeed", "Adaptation speed", 0.1d, 0.0d, 1.0d, 0.01d));
        params.add(AudioParameterDescriptor.decimal("noiseFloorDb", "Noise floor (dBFS)",
                -60.0d, -120.0d, 0.0d, 1.0d));
        params.add(AudioParameterDescriptor.bool("speechProtection", "Speech protection", true));
        params.add(AudioParameterDescriptor.decimal("minSpeechProbability", "Min speech probability",
                0.5d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.bool("adaptDuringSpeech", "Adapt during speech", false));
        params.add(AudioParameterDescriptor.bool("freezeProfile", "Freeze learned profile", false));
        params.add(AudioParameterDescriptor.decimal("artifactProtection", "Artifact protection",
                0.4d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.decimal("attackMs", "Attack (ms)", 15.0d, 0.0d, 500.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("releaseMs", "Release (ms)", 120.0d, 1.0d, 5000.0d, 5.0d));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .framing(1024, 0, 512)
                .consumesSpeechMetadata(true)
                .build();
        return descriptor(AudioBlockType.ADAPTIVE_NOISE_SUPPRESSION, AudioBlockCategory.NOISE_REDUCTION,
                params, capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.adaptiveNoiseSuppression();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "-" + formatQ(block.getDoubleParameter("maxAttenuationDb", 0.0d))
                                + " dB max · " + block.getParameter("mode", "AUTOMATIC");
                    }
                });
    }

    // ------------------------------------------------------------------ multichannel

    private static AudioBlockDescriptor channelSelector() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.integer("channelIndex", "Channel index (0-based)", 0, 0, 31));
        params.add(AudioParameterDescriptor.integer("fallbackChannel", "Fallback channel", 0, 0, 31));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .preservesChannelCount(false)
                .build();
        return descriptor(AudioBlockType.CHANNEL_SELECTOR, AudioBlockCategory.INPUT_CHANNEL, params, capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.channelSelector();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Channel " + block.getIntParameter("channelIndex", 0) + " -> mono";
                    }
                });
    }

    private static AudioBlockDescriptor matrixMixer() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.text("weights", "Input weights (comma-separated)", "0.5,0.5"));
        params.add(AudioParameterDescriptor.bool("normalize", "Normalize by total weight", true));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .preservesChannelCount(false)
                .build();
        return descriptor(AudioBlockType.MATRIX_MIXER, AudioBlockCategory.INPUT_CHANNEL, params, capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.matrixMixer();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Downmix [" + block.getParameter("weights", "") + "] -> mono";
                    }
                });
    }

    private static AudioBlockDescriptor channelGainPolarity() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.text("gainsDb", "Per-channel gain (dB, comma-separated)", "0,0"));
        params.add(AudioParameterDescriptor.text("polarityInvert", "Invert polarity (per channel, 0/1)", "0,0"));
        return descriptor(AudioBlockType.CHANNEL_GAIN_POLARITY, AudioBlockCategory.INPUT_CHANNEL, params,
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.channelGainPolarity();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Gain [" + block.getParameter("gainsDb", "") + "] dB";
                    }
                });
    }

    private static AudioBlockDescriptor phaseCorrelationAnalyzer() {
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .modifiesAudio(false)
                .producesMetadata(true)
                .build();
        return descriptor(AudioBlockType.PHASE_CORRELATION_ANALYZER, AudioBlockCategory.ANALYSIS,
                Collections.<AudioParameterDescriptor>emptyList(), capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.phaseCorrelationAnalyzer();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Stereo correlation / phase check";
                    }
                });
    }

    private static AudioBlockDescriptor channelDelayAlignment() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.choice("mode", "Mode", "AUTO", Arrays.asList(
                new AudioParameterChoice("AUTO", "Automatic (cross-correlation)"),
                new AudioParameterChoice("MANUAL", "Manual delays"))));
        params.add(AudioParameterDescriptor.integer("referenceChannel", "Reference channel", 0, 0, 31));
        params.add(AudioParameterDescriptor.integer("maxCorrectionSamples", "Max correction (samples)",
                64, 1, 4096));
        params.add(AudioParameterDescriptor.bool("fractionalDelay", "Fractional delay", true));
        params.add(AudioParameterDescriptor.text("delaysSamples", "Manual delays (samples, per channel)", ""));
        return descriptor(AudioBlockType.CHANNEL_DELAY_ALIGNMENT, AudioBlockCategory.INPUT_CHANNEL, params,
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.channelDelayAlignment();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Align to channel " + block.getIntParameter("referenceChannel", 0)
                                + " · " + block.getParameter("mode", "AUTO");
                    }
                });
    }

    private static AudioBlockDescriptor bestChannelSelector() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.integer("preferredChannel", "Preferred channel (-1 = auto)",
                -1, -1, 31));
        params.add(AudioParameterDescriptor.decimal("evaluationWindowMs", "Evaluation window (ms)",
                500.0d, 50.0d, 10000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("minHoldMs", "Minimum hold (ms)", 500.0d, 0.0d, 10000.0d, 10.0d));
        params.add(AudioParameterDescriptor.bool("switchDuringSpeech", "Allow switching during speech", false));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .preservesChannelCount(false)
                .build();
        return descriptor(AudioBlockType.BEST_CHANNEL_SELECTOR, AudioBlockCategory.INPUT_CHANNEL, params,
                capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.bestChannelSelector();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        int pref = block.getIntParameter("preferredChannel", -1);
                        return (pref < 0 ? "Auto-select best" : "Prefer channel " + pref) + " -> mono";
                    }
                });
    }

    private static AudioBlockDescriptor channelHealthAnalyzer() {
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .modifiesAudio(false)
                .producesMetadata(true)
                .build();
        return descriptor(AudioBlockType.CHANNEL_HEALTH_ANALYZER, AudioBlockCategory.ANALYSIS,
                Collections.<AudioParameterDescriptor>emptyList(), capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.channelHealthAnalyzer();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Detects silent/clipping/DC channels";
                    }
                });
    }

    private static AudioBlockDescriptor midSideProcessor() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("midGainDb", "Mid gain (dB)", 0.0d, -24.0d, 24.0d, 0.5d));
        params.add(AudioParameterDescriptor.decimal("sideGainDb", "Side gain (dB)", 0.0d, -24.0d, 24.0d, 0.5d));
        params.add(AudioParameterDescriptor.decimal("sideReduction", "Side reduction", 0.0d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.bool("monoCompatibilityProtection",
                "Mono compatibility protection", true));
        return descriptor(AudioBlockType.MID_SIDE_PROCESSOR, AudioBlockCategory.INPUT_CHANNEL, params,
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.midSideProcessor();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Mid " + formatDb(block.getDoubleParameter("midGainDb", 0.0d)) + " · Side "
                                + formatDb(block.getDoubleParameter("sideGainDb", 0.0d));
                    }
                });
    }

    private static AudioBlockDescriptor centerSpeechExtractor() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("centerAmount", "Center amount", 0.6d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.decimal("centerBoostDb", "Center boost (dB)", 0.0d, 0.0d, 12.0d, 0.5d));
        return descriptor(AudioBlockType.CENTER_SPEECH_EXTRACTOR, AudioBlockCategory.SPEECH_ENHANCEMENT, params,
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.centerSpeechExtractor();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Center " + formatQ(block.getDoubleParameter("centerAmount", 0.0d))
                                + " (stereo only)";
                    }
                });
    }

    private static AudioBlockDescriptor stereoWidthControl() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("width", "Width", 1.0d, 0.0d, 2.0d, 0.05d));
        params.add(AudioParameterDescriptor.bool("monoCompatibilityProtection",
                "Mono compatibility protection", true));
        return descriptor(AudioBlockType.STEREO_WIDTH_CONTROL, AudioBlockCategory.INPUT_CHANNEL, params,
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.stereoWidthControl();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Width " + formatQ(block.getDoubleParameter("width", 1.0d));
                    }
                });
    }

    private static AudioBlockDescriptor speechEnhancer() {
        List<AudioParameterChoice> backends = new ArrayList<AudioParameterChoice>();
        for (com.aresstack.audio.enhance.SpeechEnhancementBackend backend
                : com.aresstack.audio.enhance.SpeechEnhancementBackends.all()) {
            backends.add(new AudioParameterChoice(backend.id(), backend.displayName()));
        }
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.choice("backend", "Backend", "PURE_JAVA_DSP", backends));
        params.add(AudioParameterDescriptor.decimal("strength", "Strength", 0.6d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.bool("speechProtection", "Speech protection", true));
        params.add(AudioParameterDescriptor.decimal("artifactProtection", "Artifact protection",
                0.4d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.integer("targetSampleRate", "Target sample rate (Hz, 0 = any)",
                0, 0, 192000));
        params.add(AudioParameterDescriptor.text("modelId", "Model / profile id", ""));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .consumesSpeechMetadata(true)
                .build();
        return descriptor(AudioBlockType.SPEECH_ENHANCER, AudioBlockCategory.SPEECH_ENHANCEMENT, params,
                capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.speechEnhancer();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return block.getParameter("backend", "PURE_JAVA_DSP") + " · strength "
                                + formatQ(block.getDoubleParameter("strength", 0.0d));
                    }
                });
    }

    private static AudioBlockDescriptor voiceIsolation() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.choice("backend", "Backend", "PURE_JAVA_CENTER", Arrays.asList(
                new AudioParameterChoice("PURE_JAVA_CENTER", "Pure Java (stereo center)"),
                new AudioParameterChoice("NEURAL", "Neural (optional, not installed)"))));
        params.add(AudioParameterDescriptor.decimal("strength", "Strength", 0.7d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.text("modelId", "Model / profile id", ""));
        return descriptor(AudioBlockType.VOICE_ISOLATION, AudioBlockCategory.SPEECH_ENHANCEMENT, params,
                StaticBlockCapabilities.audioEffect(),
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.voiceIsolation();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return block.getParameter("backend", "PURE_JAVA_CENTER") + " · strength "
                                + formatQ(block.getDoubleParameter("strength", 0.0d));
                    }
                });
    }

    private static AudioBlockDescriptor directionOfArrivalAnalyzer() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.text("micPositionsMm",
                "Microphone positions (mm; x,y,z per mic, ';' separated)", ""));
        params.add(AudioParameterDescriptor.decimal("speedOfSoundMmPerS", "Speed of sound (mm/s)",
                343000.0d, 300000.0d, 360000.0d, 1000.0d));
        params.add(AudioParameterDescriptor.integer("maxLagSamples", "Max inter-mic lag (samples)", 32, 1, 512));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .modifiesAudio(false)
                .producesMetadata(true)
                .requiresKnownMicrophoneGeometry(true)
                .build();
        return descriptor(AudioBlockType.DIRECTION_OF_ARRIVAL_ANALYZER, AudioBlockCategory.ANALYSIS, params,
                capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.directionOfArrivalAnalyzer();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Estimates source azimuth";
                    }
                });
    }

    private static AudioBlockDescriptor delayAndSumBeamformer() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.text("micPositionsMm",
                "Microphone positions (mm; x,y,z per mic, ';' separated)", ""));
        params.add(AudioParameterDescriptor.decimal("targetAzimuthDeg", "Target azimuth (deg)",
                90.0d, -180.0d, 180.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("targetElevationDeg", "Target elevation (deg)",
                0.0d, -90.0d, 90.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("speedOfSoundMmPerS", "Speed of sound (mm/s)",
                343000.0d, 300000.0d, 360000.0d, 1000.0d));
        params.add(AudioParameterDescriptor.text("channelWeights", "Channel weights (comma-separated)", ""));
        params.add(AudioParameterDescriptor.decimal("outputGainDb", "Output gain (dB)", 0.0d, -24.0d, 24.0d, 0.5d));
        params.add(AudioParameterDescriptor.bool("tracking", "Track a moving speaker", false));
        params.add(AudioParameterDescriptor.integer("trackingBlockFrames", "Tracking block (samples)",
                512, 64, 16384));
        params.add(AudioParameterDescriptor.integer("maxLagSamples", "Max inter-mic lag (samples)", 32, 1, 512));
        params.add(AudioParameterDescriptor.decimal("directionSmoothing", "Direction smoothing",
                0.7d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.decimal("maxAngularSpeedDegPerBlock", "Max angular speed (deg/block)",
                15.0d, 0.0d, 180.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("minConfidence", "Minimum confidence", 0.2d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.integer("holdBlocks", "Hold blocks", 3, 0, 100));
        params.add(AudioParameterDescriptor.decimal("fallbackAzimuthDeg", "Fallback azimuth (deg)",
                90.0d, -180.0d, 180.0d, 1.0d));
        params.add(AudioParameterDescriptor.bool("updateDuringSilence", "Update during silence", false));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .preservesChannelCount(false)
                .requiresSynchronizedChannels(true)
                .requiresKnownMicrophoneGeometry(true)
                .build();
        return descriptor(AudioBlockType.DELAY_AND_SUM_BEAMFORMER, AudioBlockCategory.INPUT_CHANNEL, params,
                capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.delayAndSumBeamformer();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Beam @ " + formatQ(block.getDoubleParameter("targetAzimuthDeg", 0.0d))
                                + " deg -> mono";
                    }
                });
    }

    private static AudioBlockDescriptor speechLeveler() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("targetSpeechLevelDb", "Target speech level (dBFS RMS)",
                -20.0d, -40.0d, 0.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("maxGainDb", "Maximum gain (dB)", 18.0d, 0.0d, 48.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("maxAttenuationDb", "Maximum attenuation (dB)",
                12.0d, 0.0d, 48.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("attackMs", "Attack (ms)", 120.0d, 1.0d, 2000.0d, 5.0d));
        params.add(AudioParameterDescriptor.decimal("releaseMs", "Release (ms)", 1000.0d, 1.0d, 8000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("holdMs", "Hold (ms)", 300.0d, 0.0d, 4000.0d, 10.0d));
        params.add(AudioParameterDescriptor.decimal("maxGainChangePerSecond", "Max gain change (dB/s)",
                9.0d, 0.1d, 60.0d, 0.5d));
        params.add(AudioParameterDescriptor.decimal("minSpeechProbability", "Min speech probability",
                0.5d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.decimal("silenceGainLimitDb", "Silence gain limit (dB)",
                6.0d, 0.0d, 24.0d, 1.0d));
        params.add(AudioParameterDescriptor.bool("noiseProtection", "Noise protection", true));
        params.add(AudioParameterDescriptor.bool("clippingProtection", "Clipping protection", true));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .framing(320, 0, 0)
                .consumesSpeechMetadata(true)
                .build();
        return descriptor(AudioBlockType.SPEECH_LEVELER, AudioBlockCategory.DYNAMICS, params, capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.speechLeveler();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return formatQ(block.getDoubleParameter("targetSpeechLevelDb", 0.0d)) + " dBFS · +"
                                + formatQ(block.getDoubleParameter("maxGainDb", 0.0d)) + "/-"
                                + formatQ(block.getDoubleParameter("maxAttenuationDb", 0.0d)) + " dB";
                    }
                });
    }

    private static AudioBlockDescriptor finalLoudnessNormalizer() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.choice("mode", "Mode", "TARGET_RMS", Arrays.asList(
                new AudioParameterChoice("TARGET_RMS", "Target RMS level"),
                new AudioParameterChoice("PEAK", "Peak normalization"))));
        params.add(AudioParameterDescriptor.decimal("targetLevelDb", "Target level (dBFS)",
                -20.0d, -60.0d, 0.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("maxTotalGainDb", "Maximum total gain (dB)",
                24.0d, 0.0d, 60.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("maxTotalAttenuationDb", "Maximum total attenuation (dB)",
                24.0d, 0.0d, 60.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("peakCeilingDb", "Peak ceiling (dBFS)",
                -1.0d, -30.0d, 0.0d, 0.5d));
        params.add(AudioParameterDescriptor.bool("clippingProtection", "Clipping protection", true));
        params.add(AudioParameterDescriptor.bool("allowAmplification", "Allow amplification", true));
        params.add(AudioParameterDescriptor.bool("allowAttenuation", "Allow attenuation", true));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .streaming(false)
                .requiresCompleteSignal(true)
                .build();
        return descriptor(AudioBlockType.FINAL_LOUDNESS_NORMALIZER, AudioBlockCategory.OUTPUT, params,
                capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.finalLoudnessNormalizer();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return block.getParameter("mode", "TARGET_RMS") + " · "
                                + formatQ(block.getDoubleParameter("targetLevelDb", 0.0d)) + " dBFS · ceiling "
                                + formatQ(block.getDoubleParameter("peakCeilingDb", 0.0d)) + " dB";
                    }
                });
    }

    private static AudioBlockDescriptor roomReverbAnalyzer() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.decimal("frameDurationMs", "Frame duration (ms)",
                20.0d, 5.0d, 100.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("minDecayDb", "Minimum decay (dB)", 6.0d, 1.0d, 60.0d, 1.0d));
        params.add(AudioParameterDescriptor.decimal("maxReverbSeconds", "Maximum reverb time (s)",
                3.0d, 0.1d, 10.0d, 0.1d));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .modifiesAudio(false)
                .producesMetadata(true)
                .framing(320, 0, 0)
                .build();
        return descriptor(AudioBlockType.ROOM_REVERB_ANALYZER, AudioBlockCategory.ANALYSIS, params, capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.roomReverbAnalyzer();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "Estimates reverb (RT60, strength)";
                    }
                });
    }

    private static AudioBlockDescriptor dereverberation() {
        List<AudioParameterDescriptor> params = new ArrayList<AudioParameterDescriptor>();
        params.add(AudioParameterDescriptor.choice("mode", "Processing mode", "OFFLINE",
                Arrays.asList(new AudioParameterChoice("OFFLINE", "Offline (whole recording)"),
                        new AudioParameterChoice("BLOCK_ADAPTIVE", "Block-adaptive"),
                        new AudioParameterChoice("STREAMING", "Streaming adaptive"))));
        params.add(AudioParameterDescriptor.decimal("strength", "Strength", 0.7d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.integer("predictionDelay", "Prediction delay (frames)", 2, 1, 32));
        params.add(AudioParameterDescriptor.integer("filterLength", "Filter length (taps)", 8, 1, 32));
        params.add(AudioParameterDescriptor.integer("iterations", "Iterations", 3, 1, 10));
        params.add(AudioParameterDescriptor.decimal("earlyReflectionPreservation",
                "Early reflection preservation", 0.5d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.decimal("adaptationSpeed", "Adaptation speed (block-adaptive)",
                0.3d, 0.0d, 1.0d, 0.05d));
        params.add(AudioParameterDescriptor.integer("blockSizeFrames", "Block size (frames)", 64, 8, 4096));
        params.add(AudioParameterDescriptor.bool("speechProtection", "Speech protection", false));
        params.add(AudioParameterDescriptor.decimal("artifactProtection", "Artifact protection",
                0.2d, 0.0d, 1.0d, 0.05d));
        AudioBlockCapabilities capabilities = StaticBlockCapabilities.builder()
                .framing(512, 1024, 8192) // streaming mode uses ~8 frames look-ahead, ~64 frames history
                .consumesSpeechMetadata(true)
                .build();
        return descriptor(AudioBlockType.DEREVERBERATION, AudioBlockCategory.DEREVERBERATION, params,
                capabilities,
                new SimpleAudioBlockDescriptor.ProcessorFactory() {
                    public AudioBlockProcessor create() {
                        return AudioBlockProcessors.dereverberation();
                    }
                },
                new SimpleAudioBlockDescriptor.Summarizer() {
                    public String summarize(AudioBlockDefinition block) {
                        return "WPE · " + block.getParameter("mode", "OFFLINE") + " · strength "
                                + formatQ(block.getDoubleParameter("strength", 0.0d));
                    }
                });
    }

    private static AudioBlockCapabilities spectralCapabilities(boolean consumesSpeechMetadata) {
        return StaticBlockCapabilities.builder()
                .framing(1024, 0, 512)
                .consumesSpeechMetadata(consumesSpeechMetadata)
                .build();
    }

    private static String formatHz0(double value) {
        return String.format(java.util.Locale.ROOT, "%.0f", value);
    }

    private static String formatDb(double value) {
        return (value >= 0.0d ? "+" : "") + String.format(java.util.Locale.ROOT, "%.1f", value) + " dB";
    }

    private static String formatHz(double value) {
        return String.format(java.util.Locale.ROOT, "%.0f", value) + " Hz";
    }

    private static String formatQ(double value) {
        return String.format(java.util.Locale.ROOT, "%.1f", value);
    }

    private static SimpleAudioBlockDescriptor.Summarizer filterSummary() {
        return new SimpleAudioBlockDescriptor.Summarizer() {
            public String summarize(AudioBlockDefinition block) {
                return block.getParameter("cutoffHz", "") + " Hz · "
                        + filterDesignLabel(block.getParameter("implementation", ""));
            }
        };
    }

    private static String filterDesignLabel(String value) {
        if ("FIR_65".equals(value)) {
            return "65-tap FIR";
        }
        if ("LEGACY_IIR".equals(value)) {
            return "1st-order IIR";
        }
        if ("BUTTERWORTH".equals(value)) {
            return "Butterworth";
        }
        return value;
    }

    private static AudioBlockDescriptor descriptor(AudioBlockType type, AudioBlockCategory category,
                                                   List<AudioParameterDescriptor> params,
                                                   AudioBlockCapabilities capabilities,
                                                   SimpleAudioBlockDescriptor.ProcessorFactory factory,
                                                   SimpleAudioBlockDescriptor.Summarizer summarizer) {
        return new SimpleAudioBlockDescriptor(type, type.getDisplayName(), category, params,
                capabilities, factory, summarizer);
    }
}

package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;
import com.aresstack.audio.profile.AudioProcessingProfile;

import java.util.ArrayList;
import java.util.List;

/**
 * Validate a profile without silently reordering it: report understandable, centralized problems (invalid
 * band/filter range, out-of-range resampler rate, a filter above Nyquist for a given input rate, invalid
 * equalizer or VAD parameters, non-finite or non-parseable values) so the editor can show them instead of
 * rearranging the user's pipeline or hiding the problem behind the processors' defensive runtime bypass.
 *
 * <p>Findings are produced as {@link AudioProfileValidationIssue}s (severity, block, parameter, message).
 * The legacy {@link #validate(AudioProcessingProfile)} API is preserved by mapping issues to {@link Message}.
 * A disabled block is still validated structurally, but its errors are surfaced as warnings: they stay
 * visible without blocking processing, because a bypassed block never runs.</p>
 */
public final class AudioProfileValidator {

    public enum Severity {
        ERROR,
        WARNING,
        INFO
    }

    /** One validation finding, tied to a block when applicable (legacy shape, kept for existing callers). */
    public static final class Message {
        private final Severity severity;
        private final String blockId;
        private final String text;

        public Message(Severity severity, String blockId, String text) {
            this.severity = severity;
            this.blockId = blockId;
            this.text = text;
        }

        public Severity getSeverity() {
            return severity;
        }

        public String getBlockId() {
            return blockId;
        }

        public String getText() {
            return text;
        }
    }

    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    /** Validate structural correctness that does not depend on the input rate (legacy message list). */
    public List<Message> validate(AudioProcessingProfile profile) {
        return validate(profile, null);
    }

    /** Validate the profile against the running Nyquist frequency when a format is given (legacy list). */
    public List<Message> validate(AudioProcessingProfile profile, PcmAudioFormat inputFormat) {
        return toMessages(validateResult(profile, inputFormat));
    }

    /** Validate and return the rich, Swing-free result the editor uses to mark blocks and parameters. */
    public AudioProfileValidationResult validateResult(AudioProcessingProfile profile) {
        return validateResult(profile, null);
    }

    public AudioProfileValidationResult validateResult(AudioProcessingProfile profile,
                                                       PcmAudioFormat inputFormat) {
        List<AudioProfileValidationIssue> issues = new ArrayList<AudioProfileValidationIssue>();
        if (profile == null) {
            return new AudioProfileValidationResult(issues);
        }
        int currentRate = inputFormat == null ? 0 : inputFormat.getSampleRateHz();
        int currentChannels = inputFormat == null ? 0 : inputFormat.getChannels();
        boolean sawChannelMixer = false;
        boolean sawEnabledVad = false;
        boolean sawEnabledNoiseProfiler = false;
        boolean timeBaseChangedAfterVad = false;
        int silenceTrimmerCount = 0;
        int loudnessNormalizerCount = 0;
        List<AudioBlockDefinition> blocks = profile.getBlocks();
        for (int i = 0; i < blocks.size(); i++) {
            AudioBlockDefinition block = blocks.get(i);
            AudioBlockType type = block.getType();
            boolean enabled = block.isEnabled();
            if (type == AudioBlockType.CHANNEL_MIXER) {
                sawChannelMixer = true;
            }
            if (type == AudioBlockType.VOICE_ACTIVITY_DETECTION && enabled) {
                sawEnabledVad = true;
            }
            if (type == AudioBlockType.NOISE_PROFILER && enabled) {
                sawEnabledNoiseProfiler = true;
            }
            validateParseable(issues, block, enabled);
            switch (type) {
                case BAND_PASS:
                case BAND_STOP:
                    validateBand(issues, block, currentRate, enabled);
                    break;
                case LOW_PASS:
                case HIGH_PASS:
                    validateCutoff(issues, block, currentRate, enabled);
                    break;
                case RESAMPLER:
                    validateResampler(issues, block, sawChannelMixer, enabled);
                    if (enabled) {
                        currentRate = block.getIntParameter("targetRateHz", currentRate);
                    }
                    break;
                case GAIN:
                    validateGain(issues, block, enabled);
                    break;
                case PARAMETRIC_EQ:
                    validateEqBand(issues, block, "centerHz", currentRate, true, enabled);
                    break;
                case LOW_SHELF:
                case HIGH_SHELF:
                    validateEqBand(issues, block, "cutoffHz", currentRate, false, enabled);
                    break;
                case EQUALIZER:
                    validateEqualizer(issues, block, currentRate, enabled);
                    break;
                case VOICE_ACTIVITY_DETECTION:
                    validateVoiceActivity(issues, block, enabled);
                    break;
                case SPEECH_GATE:
                    validateSpeechGate(issues, block, enabled, sawEnabledVad, timeBaseChangedAfterVad);
                    break;
                case EXPANDER:
                    validateExpander(issues, block, enabled, sawEnabledVad);
                    break;
                case SILENCE_TRIMMER:
                    silenceTrimmerCount++;
                    validateSilenceTrimmer(issues, block, enabled, sawEnabledVad,
                            timeBaseChangedAfterVad, silenceTrimmerCount);
                    break;
                case DE_ESSER:
                case DE_ESSER_FFT:
                    validateDeEsser(issues, block, currentRate, enabled);
                    break;
                case ADAPTIVE_HUM_REMOVAL:
                case ADAPTIVE_HUM_REMOVAL_FFT:
                    validateAdaptiveHumRemoval(issues, block, currentRate, enabled, sawEnabledVad);
                    break;
                case PLOSIVE_REDUCTION:
                case PLOSIVE_REDUCTION_FFT:
                    validatePlosiveReduction(issues, block, currentRate, enabled);
                    break;
                case BREATH_REDUCTION:
                case BREATH_REDUCTION_FFT:
                    validateBreathReduction(issues, block, enabled, sawEnabledVad);
                    break;
                case NOISE_PROFILER:
                    validateNoiseProfiler(issues, block, enabled);
                    break;
                case ROOM_REVERB_ANALYZER:
                    validateRoomReverbAnalyzer(issues, block, enabled);
                    break;
                case DEREVERBERATION:
                    validateDereverberation(issues, block, enabled);
                    break;
                case CHANNEL_SELECTOR:
                    if (currentChannels == 1) {
                        add(issues, block, enabled, AudioValidationSeverity.WARNING, null,
                                "Channel Selector: the input is already mono at this point.");
                    }
                    break;
                case MATRIX_MIXER:
                    if (currentChannels == 1) {
                        add(issues, block, enabled, AudioValidationSeverity.WARNING, null,
                                "Matrix Mixer: the input is already mono at this point.");
                    }
                    break;
                case PHASE_CORRELATION_ANALYZER:
                    if (currentChannels == 1) {
                        add(issues, block, enabled, AudioValidationSeverity.WARNING, null,
                                "Phase and Correlation Analyzer: needs at least two channels; the signal is "
                                        + "mono at this point.");
                    }
                    break;
                case CHANNEL_DELAY_ALIGNMENT:
                    positiveError(issues, block, enabled, block.getType().getDisplayName(),
                            "maxCorrectionSamples", "Maximum correction");
                    if (currentChannels == 1) {
                        add(issues, block, enabled, AudioValidationSeverity.WARNING, null,
                                "Channel Delay Alignment: needs at least two channels; the signal is mono here.");
                    }
                    break;
                case BEST_CHANNEL_SELECTOR:
                    if (currentChannels == 1) {
                        add(issues, block, enabled, AudioValidationSeverity.WARNING, null,
                                "Best Channel Selector: the input is already mono at this point.");
                    }
                    break;
                case MID_SIDE_PROCESSOR:
                case STEREO_WIDTH_CONTROL:
                    if (currentChannels != 0 && currentChannels != 2) {
                        add(issues, block, enabled, AudioValidationSeverity.WARNING, null,
                                block.getType().getDisplayName() + ": works on stereo only; the signal has "
                                        + currentChannels + " channel(s) here.");
                    }
                    break;
                case CENTER_SPEECH_EXTRACTOR:
                    if (currentChannels != 0 && currentChannels != 2) {
                        add(issues, block, enabled, AudioValidationSeverity.WARNING, null,
                                "Center Speech Extractor: works on stereo only; the signal has "
                                        + currentChannels + " channel(s) here.");
                    }
                    add(issues, block, enabled, AudioValidationSeverity.WARNING, null,
                            "Center Speech Extractor: only for suitable stereo sources — it can damage "
                                    + "off-centre speech and is not a replacement for voice isolation.");
                    break;
                case DELAY_AND_SUM_BEAMFORMER:
                    validateBeamformer(issues, block, enabled, currentChannels);
                    break;
                case DIRECTION_OF_ARRIVAL_ANALYZER:
                    validateDirectionOfArrival(issues, block, enabled, currentChannels);
                    break;
                case SPEECH_ENHANCER:
                    validateSpeechEnhancer(issues, block, enabled, inputFormat);
                    break;
                case VOICE_ISOLATION:
                    validateVoiceIsolation(issues, block, enabled, currentChannels);
                    break;
                case ADAPTIVE_NOISE_SUPPRESSION:
                    validateNoiseSuppression(issues, block, enabled, sawEnabledNoiseProfiler);
                    break;
                case SPEECH_LEVELER:
                    validateSpeechLeveler(issues, block, enabled, sawEnabledVad);
                    break;
                case FINAL_LOUDNESS_NORMALIZER:
                    loudnessNormalizerCount++;
                    validateFinalLoudnessNormalizer(issues, block, enabled, blocks, i, loudnessNormalizerCount);
                    break;
                default:
                    break;
            }
            // After validating this block, note whether it shifts the time base for anything downstream.
            if (enabled && sawEnabledVad && isTimeBaseChanger(type)) {
                timeBaseChangedAfterVad = true;
            }
            // Track the channel count so downstream stereo/multichannel checks are accurate.
            if (enabled && reducesToMono(type)) {
                currentChannels = 1;
            }
        }
        return new AudioProfileValidationResult(issues);
    }

    // ------------------------------------------------------------------ generic

    /** Flag any numeric parameter whose stored value cannot be parsed (descriptor-driven, no per-type switch). */
    private void validateParseable(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                   boolean enabled) {
        AudioBlockDescriptor descriptor = registry.descriptor(block.getType());
        for (AudioParameterDescriptor parameter : descriptor.getParameters()) {
            if (parameter.getType() != AudioParameterType.INTEGER
                    && parameter.getType() != AudioParameterType.DECIMAL) {
                continue;
            }
            String raw = block.getParameters().get(parameter.getKey());
            if (raw == null || raw.trim().length() == 0) {
                continue;
            }
            if (!parses(raw.trim(), parameter.getType())) {
                add(issues, block, enabled, AudioValidationSeverity.ERROR, parameter.getKey(),
                        block.getType().getDisplayName() + ": \"" + raw.trim() + "\" is not a valid number for "
                                + parameter.getLabel() + ".");
            }
        }
    }

    private static boolean parses(String value, AudioParameterType type) {
        try {
            if (type == AudioParameterType.INTEGER) {
                Integer.parseInt(value);
            } else {
                Double.parseDouble(value);
            }
            return true;
        } catch (NumberFormatException ex) {
            return false;
        }
    }

    // ------------------------------------------------------------------ per-type checks

    private void validateBand(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block, int rate,
                              boolean enabled) {
        double center = block.getDoubleParameter("centerHz", 0.0d);
        double width = block.getDoubleParameter("widthHz", 0.0d);
        String name = block.getType().getDisplayName();
        if (center <= 0.0d || width <= 0.0d || center - width / 2.0d <= 0.0d) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, "centerHz",
                    name + ": the band (center " + center + " Hz, width " + width
                            + " Hz) must stay above 0 Hz.");
            return;
        }
        if (rate > 0 && center + width / 2.0d >= rate / 2.0d) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, "centerHz",
                    name + ": the band reaches beyond the Nyquist frequency (" + (rate / 2)
                            + " Hz) at this point and will be limited.");
        }
    }

    private void validateCutoff(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block, int rate,
                                boolean enabled) {
        double cutoff = block.getDoubleParameter("cutoffHz", 0.0d);
        String name = block.getType().getDisplayName();
        if (cutoff <= 0.0d) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, "cutoffHz",
                    name + ": the cutoff frequency must be greater than 0 Hz.");
            return;
        }
        if (rate > 0 && cutoff >= rate / 2.0d) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, "cutoffHz",
                    name + ": the cutoff (" + cutoff + " Hz) is at or above the Nyquist frequency ("
                            + (rate / 2) + " Hz) here.");
        }
    }

    private void validateGain(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                              boolean enabled) {
        double gainDb = block.getDoubleParameter("gainDb", 0.0d);
        if (!isFinite(gainDb)) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, "gainDb",
                    "Gain: the value must be a finite dB number.");
            return;
        }
        if (Math.abs(gainDb) > 48.0d) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, "gainDb",
                    "Gain: " + gainDb + " dB is extreme and will heavily clip or mute the signal.");
        }
    }

    private void validateEqBand(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                String frequencyKey, int rate, boolean peaking, boolean enabled) {
        double frequency = block.getDoubleParameter(frequencyKey, 0.0d);
        double gainDb = block.getDoubleParameter("gainDb", 0.0d);
        String name = block.getType().getDisplayName();
        String nyquist = rate > 0 ? " and below " + (rate / 2) + " Hz" : "";
        if (!isFinite(frequency) || frequency <= 0.0d) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, frequencyKey,
                    name + ": the frequency must be greater than 0 Hz" + nyquist + ".");
            return;
        }
        if (rate > 0 && frequency >= rate / 2.0d) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, frequencyKey,
                    name + ": the frequency (" + frequency + " Hz) is at or above the Nyquist frequency ("
                            + (rate / 2) + " Hz) here and the band will be bypassed.");
        }
        if (peaking) {
            double q = block.getDoubleParameter("q", 0.0d);
            if (!isFinite(q) || q <= 0.0d) {
                add(issues, block, enabled, AudioValidationSeverity.ERROR, "q",
                        name + ": the Q factor must be a finite value above 0.");
            }
        } else {
            double slope = block.getDoubleParameter("slope", 0.0d);
            if (!isFinite(slope) || slope <= 0.0d) {
                add(issues, block, enabled, AudioValidationSeverity.ERROR, "slope",
                        name + ": the shelf slope must be a finite value above 0.");
            }
        }
        if (!isFinite(gainDb)) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, "gainDb",
                    name + ": the gain must be a finite dB value.");
        } else if (Math.abs(gainDb) > 36.0d) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, "gainDb",
                    name + ": " + gainDb + " dB is an extreme equalizer gain.");
        }
    }

    private void validateEqualizer(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                   int rate, boolean enabled) {
        String name = block.getType().getDisplayName();
        frequencyChecks(issues, block, enabled, name, "lowShelfHz", rate);
        frequencyChecks(issues, block, enabled, name, "midHz", rate);
        frequencyChecks(issues, block, enabled, name, "highShelfHz", rate);
        finiteError(issues, block, enabled, name, "lowShelfGainDb", "Low shelf gain");
        finiteError(issues, block, enabled, name, "midGainDb", "Mid gain");
        finiteError(issues, block, enabled, name, "highShelfGainDb", "High shelf gain");
        double midQ = block.getDoubleParameter("midQ", 1.0d);
        if (!isFinite(midQ) || midQ <= 0.0d) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, "midQ",
                    name + ": the mid Q must be a finite value above 0.");
        }
        finiteError(issues, block, enabled, name, "gainDb", "Master gain");
        nonNegativeError(issues, block, enabled, name, "loudnessDriveDb", "Loudness drive");
        finiteError(issues, block, enabled, name, "peakCeilingDb", "Peak ceiling");
        double master = block.getDoubleParameter("gainDb", 0.0d);
        if (isFinite(master) && Math.abs(master) > 36.0d) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, "gainDb",
                    name + ": " + master + " dB is an extreme master gain.");
        }
    }

    private void validateVoiceActivity(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                       boolean enabled) {
        String name = block.getType().getDisplayName();
        rangeError(issues, block, enabled, name, "sensitivity", "Sensitivity", 0.0d, 1.0d);
        rangeError(issues, block, enabled, name, "minSpeechProbability",
                "Minimum speech probability", 0.0d, 1.0d);
        double frame = block.getDoubleParameter("frameDurationMs", 20.0d);
        if (!isFinite(frame) || frame <= 0.0d) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, "frameDurationMs",
                    name + ": the frame duration must be a finite value above 0 ms.");
        }
        nonNegativeError(issues, block, enabled, name, "attackMs", "Attack");
        nonNegativeError(issues, block, enabled, name, "releaseMs", "Release");
        nonNegativeError(issues, block, enabled, name, "hangoverMs", "Hangover");
        nonNegativeError(issues, block, enabled, name, "minSpeechMs", "Minimum speech duration");
        nonNegativeError(issues, block, enabled, name, "minSilenceMs", "Minimum silence duration");
        double adapt = block.getDoubleParameter("noiseAdaptationSpeed", 0.05d);
        if (!isFinite(adapt) || adapt <= 0.0d || adapt > 0.5d) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, "noiseAdaptationSpeed",
                    name + ": the noise adaptation speed must be a finite value in (0, 0.5].");
        }
    }

    private void validateSpeechGate(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                    boolean enabled, boolean sawEnabledVad, boolean timeBaseChangedAfterVad) {
        String name = block.getType().getDisplayName();
        if (!sawEnabledVad) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, null,
                    name + ": requires an enabled Voice Activity Detection block before it.");
        } else if (timeBaseChangedAfterVad) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, null,
                    name + ": a block that changes the time base (e.g. Resampler or Silence Trimmer) must not "
                            + "sit between Voice Activity Detection and Speech Gate.");
        }
        rangeError(issues, block, enabled, name, "minSpeechProbability", "Minimum speech probability", 0.0d, 1.0d);
        nonNegativeError(issues, block, enabled, name, "preRollMs", "Pre-roll");
        nonNegativeError(issues, block, enabled, name, "postRollMs", "Post-roll");
        nonNegativeError(issues, block, enabled, name, "attackMs", "Open fade");
        nonNegativeError(issues, block, enabled, name, "releaseMs", "Close fade");
    }

    private void validateExpander(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                  boolean enabled, boolean sawEnabledVad) {
        String name = block.getType().getDisplayName();
        double ratio = block.getDoubleParameter("ratio", 2.0d);
        if (!isFinite(ratio) || ratio < 1.0d) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, "ratio",
                    name + ": the ratio must be a finite value of 1.0 or greater.");
        }
        double maxAtt = block.getDoubleParameter("maxAttenuationDb", 0.0d);
        if (!isFinite(maxAtt) || maxAtt < 0.0d) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, "maxAttenuationDb",
                    name + ": the maximum attenuation must be a finite value of 0 dB or more.");
        }
        finiteError(issues, block, enabled, name, "thresholdDb", "Threshold");
        nonNegativeError(issues, block, enabled, name, "attackMs", "Attack");
        nonNegativeError(issues, block, enabled, name, "releaseMs", "Release");
        nonNegativeError(issues, block, enabled, name, "holdMs", "Hold");
        nonNegativeError(issues, block, enabled, name, "kneeDb", "Knee");
        rangeError(issues, block, enabled, name, "minSpeechProbability", "Minimum speech probability",
                0.0d, 1.0d);
        if (block.getBooleanParameter("speechProtection", false) && !sawEnabledVad) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, "speechProtection",
                    name + ": speech protection needs a Voice Activity Detection block upstream; "
                            + "it runs level-based only until then.");
        }
    }

    private void validateSilenceTrimmer(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                        boolean enabled, boolean sawEnabledVad,
                                        boolean timeBaseChangedAfterVad, int occurrence) {
        String name = block.getType().getDisplayName();
        if (!sawEnabledVad) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, null,
                    name + ": requires a Voice Activity Detection block before it.");
        } else if (timeBaseChangedAfterVad) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, null,
                    name + ": a block that changes the time base (e.g. Resampler) must not sit between "
                            + "Voice Activity Detection and Silence Trimmer.");
        }
        if (occurrence > 1) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, null,
                    name + ": a second silence trimmer has no additional defined effect.");
        }
        nonNegativeError(issues, block, enabled, name, "preRollMs", "Pre-roll");
        nonNegativeError(issues, block, enabled, name, "postRollMs", "Post-roll");
        nonNegativeError(issues, block, enabled, name, "minRetainedMs", "Minimum retained duration");
        nonNegativeError(issues, block, enabled, name, "zeroCrossingSearchMs", "Zero-crossing search window");
        rangeError(issues, block, enabled, name, "minSpeechProbability", "Minimum speech probability",
                0.0d, 1.0d);
    }

    private void validateDeEsser(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                 int rate, boolean enabled) {
        String name = block.getType().getDisplayName();
        frequencyChecks(issues, block, enabled, name, "targetFrequencyHz", rate);
        positiveError(issues, block, enabled, name, "bandwidthHz", "Bandwidth");
        finiteError(issues, block, enabled, name, "thresholdDb", "Threshold");
        nonNegativeError(issues, block, enabled, name, "reductionDb", "Reduction");
        nonNegativeError(issues, block, enabled, name, "attackMs", "Attack");
        nonNegativeError(issues, block, enabled, name, "releaseMs", "Release");
    }

    private void validateAdaptiveHumRemoval(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                            int rate, boolean enabled, boolean sawEnabledVad) {
        String name = block.getType().getDisplayName();
        frequencyChecks(issues, block, enabled, name, "baseFrequencyHz", rate);
        nonNegativeError(issues, block, enabled, name, "searchRangeHz", "Search range");
        double speed = block.getDoubleParameter("adaptationSpeed", 0.1d);
        if (!isFinite(speed) || speed < 0.0d || speed > 1.0d) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, "adaptationSpeed",
                    name + ": the adaptation speed must be a finite value between 0 and 1.");
        }
        double harmonics = block.getDoubleParameter("harmonics", 3.0d);
        if (!isFinite(harmonics) || harmonics < 1.0d) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, "harmonics",
                    name + ": the number of harmonics must be at least 1.");
        }
        nonNegativeError(issues, block, enabled, name, "maxAttenuationDb", "Maximum attenuation");
        if (block.getBooleanParameter("speechProtection", false) && !sawEnabledVad) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, "speechProtection",
                    name + ": speech protection needs a Voice Activity Detection block upstream.");
        }
    }

    private void validatePlosiveReduction(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                          int rate, boolean enabled) {
        String name = block.getType().getDisplayName();
        rangeError(issues, block, enabled, name, "strength", "Strength", 0.0d, 1.0d);
        frequencyChecks(issues, block, enabled, name, "targetFrequencyHz", rate);
        nonNegativeError(issues, block, enabled, name, "attackMs", "Attack");
        nonNegativeError(issues, block, enabled, name, "releaseMs", "Release");
    }

    private void validateBreathReduction(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                         boolean enabled, boolean sawEnabledVad) {
        String name = block.getType().getDisplayName();
        rangeError(issues, block, enabled, name, "sensitivity", "Sensitivity", 0.0d, 1.0d);
        nonNegativeError(issues, block, enabled, name, "maxAttenuationDb", "Maximum attenuation");
        nonNegativeError(issues, block, enabled, name, "attackMs", "Attack");
        nonNegativeError(issues, block, enabled, name, "releaseMs", "Release");
        if (block.getBooleanParameter("speechProtection", true) && !sawEnabledVad) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, "speechProtection",
                    name + ": speech protection needs a Voice Activity Detection block upstream.");
        }
    }

    private void validateNoiseProfiler(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                       boolean enabled) {
        String name = block.getType().getDisplayName();
        nonNegativeError(issues, block, enabled, name, "learnTimeMs", "Learn time");
        rangeError(issues, block, enabled, name, "minConfidence", "Minimum confidence", 0.0d, 1.0d);
    }

    private void validateBeamformer(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                    boolean enabled, int currentChannels) {
        String name = block.getType().getDisplayName();
        com.aresstack.audio.dsp.MicrophoneArrayProfile array =
                com.aresstack.audio.dsp.MicrophoneArrayProfile.parse(block.getId(), "inline",
                        block.getParameter("micPositionsMm", ""));
        if (array == null) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, "micPositionsMm",
                    name + ": needs a valid microphone-array geometry (at least two \"x,y,z\" positions in "
                            + "millimetres); it will not run with invented positions.");
        } else if (currentChannels != 0 && array.getMicrophoneCount() != currentChannels) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, "micPositionsMm",
                    name + ": the geometry has " + array.getMicrophoneCount() + " microphones but the signal "
                            + "has " + currentChannels + " channel(s) here.");
        }
        if (currentChannels != 0 && currentChannels < 2) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, null,
                    name + ": needs at least two synchronized channels; the signal is mono at this point.");
        }
        positiveError(issues, block, enabled, name, "speedOfSoundMmPerS", "Speed of sound");
    }

    private void validateVoiceIsolation(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                        boolean enabled, int currentChannels) {
        String name = block.getType().getDisplayName();
        rangeError(issues, block, enabled, name, "strength", "Strength", 0.0d, 1.0d);
        String backend = block.getParameter("backend", "PURE_JAVA_CENTER");
        if ("NEURAL".equals(backend)) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, "backend",
                    name + ": the neural backend is not installed; the block passes the audio through "
                            + "unchanged until it is available.");
        } else if (currentChannels != 0 && currentChannels != 2) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, null,
                    name + ": the pure-Java center backend needs a stereo source; the signal has "
                            + currentChannels + " channel(s) here.");
        }
        add(issues, block, enabled, AudioValidationSeverity.WARNING, null,
                name + ": voice isolation is distinct from noise suppression and only reduces other "
                        + "voices/music approximately.");
    }

    private void validateSpeechEnhancer(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                        boolean enabled, PcmAudioFormat inputFormat) {
        String name = block.getType().getDisplayName();
        rangeError(issues, block, enabled, name, "strength", "Strength", 0.0d, 1.0d);
        rangeError(issues, block, enabled, name, "artifactProtection", "Artifact protection", 0.0d, 1.0d);
        String backendId = block.getParameter("backend", "PURE_JAVA_DSP");
        com.aresstack.audio.enhance.BackendAvailability availability =
                com.aresstack.audio.enhance.SpeechEnhancementBackends.availability(backendId, inputFormat);
        if (!availability.isRunnable()) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, "backend",
                    name + ": the \"" + backendId + "\" backend is " + availability.getDisplayName().toLowerCase()
                            + "; the block passes the audio through unchanged until it is available.");
        }
    }

    private void validateDirectionOfArrival(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                            boolean enabled, int currentChannels) {
        String name = block.getType().getDisplayName();
        com.aresstack.audio.dsp.MicrophoneArrayProfile array =
                com.aresstack.audio.dsp.MicrophoneArrayProfile.parse(block.getId(), "inline",
                        block.getParameter("micPositionsMm", ""));
        if (array == null) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, "micPositionsMm",
                    name + ": needs a valid microphone-array geometry to estimate a direction.");
        }
        if (currentChannels != 0 && currentChannels < 2) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, null,
                    name + ": needs at least two synchronized channels; the signal is mono at this point.");
        }
        positiveError(issues, block, enabled, name, "speedOfSoundMmPerS", "Speed of sound");
    }

    private void validateDereverberation(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                         boolean enabled) {
        String name = block.getType().getDisplayName();
        rangeError(issues, block, enabled, name, "strength", "Strength", 0.0d, 1.0d);
        positiveError(issues, block, enabled, name, "predictionDelay", "Prediction delay");
        positiveError(issues, block, enabled, name, "filterLength", "Filter length");
        positiveError(issues, block, enabled, name, "iterations", "Iterations");
        rangeError(issues, block, enabled, name, "earlyReflectionPreservation",
                "Early reflection preservation", 0.0d, 1.0d);
        rangeError(issues, block, enabled, name, "artifactProtection", "Artifact protection", 0.0d, 1.0d);
        if ("BLOCK_ADAPTIVE".equals(block.getParameter("mode", "OFFLINE"))
                || "STREAMING".equals(block.getParameter("mode", "OFFLINE"))) {
            rangeError(issues, block, enabled, name, "adaptationSpeed", "Adaptation speed", 0.0d, 1.0d);
            positiveError(issues, block, enabled, name, "blockSizeFrames", "Block size");
        }
    }

    private void validateRoomReverbAnalyzer(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                            boolean enabled) {
        String name = block.getType().getDisplayName();
        positiveError(issues, block, enabled, name, "frameDurationMs", "Frame duration");
        positiveError(issues, block, enabled, name, "minDecayDb", "Minimum decay");
        positiveError(issues, block, enabled, name, "maxReverbSeconds", "Maximum reverb time");
    }

    private void validateNoiseSuppression(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                          boolean enabled, boolean sawEnabledNoiseProfiler) {
        String name = block.getType().getDisplayName();
        nonNegativeError(issues, block, enabled, name, "maxAttenuationDb", "Maximum attenuation");
        rangeError(issues, block, enabled, name, "adaptationSpeed", "Adaptation speed", 0.0d, 1.0d);
        finiteError(issues, block, enabled, name, "noiseFloorDb", "Noise floor");
        rangeError(issues, block, enabled, name, "minSpeechProbability", "Minimum speech probability", 0.0d, 1.0d);
        rangeError(issues, block, enabled, name, "artifactProtection", "Artifact protection", 0.0d, 1.0d);
        nonNegativeError(issues, block, enabled, name, "attackMs", "Attack");
        nonNegativeError(issues, block, enabled, name, "releaseMs", "Release");
        if ("USE_FIXED_PROFILE".equals(block.getParameter("mode", "AUTOMATIC")) && !sawEnabledNoiseProfiler) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, "mode",
                    name + ": \"Use a learned noise profile\" needs a Noise Profiler block upstream; "
                            + "it falls back to tracking the noise floor until then.");
        }
        // Speech protection is optional here: Adaptive Noise Suppression works fully without an upstream
        // Voice Activity Detection block (it just cannot ease suppression during detected speech). Do not
        // warn about a missing VAD, so the block is not needlessly reordered and quality is not hurt.
    }

    private void validateFinalLoudnessNormalizer(List<AudioProfileValidationIssue> issues,
                                                 AudioBlockDefinition block, boolean enabled,
                                                 List<AudioBlockDefinition> blocks, int index, int occurrence) {
        String name = block.getType().getDisplayName();
        finiteError(issues, block, enabled, name, "targetLevelDb", "Target level");
        nonNegativeError(issues, block, enabled, name, "maxTotalGainDb", "Maximum total gain");
        nonNegativeError(issues, block, enabled, name, "maxTotalAttenuationDb", "Maximum total attenuation");
        finiteError(issues, block, enabled, name, "peakCeilingDb", "Peak ceiling");
        if (occurrence > 1) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, null,
                    name + ": a second final loudness normalizer causes an uncontrolled double normalization.");
        }
        for (int j = index + 1; j < blocks.size(); j++) {
            AudioBlockDefinition later = blocks.get(j);
            if (later.isEnabled() && registry.descriptor(later.getType()).getCapabilities().modifiesAudio()) {
                add(issues, block, enabled, AudioValidationSeverity.WARNING, null,
                        name + ": it should be the last level-changing stage, but "
                                + later.getType().getDisplayName() + " follows and can change the level again.");
                break;
            }
        }
    }

    private void validateSpeechLeveler(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                       boolean enabled, boolean sawEnabledVad) {
        String name = block.getType().getDisplayName();
        finiteError(issues, block, enabled, name, "targetSpeechLevelDb", "Target speech level");
        nonNegativeError(issues, block, enabled, name, "maxGainDb", "Maximum gain");
        nonNegativeError(issues, block, enabled, name, "maxAttenuationDb", "Maximum attenuation");
        positiveError(issues, block, enabled, name, "attackMs", "Attack");
        positiveError(issues, block, enabled, name, "releaseMs", "Release");
        nonNegativeError(issues, block, enabled, name, "holdMs", "Hold");
        positiveError(issues, block, enabled, name, "maxGainChangePerSecond", "Maximum gain change per second");
        rangeError(issues, block, enabled, name, "minSpeechProbability", "Minimum speech probability", 0.0d, 1.0d);
        nonNegativeError(issues, block, enabled, name, "silenceGainLimitDb", "Silence gain limit");
        double maxGain = block.getDoubleParameter("maxGainDb", 0.0d);
        if (isFinite(maxGain) && maxGain > 30.0d) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, "maxGainDb",
                    name + ": a maximum gain above 30 dB will strongly amplify noise in quiet passages.");
        }
        if (!sawEnabledVad) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, null,
                    name + ": add a Voice Activity Detection block before it; without a speech probability "
                            + "it levels purely by signal energy and may lift background noise.");
        }
    }

    private void frequencyChecks(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                 boolean enabled, String name, String key, int rate) {
        double frequency = block.getDoubleParameter(key, 0.0d);
        if (!isFinite(frequency) || frequency <= 0.0d) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, key,
                    name + ": the frequency must be a finite value above 0 Hz.");
        } else if (rate > 0 && frequency >= rate / 2.0d) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, key,
                    name + ": the frequency (" + frequency + " Hz) is at or above the Nyquist frequency ("
                            + (rate / 2) + " Hz) here.");
        }
    }

    private void positiveError(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                               boolean enabled, String name, String key, String label) {
        double value = block.getDoubleParameter(key, 0.0d);
        if (!isFinite(value) || value <= 0.0d) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, key,
                    name + ": " + label + " must be a finite value above 0.");
        }
    }

    private static boolean reducesToMono(AudioBlockType type) {
        return type == AudioBlockType.CHANNEL_MIXER || type == AudioBlockType.CHANNEL_SELECTOR
                || type == AudioBlockType.MATRIX_MIXER || type == AudioBlockType.BEST_CHANNEL_SELECTOR
                || type == AudioBlockType.DELAY_AND_SUM_BEAMFORMER;
    }

    private static boolean isTimeBaseChanger(AudioBlockType type) {
        return type == AudioBlockType.RESAMPLER || type == AudioBlockType.SILENCE_TRIMMER;
    }

    private void finiteError(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                             boolean enabled, String name, String key, String label) {
        double value = block.getDoubleParameter(key, 0.0d);
        if (!isFinite(value)) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, key,
                    name + ": " + label + " must be a finite value.");
        }
    }

    private void validateResampler(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                   boolean sawChannelMixer, boolean enabled) {
        int target = block.getIntParameter("targetRateHz", 0);
        if (target < 4000 || target > 192000) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, "targetRateHz",
                    "Resampler: the target rate " + target
                            + " Hz is outside the supported 4000–192000 Hz range.");
        }
        if (!sawChannelMixer) {
            add(issues, block, enabled, AudioValidationSeverity.WARNING, null,
                    "Resampler: place a channel mixer before the resampler so the input is mono.");
        }
    }

    private void rangeError(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                            boolean enabled, String name, String key, String label, double min, double max) {
        double value = block.getDoubleParameter(key, Double.NaN);
        if (!isFinite(value) || value < min || value > max) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, key,
                    name + ": " + label + " must be a finite value between " + min + " and " + max + ".");
        }
    }

    private void nonNegativeError(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block,
                                  boolean enabled, String name, String key, String label) {
        double value = block.getDoubleParameter(key, 0.0d);
        if (!isFinite(value) || value < 0.0d) {
            add(issues, block, enabled, AudioValidationSeverity.ERROR, key,
                    name + ": " + label + " must be a finite value of 0 ms or more.");
        }
    }

    /** Add an issue, downgrading an error to a warning for a disabled (bypassed) block so it never blocks. */
    private void add(List<AudioProfileValidationIssue> issues, AudioBlockDefinition block, boolean enabled,
                     AudioValidationSeverity wanted, String parameterKey, String message) {
        AudioValidationSeverity severity = !enabled && wanted == AudioValidationSeverity.ERROR
                ? AudioValidationSeverity.WARNING : wanted;
        issues.add(new AudioProfileValidationIssue(severity, block.getId(), block.getType(),
                parameterKey, message));
    }

    private static List<Message> toMessages(AudioProfileValidationResult result) {
        List<Message> messages = new ArrayList<Message>();
        for (AudioProfileValidationIssue issue : result.getIssues()) {
            Severity severity = issue.getSeverity() == AudioValidationSeverity.ERROR
                    ? Severity.ERROR : Severity.WARNING;
            messages.add(new Message(severity, issue.getBlockId(), issue.getMessage()));
        }
        return messages;
    }

    private static boolean isFinite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }
}

package com.aresstack.audio.pipeline;

import com.aresstack.audio.profile.AudioBlockType;

import java.util.Collections;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Map;

/** Provide plain-language help text for every built-in DSP block and editable parameter. */
public final class AudioDspHelp {

    private static final Map<AudioBlockType, String> BLOCK_DESCRIPTIONS;
    private static final Map<String, String> PARAMETER_DESCRIPTIONS;
    private static final Map<String, String> CONTEXT_PARAMETER_DESCRIPTIONS;

    static {
        EnumMap<AudioBlockType, String> blocks = new EnumMap<AudioBlockType, String>(AudioBlockType.class);
        blocks.put(AudioBlockType.PASSTHROUGH,
                "A neutral placeholder that leaves the audio unchanged. Choose a function in the inspector to make this block do something.");
        blocks.put(AudioBlockType.CHANNEL_MIXER,
                "Combines a multi-channel recording into one mono channel so the speech-processing pipeline can continue.");
        blocks.put(AudioBlockType.LOW_PASS,
                "Keeps frequencies below the cutoff and reduces higher frequencies such as hiss or ultrasonic content.");
        blocks.put(AudioBlockType.HIGH_PASS,
                "Keeps frequencies above the cutoff and reduces rumble, handling noise and very low-frequency drift.");
        blocks.put(AudioBlockType.BAND_PASS,
                "Keeps a selected frequency band and reduces frequencies above and below it.");
        blocks.put(AudioBlockType.BAND_STOP,
                "Reduces a selected frequency band while leaving frequencies outside that band mostly unchanged.");
        blocks.put(AudioBlockType.RESAMPLER,
                "Changes the sample rate of mono audio, for example from 48 kHz microphone audio to 16 kHz speech-model input.");
        blocks.put(AudioBlockType.DC_OFFSET_REMOVAL,
                "Removes a constant electrical offset that shifts the waveform away from its zero line.");
        blocks.put(AudioBlockType.NOISE_GATE,
                "Turns quiet passages down when their level falls below a threshold. This is a hard gate, unlike the smoother Expander.");
        blocks.put(AudioBlockType.COMPRESSOR,
                "Reduces the level difference between loud and quiet passages by turning down signal above a threshold.");
        blocks.put(AudioBlockType.LIMITER,
                "Prevents peaks from exceeding a fixed ceiling and protects the following output stage from clipping.");
        blocks.put(AudioBlockType.GAIN,
                "Raises or lowers the entire signal by a fixed number of decibels.");
        blocks.put(AudioBlockType.PARAMETRIC_EQ,
                "Boosts or cuts a controllable frequency region around a center frequency.");
        blocks.put(AudioBlockType.LOW_SHELF,
                "Boosts or cuts low frequencies below the selected cutoff with a smooth transition.");
        blocks.put(AudioBlockType.HIGH_SHELF,
                "Boosts or cuts high frequencies above the selected cutoff with a smooth transition.");
        blocks.put(AudioBlockType.EQUALIZER,
                "A channel-strip equalizer: a low shelf, a parametric mid band and a high shelf for tone, plus a master gain and an optional loudness stage. Loudness folds the make-up gain into a soft saturation curve so the signal sounds louder and fuller while peaks are limited toward the ceiling instead of clipping.");
        blocks.put(AudioBlockType.VOICE_ACTIVITY_DETECTION,
                "Analyzes the recording and marks which time frames probably contain speech. Later blocks can use this metadata to protect speech or trim silence.");
        blocks.put(AudioBlockType.SPEECH_GATE,
                "Uses upstream Voice Activity Detection to keep speech and turn every non-speech region into exact digital silence. Pre-/post-roll protect word boundaries and short fades avoid clicks; the recording length is not changed.");
        blocks.put(AudioBlockType.EXPANDER,
                "Smoothly turns down low-level sound below a threshold while leaving louder speech unchanged. It is gentler than a Noise Gate.");
        blocks.put(AudioBlockType.SILENCE_TRIMMER,
                "Removes silence only from the beginning and/or end of a recording using upstream Voice Activity Detection metadata. Internal pauses remain intact.");
        blocks.put(AudioBlockType.DE_ESSER,
                "Detects harsh sibilants such as S and SH sounds and dynamically reduces their frequency band using time-domain filters.");
        blocks.put(AudioBlockType.ADAPTIVE_HUM_REMOVAL,
                "Tracks mains hum near 50 or 60 Hz and its harmonics, then reduces the detected tones with adaptive time-domain filters.");
        blocks.put(AudioBlockType.PLOSIVE_REDUCTION,
                "Detects short low-frequency bursts from P and B sounds and temporarily turns down only the affected low band.");
        blocks.put(AudioBlockType.BREATH_REDUCTION,
                "Uses upstream speech detection to turn down audible breathing and other low-level non-speech passages while protecting speech.");
        blocks.put(AudioBlockType.DE_ESSER_FFT,
                "Detects harsh sibilants in the frequency spectrum and reduces only the affected FFT bins while preserving phase.");
        blocks.put(AudioBlockType.ADAPTIVE_HUM_REMOVAL_FFT,
                "Finds drifting mains hum in the frequency spectrum and reduces the fundamental tone and selected harmonics.");
        blocks.put(AudioBlockType.PLOSIVE_REDUCTION_FFT,
                "Detects sudden low-frequency spectral energy from plosives and temporarily reduces the affected FFT bins.");
        blocks.put(AudioBlockType.BREATH_REDUCTION_FFT,
                "Detects noise-like breath spectra during non-speech frames and reduces them while protecting detected speech.");
        blocks.put(AudioBlockType.NOISE_PROFILER,
                "Analyzes background noise and creates a spectral Noise Profile for a following Adaptive Noise Suppression block. It does not change the audio itself.");
        blocks.put(AudioBlockType.ADAPTIVE_NOISE_SUPPRESSION,
                "Reduces stationary or slowly changing background noise by estimating or consuming a Noise Profile and applying frequency-dependent attenuation.");
        blocks.put(AudioBlockType.SPEECH_LEVELER,
                "Automatically evens out speech recorded at changing distances while limiting noise amplification and clipping.");
        blocks.put(AudioBlockType.FINAL_LOUDNESS_NORMALIZER,
                "Applies one final gain to the completed recording so its overall RMS or peak level reaches a defined target.");
        blocks.put(AudioBlockType.ROOM_REVERB_ANALYZER,
                "Estimates room decay and reverberation characteristics and publishes a Room Profile for later processing. It does not change the audio itself.");
        blocks.put(AudioBlockType.DEREVERBERATION,
                "Reduces late room reverberation with WPE processing. Offline mode uses the complete recording; adaptive modes use limited history.");
        blocks.put(AudioBlockType.CHANNEL_SELECTOR,
                "Selects one input channel and outputs it as mono, with an optional fallback when the requested channel does not exist.");
        blocks.put(AudioBlockType.MATRIX_MIXER,
                "Combines all input channels into mono using configurable per-channel weights.");
        blocks.put(AudioBlockType.CHANNEL_GAIN_POLARITY,
                "Applies an independent gain and optional polarity inversion to each channel without changing the channel count.");
        blocks.put(AudioBlockType.PHASE_CORRELATION_ANALYZER,
                "Measures inter-channel phase and correlation and publishes diagnostics without changing the audio.");
        blocks.put(AudioBlockType.CHANNEL_DELAY_ALIGNMENT,
                "Aligns channels in time using measured or manually entered delays before mixing or beamforming.");
        blocks.put(AudioBlockType.BEST_CHANNEL_SELECTOR,
                "Evaluates channel quality and outputs the best channel as mono, optionally preferring a specific channel.");
        blocks.put(AudioBlockType.CHANNEL_HEALTH_ANALYZER,
                "Checks channels for silence, clipping, dropouts, DC offset, noise and polarity problems without changing the audio.");
        blocks.put(AudioBlockType.MID_SIDE_PROCESSOR,
                "Separates stereo into center (Mid) and difference (Side) content so each part can be adjusted independently.");
        blocks.put(AudioBlockType.CENTER_SPEECH_EXTRACTOR,
                "Emphasizes speech mixed near the stereo center and reduces content that is stronger on the sides.");
        blocks.put(AudioBlockType.STEREO_WIDTH_CONTROL,
                "Narrows or widens a stereo signal by changing its side component while retaining the center.");
        blocks.put(AudioBlockType.DELAY_AND_SUM_BEAMFORMER,
                "Uses a known microphone-array geometry to align channels for a target direction and sum them into one focused mono signal.");
        blocks.put(AudioBlockType.DIRECTION_OF_ARRIVAL_ANALYZER,
                "Estimates the direction of a sound source from timing differences between synchronized microphone channels.");
        blocks.put(AudioBlockType.SPEECH_ENHANCER,
                "Runs the selected speech-enhancement backend. Available backends may combine denoising, dereverberation or optional model-based processing.");
        blocks.put(AudioBlockType.VOICE_ISOLATION,
                "Emphasizes a dominant voice and reduces competing content. The Pure Java backend is a stereo-center approximation; true neural separation requires an optional backend.");
        BLOCK_DESCRIPTIONS = Collections.unmodifiableMap(blocks);

        Map<String, String> parameters = new HashMap<String, String>();
        parameters.put("channels", "Sets the number of output channels. The current Channel Mixer supports mono output only.");
        parameters.put("implementation", "Selects the filter algorithm. Different designs trade CPU cost, transition sharpness and phase behavior.");
        parameters.put("cutoffHz", "Sets the boundary frequency where this filter begins its transition.");
        parameters.put("order", "Sets the filter order. Higher orders make a steeper transition but can ring more and use more CPU.");
        parameters.put("centerHz", "Sets the center frequency of the affected band.");
        parameters.put("widthHz", "Sets the total width of the affected frequency band.");
        parameters.put("targetRateHz", "Sets the output sample rate in samples per second.");
        parameters.put("quality", "Selects the resampling quality and CPU-cost trade-off. HIGH uses more processing than FAST.");
        parameters.put("hiddenAntiAliasing", "Adds an anti-alias low-pass stage inside the resampler. Enable it when lowering the sample rate if no suitable low-pass block precedes this block.");
        parameters.put("threshold", "Sets the signal level where this dynamics block starts changing its gain.");
        parameters.put("closedGain", "Sets the remaining gain while the Noise Gate is closed. Zero is silence; one leaves the signal unchanged.");
        parameters.put("attackMillis", "Sets how quickly the block reacts when the signal crosses into attenuation or compression.");
        parameters.put("releaseMillis", "Sets how quickly the gain returns after the triggering signal has ended.");
        parameters.put("ratio", "Sets the strength of the level change. A larger ratio produces stronger compression or expansion.");
        parameters.put("ceiling", "Sets the highest allowed PCM sample value. Peaks above it are limited.");
        parameters.put("gainDb", "Sets a fixed gain in decibels. Positive values boost; negative values attenuate.");
        parameters.put("q", "Sets the bandwidth around the EQ center frequency. Higher Q values affect a narrower band.");
        parameters.put("slope", "Sets how quickly the shelf reaches its full boost or cut around the cutoff.");
        parameters.put("sensitivity", "Sets how readily the detector reacts. Higher values classify weaker evidence as a match.");
        parameters.put("minSpeechProbability", "Sets the minimum upstream speech probability required to treat a frame as speech.");
        parameters.put("frameDurationMs", "Sets the analysis-frame duration. Shorter frames react faster; longer frames provide steadier measurements.");
        parameters.put("attackMs", "Sets how quickly the effect engages after its detector triggers.");
        parameters.put("releaseMs", "Sets how quickly the effect relaxes after the detected event ends.");
        parameters.put("hangoverMs", "Keeps speech active for this long after the detector falls below its threshold, preventing clipped word endings.");
        parameters.put("minSpeechMs", "Requires speech to remain detected for this duration before a speech segment is accepted.");
        parameters.put("minSilenceMs", "Requires silence to remain detected for this duration before a silence segment is accepted.");
        parameters.put("noiseAdaptationSpeed", "Sets how quickly Voice Activity Detection updates its background-noise estimate.");
        parameters.put("adaptNoiseDuringSpeech", "Allows the background-noise estimate to update during detected speech. Disable it to reduce the risk of learning speech as noise.");
        parameters.put("thresholdDb", "Sets the detector threshold in dBFS. More negative values trigger on quieter sound.");
        parameters.put("kneeDb", "Sets how gradually the gain curve transitions around the threshold. A larger knee sounds softer.");
        parameters.put("holdMs", "Keeps the current action active for this minimum time before release begins.");
        parameters.put("maxAttenuationDb", "Limits how many decibels the block may remove. Larger values allow stronger reduction.");
        parameters.put("detectorWindowMs", "Sets the time window used to estimate the signal level that drives the Expander.");
        parameters.put("speechProtection", "Uses upstream speech metadata, when available, to reduce or suspend processing during detected speech.");
        parameters.put("trimLeading", "Removes detected silence before the first speech segment.");
        parameters.put("trimTrailing", "Removes detected silence after the last speech segment.");
        parameters.put("preRollMs", "Keeps this much audio before the first detected speech frame so word beginnings are not cut off.");
        parameters.put("postRollMs", "Keeps this much audio after the last detected speech frame so word endings are not cut off.");
        parameters.put("minRetainedMs", "Prevents trimming from reducing the output below this minimum duration.");
        parameters.put("noSpeechBehavior", "Selects what happens when no speech is detected: keep the original recording or fail processing.");
        parameters.put("zeroCrossingAlignment", "Moves trim points toward a nearby waveform zero crossing to reduce clicks.");
        parameters.put("zeroCrossingSearchMs", "Sets how far the trimmer may search around each cut for a zero crossing.");
        parameters.put("targetFrequencyHz", "Sets the main frequency around which the detector and reduction operate.");
        parameters.put("bandwidthHz", "Sets the width of the detected and attenuated frequency region.");
        parameters.put("reductionDb", "Sets the maximum gain reduction applied to the detected frequency band.");
        parameters.put("baseFrequencyHz", "Sets the expected mains-hum fundamental, normally 50 Hz or 60 Hz.");
        parameters.put("searchRangeHz", "Sets how far around the expected base frequency the block searches for drifting hum.");
        parameters.put("adaptationSpeed", "Sets how quickly an adaptive estimate follows changing noise, hum, direction or room conditions.");
        parameters.put("harmonics", "Sets how many multiples of the detected hum fundamental are also reduced.");
        parameters.put("strength", "Sets the overall effect amount. Higher values process more aggressively and may introduce more artifacts.");
        parameters.put("learnTimeMs", "Limits how much audio is used to learn the profile. Zero uses all eligible audio.");
        parameters.put("minConfidence", "Sets the minimum confidence required before an estimate or profile is accepted.");
        parameters.put("noiseFloorDb", "Sets the lowest expected background-noise level in dBFS and prevents unstable adaptation below it.");
        parameters.put("adaptDuringSpeech", "Allows the noise estimate to keep changing during speech. Disable it to avoid learning speech as noise.");
        parameters.put("freezeProfile", "Stops updating the learned Noise Profile after it has been established.");
        parameters.put("artifactProtection", "Limits rapid or extreme spectral changes that can create musical-noise or metallic artifacts.");
        parameters.put("channelIndex", "Selects the zero-based input channel to output.");
        parameters.put("fallbackChannel", "Selects the channel used when the requested channel does not exist.");
        parameters.put("weights", "Lists one mixing weight per input channel, separated by commas. Larger values make that channel contribute more to the mono output.");
        parameters.put("normalize", "Divides the mix by the total weight so increasing the number of contributing channels does not automatically make the result louder.");
        parameters.put("gainsDb", "Lists one gain in decibels per channel, separated by commas.");
        parameters.put("polarityInvert", "Lists one 0 or 1 per channel. Use 1 to invert that channel's waveform polarity.");
        parameters.put("referenceChannel", "Selects the channel against which all other channel delays are measured.");
        parameters.put("maxCorrectionSamples", "Limits the largest channel delay correction to avoid matching unrelated waveform features.");
        parameters.put("fractionalDelay", "Allows sub-sample delay correction by interpolation instead of rounding every delay to a whole sample.");
        parameters.put("delaysSamples", "Lists manual per-channel delays in samples, separated by commas.");
        parameters.put("preferredChannel", "Prefers this zero-based channel when it is usable. Use -1 for fully automatic selection.");
        parameters.put("evaluationWindowMs", "Sets the time window used to evaluate and compare channel quality.");
        parameters.put("minHoldMs", "Keeps a selected channel for at least this duration before another channel may replace it.");
        parameters.put("switchDuringSpeech", "Allows the selected channel to change during detected speech. Disable it to avoid audible mid-sentence switching.");
        parameters.put("midGainDb", "Sets the gain of the stereo center component shared by left and right.");
        parameters.put("sideGainDb", "Sets the gain of the stereo difference component.");
        parameters.put("sideReduction", "Reduces side content to make centered speech more prominent.");
        parameters.put("monoCompatibilityProtection", "Limits side processing when it would create poor cancellation or instability after mono downmixing.");
        parameters.put("centerAmount", "Sets how strongly the extractor favors content that is similar in the left and right channels.");
        parameters.put("centerBoostDb", "Adds gain to the extracted center content after side reduction.");
        parameters.put("width", "Sets stereo width. Zero approaches mono, one preserves the original width and values above one widen the sides.");
        parameters.put("backend", "Selects the processing implementation. Optional backends may report that a runtime or model is not installed.");
        parameters.put("targetSampleRate", "Requests the sample rate expected by the selected backend. Zero keeps any supported input rate.");
        parameters.put("modelId", "Identifies the optional model or backend profile to use when the selected backend supports one.");
        parameters.put("micPositionsMm", "Lists microphone coordinates as x,y,z values in millimetres, separated by semicolons, in input-channel order.");
        parameters.put("speedOfSoundMmPerS", "Sets the assumed speed of sound used to convert arrival-time differences into direction and delay.");
        parameters.put("maxLagSamples", "Limits the largest inter-microphone delay considered during correlation and direction estimation.");
        parameters.put("targetAzimuthDeg", "Sets the horizontal target direction in degrees for static beamforming.");
        parameters.put("targetElevationDeg", "Sets the vertical target direction in degrees for static beamforming.");
        parameters.put("channelWeights", "Lists one beamforming weight per microphone channel, separated by commas.");
        parameters.put("outputGainDb", "Sets a final gain applied after channels are aligned and summed.");
        parameters.put("tracking", "Uses Direction of Arrival estimates to move the beamformer toward a changing speaker direction.");
        parameters.put("trackingBlockFrames", "Sets how many samples are analyzed for each direction-tracking update.");
        parameters.put("directionSmoothing", "Smooths direction changes. Higher values reduce jitter but follow movement more slowly.");
        parameters.put("maxAngularSpeedDegPerBlock", "Limits how far the tracked direction may move during one update.");
        parameters.put("holdBlocks", "Keeps the last reliable direction for this many updates when confidence drops.");
        parameters.put("fallbackAzimuthDeg", "Sets the horizontal direction used when no reliable tracked direction is available.");
        parameters.put("updateDuringSilence", "Allows direction tracking to update when no speech is active. Disable it to keep noise from steering the beam.");
        parameters.put("targetSpeechLevelDb", "Sets the desired RMS level for detected speech in dBFS.");
        parameters.put("maxGainDb", "Limits how much the Speech Leveler may amplify quiet speech.");
        parameters.put("maxGainChangePerSecond", "Limits gain movement per second to prevent sudden level jumps and pumping.");
        parameters.put("silenceGainLimitDb", "Limits amplification during silence so room noise is not raised to the speech target.");
        parameters.put("noiseProtection", "Reduces leveler gain when the available signal is likely background noise rather than speech.");
        parameters.put("clippingProtection", "Reduces or limits gain when the predicted output would clip.");
        parameters.put("targetLevelDb", "Sets the final RMS or peak target, depending on the selected normalization mode.");
        parameters.put("maxTotalGainDb", "Limits the single final gain that may be applied to make the recording louder.");
        parameters.put("maxTotalAttenuationDb", "Limits the single final attenuation that may be applied to make the recording quieter.");
        parameters.put("peakCeilingDb", "Sets the highest allowed output peak in dBFS after normalization.");
        parameters.put("allowAmplification", "Allows the normalizer to raise recordings that are below the target.");
        parameters.put("allowAttenuation", "Allows the normalizer to lower recordings that are above the target.");
        parameters.put("minDecayDb", "Sets the minimum measured decay required before the analyzer reports a reverberation estimate.");
        parameters.put("maxReverbSeconds", "Limits the longest reverberation time the analyzer will report.");
        parameters.put("predictionDelay", "Sets how many STFT frames WPE skips before predicting late reverberation, preserving more early reflections at larger values.");
        parameters.put("filterLength", "Sets the number of past spectral frames used by the WPE prediction filter.");
        parameters.put("iterations", "Sets how many times the offline WPE filter is re-estimated. More iterations cost more CPU.");
        parameters.put("earlyReflectionPreservation", "Reduces processing of the early room response so direct speech remains natural.");
        parameters.put("blockSizeFrames", "Sets how many spectral frames are processed before a block-adaptive WPE filter update.");
        parameters.put("lowShelfHz", "Sets the corner frequency of the low shelf. Frequencies below it are boosted or cut.");
        parameters.put("lowShelfGainDb", "Boosts or cuts the low band. Zero leaves the low frequencies unchanged.");
        parameters.put("midHz", "Sets the center frequency of the parametric mid band.");
        parameters.put("midGainDb", "Boosts or cuts the mid band around its center frequency. Zero leaves it unchanged.");
        parameters.put("midQ", "Sets the bandwidth of the mid band. Higher Q affects a narrower range of frequencies.");
        parameters.put("highShelfHz", "Sets the corner frequency of the high shelf. Frequencies above it are boosted or cut.");
        parameters.put("highShelfGainDb", "Boosts or cuts the high band. Zero leaves the high frequencies unchanged.");
        parameters.put("loudness", "Enables a soft saturation loudness stage. It raises perceived loudness and limits peaks toward the ceiling instead of clipping, applying the master gain through the limiter.");
        parameters.put("loudnessDriveDb", "Adds extra drive into the loudness saturation. More drive means a louder, denser sound with gentler peaks; it has no effect unless loudness is enabled.");
        PARAMETER_DESCRIPTIONS = Collections.unmodifiableMap(parameters);

        Map<String, String> contextual = new HashMap<String, String>();
        contextual.put(contextKey(AudioBlockType.NOISE_PROFILER, "mode"),
                "Selects whether the profile is learned from detected speech pauses, from the complete recording as noise, or kept from an existing in-run profile.");
        contextual.put(contextKey(AudioBlockType.ADAPTIVE_NOISE_SUPPRESSION, "mode"),
                "Selects whether noise is tracked automatically, learned from silence, or taken from an upstream learned Noise Profile.");
        contextual.put(contextKey(AudioBlockType.CHANNEL_DELAY_ALIGNMENT, "mode"),
                "Selects automatic cross-correlation alignment or manually entered per-channel delays.");
        contextual.put(contextKey(AudioBlockType.FINAL_LOUDNESS_NORMALIZER, "mode"),
                "Selects normalization toward an overall RMS level or toward a peak level.");
        contextual.put(contextKey(AudioBlockType.DEREVERBERATION, "mode"),
                "Selects whole-recording offline WPE, block-adaptive WPE, or the bounded-history streaming algorithm.");
        contextual.put(contextKey(AudioBlockType.SPEECH_ENHANCER, "strength"),
                "Sets how strongly the selected speech-enhancement backend reduces noise and/or reverberation.");
        contextual.put(contextKey(AudioBlockType.VOICE_ISOLATION, "strength"),
                "Sets how strongly competing side content or an optional neural backend is reduced relative to the dominant voice.");
        contextual.put(contextKey(AudioBlockType.PLOSIVE_REDUCTION, "targetFrequencyHz"),
                "Sets the upper region of the low-frequency band watched for plosive bursts.");
        contextual.put(contextKey(AudioBlockType.PLOSIVE_REDUCTION_FFT, "targetFrequencyHz"),
                "Sets the upper region of the low-frequency FFT band watched for plosive bursts.");
        contextual.put(contextKey(AudioBlockType.DE_ESSER, "targetFrequencyHz"),
                "Sets the center of the sibilance band watched for harsh S and SH sounds.");
        contextual.put(contextKey(AudioBlockType.DE_ESSER_FFT, "targetFrequencyHz"),
                "Sets the center of the FFT sibilance band watched for harsh S and SH sounds.");
        contextual.put(contextKey(AudioBlockType.ADAPTIVE_NOISE_SUPPRESSION, "speechProtection"),
                "Optional: eases suppression during detected speech when a Voice Activity Detection block is present upstream. Adaptive Noise Suppression works fully without one, so no VAD block is required.");
        contextual.put(contextKey(AudioBlockType.SPEECH_GATE, "attackMs"),
                "Sets a short fade-in when the gate opens so speech onsets do not start with an audible click.");
        contextual.put(contextKey(AudioBlockType.SPEECH_GATE, "releaseMs"),
                "Sets a short fade-out before the gate reaches exact digital silence, avoiding a click at the close.");
        contextual.put(contextKey(AudioBlockType.SPEECH_GATE, "preRollMs"),
                "Opens the gate this long before detected speech so consonant onsets are preserved.");
        contextual.put(contextKey(AudioBlockType.SPEECH_GATE, "postRollMs"),
                "Holds the gate open this long after detected speech so word endings are not cut off.");
        CONTEXT_PARAMETER_DESCRIPTIONS = Collections.unmodifiableMap(contextual);
    }

    private AudioDspHelp() {
    }

    /** Return a plain-language description for the given built-in block type. */
    public static String blockDescription(AudioBlockType type) {
        return type == null ? "" : valueOrEmpty(BLOCK_DESCRIPTIONS.get(type));
    }

    /** Return a plain-language description for one editable parameter of the given block type. */
    public static String parameterDescription(AudioBlockType type, String parameterKey) {
        if (parameterKey == null) {
            return "";
        }
        String contextual = CONTEXT_PARAMETER_DESCRIPTIONS.get(contextKey(type, parameterKey));
        return contextual == null ? valueOrEmpty(PARAMETER_DESCRIPTIONS.get(parameterKey)) : contextual;
    }

    private static String contextKey(AudioBlockType type, String parameterKey) {
        return (type == null ? "" : type.name()) + ':' + parameterKey;
    }

    private static String valueOrEmpty(String value) {
        return value == null ? "" : value;
    }
}

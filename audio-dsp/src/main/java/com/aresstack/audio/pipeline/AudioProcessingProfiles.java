package com.aresstack.audio.pipeline;

import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;
import com.aresstack.audio.profile.AudioProcessingProfile;

import java.util.ArrayList;
import java.util.List;

/**
 * Provide the immutable built-in profiles shipped with AskAI. Lives in {@code pipeline} (not
 * {@code profile}) because it builds its blocks from the {@link AudioBlockRegistry} descriptors — the
 * single source of default parameters — which keeps {@code profile} free of any dependency back on the
 * registry. All built-in profiles are protected: they can never be overwritten, deleted, imported over or
 * exported.
 */
public final class AudioProcessingProfiles {

    public static final String DEFAULT_PROFILE_ID = "default-speech";
    public static final String OFF_PROFILE_ID = "off";
    public static final String CRYSTAL_VOICE_PROFILE_ID = "crystal-voice";

    private AudioProcessingProfiles() {
    }

    /** @return all built-in profiles, in the order they should appear (Off, Default speech, Crystal voice). */
    public static List<AudioProcessingProfile> builtIns() {
        List<AudioProcessingProfile> builtIns = new ArrayList<AudioProcessingProfile>();
        builtIns.add(off());
        builtIns.add(defaultSpeech());
        builtIns.add(crystalVoice());
        return builtIns;
    }

    /** @return the built-in profile for the id, or null when the id is not a built-in. */
    public static AudioProcessingProfile builtInById(String id) {
        if (id == null) {
            return null;
        }
        String trimmed = id.trim();
        if (DEFAULT_PROFILE_ID.equals(trimmed)) {
            return defaultSpeech();
        }
        if (OFF_PROFILE_ID.equals(trimmed)) {
            return off();
        }
        if (CRYSTAL_VOICE_PROFILE_ID.equals(trimmed)) {
            return crystalVoice();
        }
        return null;
    }

    public static boolean isBuiltInId(String id) {
        return builtInById(id) != null;
    }

    /** A neutral built-in that leaves the audio unchanged (no blocks). */
    public static AudioProcessingProfile off() {
        return new AudioProcessingProfile(OFF_PROFILE_ID, "Off", true,
                new ArrayList<AudioBlockDefinition>());
    }

    public static AudioProcessingProfile defaultSpeech() {
        AudioBlockRegistry registry = AudioBlockRegistry.getInstance();
        List<AudioBlockDefinition> blocks = new ArrayList<AudioBlockDefinition>();
        blocks.add(registry.defaultDefinition(AudioBlockType.CHANNEL_MIXER, "channel-mixer"));
        blocks.add(registry.defaultDefinition(AudioBlockType.LOW_PASS, "anti-alias-low-pass"));
        blocks.add(registry.defaultDefinition(AudioBlockType.RESAMPLER, "resampler"));
        blocks.add(registry.defaultDefinition(AudioBlockType.DC_OFFSET_REMOVAL, "dc-offset"));
        blocks.add(registry.defaultDefinition(AudioBlockType.HIGH_PASS, "high-pass"));
        blocks.add(registry.defaultDefinition(AudioBlockType.NOISE_GATE, "noise-gate"));
        blocks.add(registry.defaultDefinition(AudioBlockType.COMPRESSOR, "compressor"));
        blocks.add(registry.defaultDefinition(AudioBlockType.LIMITER, "limiter"));
        return new AudioProcessingProfile(DEFAULT_PROFILE_ID, "Default speech", true, blocks);
    }

    /**
     * A fuller speech-cleaning chain on top of the default speech path: adaptive noise suppression, a
     * de-esser, voice-activity-gated noise gating and a final equalizer/loudness stage. The equalizer is
     * intentionally shipped disabled because enabling it re-introduces noise; it is present so it can be
     * switched on and tuned per recording.
     */
    public static AudioProcessingProfile crystalVoice() {
        AudioBlockRegistry registry = AudioBlockRegistry.getInstance();
        List<AudioBlockDefinition> blocks = new ArrayList<AudioBlockDefinition>();
        blocks.add(registry.defaultDefinition(AudioBlockType.CHANNEL_MIXER, "cv-channel-mixer"));
        blocks.add(registry.defaultDefinition(AudioBlockType.LOW_PASS, "cv-anti-alias-low-pass"));
        blocks.add(registry.defaultDefinition(AudioBlockType.RESAMPLER, "cv-resampler"));
        blocks.add(registry.defaultDefinition(AudioBlockType.DC_OFFSET_REMOVAL, "cv-dc-offset"));
        blocks.add(registry.defaultDefinition(AudioBlockType.HIGH_PASS, "cv-high-pass"));
        blocks.add(registry.defaultDefinition(AudioBlockType.NOISE_GATE, "cv-noise-gate"));
        blocks.add(registry.defaultDefinition(AudioBlockType.COMPRESSOR, "cv-compressor"));
        blocks.add(registry.defaultDefinition(AudioBlockType.LIMITER, "cv-limiter"));
        blocks.add(registry.defaultDefinition(AudioBlockType.ADAPTIVE_NOISE_SUPPRESSION, "cv-noise-suppression")
                .withParameter("maxAttenuationDb", "16"));
        blocks.add(registry.defaultDefinition(AudioBlockType.DE_ESSER, "cv-de-esser"));
        blocks.add(registry.defaultDefinition(AudioBlockType.VOICE_ACTIVITY_DETECTION, "cv-vad"));
        blocks.add(registry.defaultDefinition(AudioBlockType.NOISE_GATE, "cv-noise-gate-2"));
        blocks.add(registry.defaultDefinition(AudioBlockType.EQUALIZER, "cv-equalizer")
                .withParameter("gainDb", "5")
                .withParameter("loudness", "true")
                .withParameter("loudnessDriveDb", "10")
                .withEnabled(false)); // intentionally disabled: enabling it re-introduces noise
        return new AudioProcessingProfile(CRYSTAL_VOICE_PROFILE_ID, "Crystal voice", true, blocks);
    }
}

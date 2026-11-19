package com.aresstack.audio.pipeline;

import com.aresstack.audio.dsp.DirectionEstimate;
import com.aresstack.audio.dsp.NoiseProfile;
import com.aresstack.audio.dsp.RoomProfile;
import com.aresstack.audio.dsp.SpeechActivityTrack;

import java.util.HashMap;
import java.util.Map;

/**
 * Carry shared, per-run metadata between blocks of one pipeline execution (for example a learned noise
 * profile or a speech-probability track produced by an analysis block and consumed by a later block).
 *
 * <p>This is transient runtime state for a single {@code process(...)} call — it is never persisted into a
 * profile. Phase 1 ships it empty but real, so later analysis/adaptive blocks have a defined channel.</p>
 */
public final class AudioProcessingContext {

    private final Map<String, Object> metadata = new HashMap<String, Object>();
    private SpeechActivityTrack speechActivity;
    private NoiseProfile noiseProfile;
    private RoomProfile roomProfile;
    private Double channelCorrelation;
    private String channelHealthSummary;
    private DirectionEstimate directionEstimate;

    /** Store the per-frame speech-activity track produced by the voice-activity block (typed, no string key). */
    public void setSpeechActivity(SpeechActivityTrack track) {
        this.speechActivity = track;
    }

    /** @return the speech-activity track for this run, or null if no voice-activity block ran. */
    public SpeechActivityTrack getSpeechActivity() {
        return speechActivity;
    }

    /** Store the noise model learned by a Noise Profiler for a later Adaptive Noise Suppression block. */
    public void setNoiseProfile(NoiseProfile profile) {
        this.noiseProfile = profile;
    }

    /** @return the learned noise profile for this run, or null if no Noise Profiler ran. */
    public NoiseProfile getNoiseProfile() {
        return noiseProfile;
    }

    /** Store the room/reverberation estimate produced by a Room/Reverb Analyzer for a later dereverb block. */
    public void setRoomProfile(RoomProfile profile) {
        this.roomProfile = profile;
    }

    /** @return the estimated room profile for this run, or null if no Room/Reverb Analyzer ran. */
    public RoomProfile getRoomProfile() {
        return roomProfile;
    }

    /** Store the measured inter-channel correlation from a Phase and Correlation Analyzer. */
    public void setChannelCorrelation(double correlation) {
        this.channelCorrelation = correlation;
    }

    /** @return the measured inter-channel correlation for this run, or null if none was measured. */
    public Double getChannelCorrelation() {
        return channelCorrelation;
    }

    /** Store the per-channel health summary from a Channel Health Analyzer. */
    public void setChannelHealthSummary(String summary) {
        this.channelHealthSummary = summary;
    }

    /** @return the channel health summary for this run, or null if no analyzer ran. */
    public String getChannelHealthSummary() {
        return channelHealthSummary;
    }

    /** Store the estimated direction of arrival produced by a Direction of Arrival Analyzer. */
    public void setDirectionEstimate(DirectionEstimate estimate) {
        this.directionEstimate = estimate;
    }

    /** @return the estimated direction of arrival for this run, or null if no analyzer ran. */
    public DirectionEstimate getDirectionEstimate() {
        return directionEstimate;
    }

    public void put(String key, Object value) {
        metadata.put(key, value);
    }

    public Object get(String key) {
        return metadata.get(key);
    }

    public boolean has(String key) {
        return metadata.containsKey(key);
    }
}

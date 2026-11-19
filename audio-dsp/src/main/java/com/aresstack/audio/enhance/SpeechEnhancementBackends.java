package com.aresstack.audio.enhance;

import com.aresstack.audio.domain.PcmAudioFormat;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;

/**
 * Registry of speech-enhancement backends. It holds the always-available pure-Java backends, the known
 * optional (native/model) adapters that report their own availability, and any providers contributed on the
 * classpath through the {@link SpeechEnhancementBackend} service loader (an optional module can add a real
 * native backend without the core depending on it). Resolution is by stable id, never by display name.
 */
public final class SpeechEnhancementBackends {

    private static final Map<String, SpeechEnhancementBackend> BACKENDS =
            new LinkedHashMap<String, SpeechEnhancementBackend>();

    static {
        register(new PureJavaSpeechEnhancer());
        register(new CombinedDenoiseDereverbEnhancer());
        register(new RnnoiseSpeechEnhancer());
        register(new DeepFilterNetSpeechEnhancer());
        for (SpeechEnhancementBackend backend : ServiceLoader.load(SpeechEnhancementBackend.class)) {
            register(backend); // optional external modules override/extend the built-ins
        }
    }

    private SpeechEnhancementBackends() {
    }

    public static synchronized void register(SpeechEnhancementBackend backend) {
        if (backend != null && backend.id() != null) {
            BACKENDS.put(backend.id(), backend);
        }
    }

    /** @return the backend for an id, or null when the id is unknown (a stale/uninstalled reference). */
    public static synchronized SpeechEnhancementBackend resolve(String id) {
        return id == null ? null : BACKENDS.get(id.trim());
    }

    public static synchronized List<SpeechEnhancementBackend> all() {
        return new ArrayList<SpeechEnhancementBackend>(BACKENDS.values());
    }

    /** @return the availability of a backend id for a format, or NOT_INSTALLED when the id is unknown. */
    public static BackendAvailability availability(String id, PcmAudioFormat format) {
        SpeechEnhancementBackend backend = resolve(id);
        return backend == null ? BackendAvailability.NOT_INSTALLED : backend.availability(format);
    }
}

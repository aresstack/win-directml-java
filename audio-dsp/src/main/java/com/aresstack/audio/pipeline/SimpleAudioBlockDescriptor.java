package com.aresstack.audio.pipeline;

import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A data-driven {@link AudioBlockDescriptor} built from a type, its parameters, capabilities, a processor
 * factory and a summary function. Lets the registry declare every block in one place without a subclass
 * per type.
 */
public final class SimpleAudioBlockDescriptor implements AudioBlockDescriptor {

    /** Creates a fresh processor for one pipeline run. */
    public interface ProcessorFactory {
        AudioBlockProcessor create();
    }

    /** Produces a short canvas summary for a block. */
    public interface Summarizer {
        String summarize(AudioBlockDefinition block);
    }

    private final AudioBlockType type;
    private final String displayName;
    private final AudioBlockCategory category;
    private final List<AudioParameterDescriptor> parameters;
    private final AudioBlockCapabilities capabilities;
    private final ProcessorFactory processorFactory;
    private final Summarizer summarizer;

    public SimpleAudioBlockDescriptor(AudioBlockType type, String displayName, AudioBlockCategory category,
                                      List<AudioParameterDescriptor> parameters,
                                      AudioBlockCapabilities capabilities,
                                      ProcessorFactory processorFactory, Summarizer summarizer) {
        this.type = type;
        this.displayName = displayName;
        this.category = category;
        this.parameters = Collections.unmodifiableList(new ArrayList<AudioParameterDescriptor>(parameters));
        this.capabilities = capabilities;
        this.processorFactory = processorFactory;
        this.summarizer = summarizer;
    }

    public AudioBlockType getType() {
        return type;
    }

    public String getTypeId() {
        return type.name();
    }

    public String getDisplayName() {
        return displayName;
    }

    public AudioBlockCategory getCategory() {
        return category;
    }

    public List<AudioParameterDescriptor> getParameters() {
        return parameters;
    }

    public Map<String, String> defaultParameters() {
        Map<String, String> defaults = new LinkedHashMap<String, String>();
        for (int i = 0; i < parameters.size(); i++) {
            AudioParameterDescriptor parameter = parameters.get(i);
            defaults.put(parameter.getKey(), parameter.getDefaultValue());
        }
        return defaults;
    }

    public AudioBlockDefinition createDefaultDefinition(String id) {
        return new AudioBlockDefinition(id, type, true, defaultParameters());
    }

    public AudioBlockProcessor createProcessor() {
        return processorFactory.create();
    }

    public AudioBlockCapabilities getCapabilities() {
        return capabilities;
    }

    public String summarize(AudioBlockDefinition block) {
        return summarizer.summarize(block);
    }
}

package com.aresstack.windirectml.sidecar;

import com.aresstack.windirectml.encoder.EmbeddingModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Locale;

/**
 * Wählt das Embedding-Backend anhand des Systemproperty
 * {@code -Dembed.backend}.
 * <p>
 * Drei Modi:
 * <ul>
 *   <li>{@code cpu} – {@code CpuMiniLmEncoder} erzwingen. Fehler beim
 *       Laden wird als {@link IllegalStateException} hochgeworfen.</li>
 *   <li>{@code directml} – {@code DirectMlMiniLmEncoder} erzwingen. Wenn
 *       DirectML nicht verfügbar ist, schlägt die Initialisierung sichtbar
 *       fehl ({@link IllegalStateException}). Es gibt explizit keinen
 *       stillen Fallback in diesem Modus.</li>
 *   <li>{@code auto} (Default) – DirectML versuchen, bei Fehler sauber auf
 *       CPU zurückfallen und eine Warnung in den Logger schreiben.</li>
 * </ul>
 * <p>
 * Der Selector kennt die konkreten Encoder-Klassen nicht direkt: die
 * Loader werden als {@link EncoderLoader} hineingereicht. Das hält die
 * Klasse für Unit-Tests ohne GPU testbar.
 */
public final class EmbeddingBackendSelector {

    private static final Logger log = LoggerFactory.getLogger(EmbeddingBackendSelector.class);

    /**
     * Backend-Modus, abgeleitet aus {@code -Dembed.backend}.
     */
    public enum Mode {
        CPU, DIRECTML, AUTO;

        /**
         * Parst den Wert eines {@code -Dembed.backend}-Properties.
         * {@code null}/leer ⇒ {@link #AUTO}. Unbekannte Werte werfen
         * {@link IllegalArgumentException}.
         */
        public static Mode parse(String raw) {
            if (raw == null || raw.isBlank()) return AUTO;
            return switch (raw.trim().toLowerCase(Locale.ROOT)) {
                case "cpu" -> CPU;
                case "directml", "dml" -> DIRECTML;
                case "auto" -> AUTO;
                default -> throw new IllegalArgumentException(
                        "Unknown embed.backend: '" + raw
                                + "' (expected one of: cpu, directml, auto)");
            };
        }

        /**
         * Stable lowercase token used in health/log output.
         */
        public String token() {
            return switch (this) {
                case CPU -> "cpu";
                case DIRECTML -> "directml";
                case AUTO -> "auto";
            };
        }
    }

    /**
     * Lädt einen {@link EmbeddingModel} aus einem Modellverzeichnis.
     */
    @FunctionalInterface
    public interface EncoderLoader {
        EmbeddingModel load(Path modelDir) throws Exception;
    }

    /**
     * Ergebnis einer Selector-Auswahl.
     */
    public record Selection(EmbeddingModel model,
                            String backend,
                            String warning,
                            boolean fallback) {
    }

    private final EncoderLoader cpuLoader;
    private final EncoderLoader directmlLoader;

    public EmbeddingBackendSelector(EncoderLoader cpuLoader, EncoderLoader directmlLoader) {
        this.cpuLoader = cpuLoader;
        this.directmlLoader = directmlLoader;
    }

    /**
     * Wählt den passenden Encoder entsprechend {@code mode} und lädt ihn
     * aus {@code modelDir}.
     *
     * @throws IllegalStateException wenn der erzwungene Pfad fehlschlägt
     *                               bzw. bei {@code AUTO} sowohl DirectML als auch CPU scheitern.
     */
    public Selection select(Mode mode, Path modelDir) {
        switch (mode) {
            case CPU: {
                try {
                    EmbeddingModel m = cpuLoader.load(modelDir);
                    log.info("embed.backend=cpu: CPU encoder loaded from {}", modelDir);
                    return new Selection(m, "cpu", null, false);
                } catch (Exception e) {
                    throw new IllegalStateException(
                            "embed.backend=cpu requested but CPU encoder failed to load: "
                                    + e.getMessage(), e);
                }
            }
            case DIRECTML: {
                try {
                    EmbeddingModel m = directmlLoader.load(modelDir);
                    log.info("embed.backend=directml: DirectML encoder loaded from {}", modelDir);
                    return new Selection(m, "directml", null, false);
                } catch (Exception e) {
                    throw new IllegalStateException(
                            "embed.backend=directml requested but DirectML encoder failed to load: "
                                    + e.getMessage(), e);
                }
            }
            case AUTO: {
                try {
                    EmbeddingModel m = directmlLoader.load(modelDir);
                    log.info("embed.backend=auto: DirectML encoder loaded from {}", modelDir);
                    return new Selection(m, "directml", null, false);
                } catch (Exception primary) {
                    String warn = "embed.backend=auto: DirectML unavailable, falling back to CPU – "
                            + primary.getMessage();
                    log.warn(warn);
                    try {
                        EmbeddingModel m = cpuLoader.load(modelDir);
                        return new Selection(m, "cpu", warn, true);
                    } catch (Exception secondary) {
                        throw new IllegalStateException(
                                "embed.backend=auto: both DirectML and CPU encoders failed (directml="
                                        + primary.getMessage() + "; cpu=" + secondary.getMessage() + ")",
                                secondary);
                    }
                }
            }
            default:
                throw new IllegalStateException("unreachable mode: " + mode);
        }
    }
}


package com.aresstack.windirectml.examples;

import com.aresstack.windirectml.encoder.e5.E5Variant;
import com.aresstack.windirectml.runtime.facade.*;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * Minimal example showing Java 21 direct API use without the sidecar.
 * <p>
 * This demonstrates:
 * <ol>
 *   <li>Creating a {@link LocalMlRuntime} with a backend configuration.</li>
 *   <li>Loading an embedding model and producing vectors.</li>
 *   <li>Batch-embedding multiple texts.</li>
 *   <li>Loading a reranker model and scoring documents.</li>
 * </ol>
 *
 * <p>Run with:
 * <pre>{@code ./gradlew :examples:java21-direct-api:run}</pre>
 */
public class DirectApiExample {

    public static void main(String[] args) {
        // 1. Create runtime (defaults to Backend.AUTO: try DirectML, fall back to CPU)
        LocalMlRuntime runtime = LocalMlRuntime.create();
        System.out.println("Runtime created with backend: " + runtime.backend());

        // 2. Load embedding model
        Path embeddingModelDir = Path.of("model/all-MiniLM-L6-v2");
        var embeddingConfig = EmbeddingModelConfig.miniLm(embeddingModelDir);

        try (LocalEmbeddingModel embeddings = runtime.loadEmbeddingModel(embeddingConfig)) {
            System.out.println("Embedding model loaded, dimension=" + embeddings.dimension());

            // Single embedding
            float[] vector = embeddings.embed("Hello, world!");
            System.out.println("Single embed: dim=" + vector.length
                    + ", first 5 values=" + Arrays.toString(Arrays.copyOf(vector, 5)));

            // Batch embedding
            List<float[]> batch = embeddings.embedBatch(List.of(
                    "The cat sat on the mat.",
                    "Machine learning is fascinating.",
                    "Java 21 brings modern features."
            ));
            System.out.println("Batch embed: " + batch.size() + " vectors");
            for (int i = 0; i < batch.size(); i++) {
                System.out.println("  [" + i + "] dim=" + batch.get(i).length);
            }
        } catch (Exception e) {
            System.err.println("Embedding failed (model files present?): " + e.getMessage());
        }

        // 3. Load reranker model
        Path rerankerModelDir = Path.of("model/cross-encoder-ms-marco-MiniLM-L-6-v2");
        var rerankerConfig = new RerankerModelConfig(rerankerModelDir);

        try (LocalRerankerModel reranker = runtime.loadRerankerModel(rerankerConfig)) {
            System.out.println("\nReranker loaded: " + reranker.modelName());

            var results = reranker.rerank("What is machine learning?", List.of(
                    "Machine learning is a branch of artificial intelligence.",
                    "The weather today is sunny.",
                    "Deep learning uses neural networks."
            ));
            System.out.println("Rerank results:");
            for (var r : results) {
                System.out.printf("  index=%d, score=%.4f%n", r.originalIndex(), r.score());
            }
        } catch (Exception e) {
            System.err.println("Reranking failed (model files present?): " + e.getMessage());
        }

        // 4. Demonstrate unsupported model error
        try {
            runtime.loadEmbeddingModel(new EmbeddingModelConfig(
                    Path.of("dummy"), "sentencepiece-xlmr", null, null));
        } catch (UnsupportedModelException e) {
            System.out.println("\nExpected error for unsupported family: " + e.getMessage());
        } catch (Exception e) {
            // Won't reach here for unsupported families (they throw before loading)
            System.err.println("Unexpected: " + e.getMessage());
        }
    }
}

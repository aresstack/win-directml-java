package com.aresstack.windirectml.examples.java8;

import com.aresstack.windirectml.sidecar.client.BatchEmbeddingResult;
import com.aresstack.windirectml.sidecar.client.EmbeddingResult;
import com.aresstack.windirectml.sidecar.client.HealthResult;
import com.aresstack.windirectml.sidecar.client.JsonRpcError;
import com.aresstack.windirectml.sidecar.client.RerankResult;
import com.aresstack.windirectml.sidecar.client.SidecarClient;
import com.aresstack.windirectml.sidecar.client.SidecarClientConfig;
import com.aresstack.windirectml.sidecar.client.SidecarException;
import com.aresstack.windirectml.sidecar.client.SidecarTimeoutException;

import java.util.Arrays;
import java.util.List;

public final class Java8ClientExample {
    private Java8ClientExample() {
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            printUsage();
            return;
        }

        String backend = args.length >= 3 ? args[2] : "auto";

        SidecarClientConfig config = new SidecarClientConfig();
        config.setJavaExecutable("java");
        config.setSidecarJarPath(args[0]);
        config.setModelDirectory(args[1]);
        config.setEmbedModel("minilm");
        config.setEmbedBackend(backend);
        config.setRerankBackend(backend);
        if (args.length >= 4) {
            config.setRerankModelDirectory(args[3]);
        }
        config.setRequestTimeoutMillis(60000L);
        config.setSummarizeTimeoutMillis(300000L);

        SidecarClient client = new SidecarClient(config);
        try {
            client.start();

            HealthResult health = client.health();
            System.out.println("embeddingBackend=" + health.getEmbeddingBackend()
                    + ", embeddingReady=" + health.isEmbeddingReady()
                    + ", rerankerReady=" + health.isRerankerReady());

            EmbeddingResult first = client.embed("A cat sits on the mat.");
            EmbeddingResult second = client.embed("A feline rests on a rug.");
            System.out.printf("dim=%d cosine=%.4f%n",
                    first.getDimension(),
                    EmbeddingResult.cosine(first.getVector(), second.getVector()));

            List<String> documents = Arrays.asList(
                    "GPUs accelerate linear algebra.",
                    "DirectML targets Windows hardware.",
                    "Cats often nap on rugs.");
            BatchEmbeddingResult batch = client.embedBatch(documents);
            System.out.println("batchCount=" + batch.getCount()
                    + ", batchDim=" + batch.getDimension());

            if (health.isRerankerReady()) {
                RerankResult reranked = client.rerank(
                        "How do GPUs speed up linear algebra?",
                        documents,
                        3);
                for (RerankResult.Item item : reranked.getItems()) {
                    System.out.printf("index=%d score=%.4f%n",
                            item.getIndex(), item.getScore());
                }
            }

            try {
                client.summarize("demonstrate structured errors", 32);
            } catch (JsonRpcError e) {
                System.out.println("summarize JSON-RPC error " + e.getCode()
                        + ": " + e.getMessage());
            }
        } catch (JsonRpcError e) {
            System.err.println("JSON-RPC error " + e.getCode() + ": " + e.getMessage());
        } catch (SidecarTimeoutException e) {
            System.err.println("Timeout: " + e.getMessage());
        } catch (SidecarException e) {
            System.err.println("Sidecar failure: " + e.getMessage());
        } finally {
            client.shutdown();
            Integer exit = client.lastExitCode();
            System.out.println("exit=" + (exit == null ? "unknown" : exit.toString())
                    + (client.lastStopForced() ? " forced" : ""));
        }
    }

    private static void printUsage() {
        System.out.println("Usage: java -jar examples-java8-client.jar <runtimeJar> <modelDir> [auto|directml|cpu] [rerankerDir]");
        System.out.println("No arguments: print this help and exit without starting a child process.");
    }
}

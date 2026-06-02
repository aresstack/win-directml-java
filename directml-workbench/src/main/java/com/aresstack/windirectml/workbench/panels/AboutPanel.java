package com.aresstack.windirectml.workbench.panels;

import javax.swing.*;
import java.awt.*;

/**
 * About panel explaining this workbench uses the direct Java 21 API.
 */
public final class AboutPanel extends JPanel {

    public AboutPanel() {
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        var text = new JTextArea();
        text.setEditable(false);
        text.setLineWrap(true);
        text.setWrapStyleWord(true);
        text.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 13));
        text.setText("""
                ═══════════════════════════════════════════════════════════
                  DirectML Workbench – Java 21 Direct API Demo
                ═══════════════════════════════════════════════════════════
                
                This workbench demonstrates the public directml-runtime Java 21 API
                for local ML inference on Windows.
                
                Architecture:
                  directml-workbench
                    → directml-runtime
                    → directml-encoder
                    → directml-windows-bindings
                    → Windows CPU / DirectML
                
                This is NOT the legacy JSON-RPC sidecar workbench.
                The JSON-RPC sidecar (directml-sidecar) is legacy beta code and is
                not required for this demo. This workbench calls the public API
                (LocalMlRuntime, EmbeddingModelConfig, RerankerModelConfig) directly.
                
                ═══════════════════════════════════════════════════════════
                  Required JVM Flags
                ═══════════════════════════════════════════════════════════
                
                The FFM (Foreign Function & Memory) preview runtime requires:
                
                  --enable-preview --enable-native-access=ALL-UNNAMED
                
                Example:
                  java --enable-preview --enable-native-access=ALL-UNNAMED \\
                       -jar directml-workbench-all.jar
                
                ═══════════════════════════════════════════════════════════
                  Supported Models
                ═══════════════════════════════════════════════════════════
                
                Embedding models:
                  • sentence-transformers/all-MiniLM-L6-v2 (384 dim)
                  • intfloat/e5-small-v2 (384 dim)
                  • intfloat/e5-base-v2 (768 dim)
                  • intfloat/e5-large-v2 (1024 dim)
                
                Reranker models:
                  • cross-encoder/ms-marco-MiniLM-L-6-v2
                  • cross-encoder/ms-marco-MiniLM-L-12-v2
                
                Use the Download tab to fetch model files from Hugging Face.
                Models are stored in the model/ directory and are NOT bundled.
                
                ═══════════════════════════════════════════════════════════
                  License: MIT
                ═══════════════════════════════════════════════════════════
                """);

        add(new JScrollPane(text), BorderLayout.CENTER);
    }
}

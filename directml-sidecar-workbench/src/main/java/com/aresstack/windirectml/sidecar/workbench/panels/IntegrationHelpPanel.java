package com.aresstack.windirectml.sidecar.workbench.panels;

import javax.swing.BorderFactory;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

import java.awt.BorderLayout;
import java.awt.Font;

/**
 * Read-only tab that shows Java-8 sample code for embedding the sidecar
 * into a host application using the same {@code directml-sidecar-client-java8}
 * library that the workbench itself uses.
 */
public final class IntegrationHelpPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private static final String SAMPLE =
        "// Java-8 integration: same library the workbench uses.\n"
      + "// build.gradle:\n"
      + "//   implementation project(':directml-sidecar-client-java8')\n"
      + "\n"
      + "import com.aresstack.windirectml.sidecar.client.*;\n"
      + "\n"
      + "public class HostApp {\n"
      + "    public static void main(String[] args) throws Exception {\n"
      + "        SidecarClientConfig config = new SidecarClientConfig();\n"
      + "        config.setJavaExecutable(\"C:\\\\Program Files\\\\Java\\\\jdk-21\\\\bin\\\\java.exe\");\n"
      + "        config.setSidecarJarPath(\"directml-sidecar.jar\");\n"
      + "        config.setModelDirectory(\"model/all-MiniLM-L6-v2\");\n"
      + "        config.setEmbedBackend(\"directml\");        // or \"auto\" / \"cpu\"\n"
      + "        config.setDirectmlDebug(false);\n"
      + "        config.setRequestTimeoutMillis(30_000L);\n"
      + "\n"
      + "        SidecarClient client = new SidecarClient(config);\n"
      + "        try {\n"
      + "            client.start();\n"
      + "\n"
      + "            HealthResult health = client.health();\n"
      + "            System.out.println(\"backend = \" + health.getEmbeddingBackend()\n"
      + "                    + \", ready = \" + health.isEmbeddingReady());\n"
      + "\n"
      + "            EmbeddingResult a = client.embed(\"A cat sits on the mat.\");\n"
      + "            EmbeddingResult b = client.embed(\"A feline rests on a rug.\");\n"
      + "            double cos = EmbeddingResult.cosine(a.getVector(), b.getVector());\n"
      + "            System.out.printf(\"cos(A,B) = %.4f%n\", cos);\n"
      + "\n"
      + "            try {\n"
      + "                SummaryResult s = client.summarize(\"long input text…\", 256);\n"
      + "                System.out.println(s.getText());\n"
      + "            } catch (JsonRpcError e) {\n"
      + "                System.err.println(\"summarize not available: \"\n"
      + "                        + e.getCode() + \" / \" + e.getMessage());\n"
      + "            }\n"
      + "        } finally {\n"
      + "            client.shutdown();\n"
      + "        }\n"
      + "    }\n"
      + "}\n"
      + "\n"
      + "// Threading reminder for Swing host apps:\n"
      + "//   – never call health()/embed()/summarize() on the EDT\n"
      + "//   – wrap them in SwingWorker.doInBackground() like the workbench does\n"
      + "//   – shutdown() is safe to call multiple times\n";

    public IntegrationHelpPanel() {
        setLayout(new BorderLayout());
        setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));
        JTextArea area = new JTextArea(SAMPLE);
        area.setEditable(false);
        area.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        JScrollPane sp = new JScrollPane(area);
        sp.setBorder(BorderFactory.createTitledBorder(
                "Java-8 host integration via directml-sidecar-client-java8"));
        add(sp, BorderLayout.CENTER);
    }
}


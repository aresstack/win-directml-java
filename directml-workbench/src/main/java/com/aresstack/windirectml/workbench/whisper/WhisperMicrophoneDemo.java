package com.aresstack.windirectml.workbench.whisper;

import com.aresstack.windirectml.inference.whisper.WhisperTranscriber;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.Font;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Record local German speech and display Whisper recognition in a standalone window. */
public final class WhisperMicrophoneDemo extends JFrame {
    private final JComboBox<MicrophoneRecorder.Device> devices = new JComboBox<>();
    private final JButton start = new JButton("Aufnahme starten");
    private final JButton stop = new JButton("Aufnahme stoppen");
    private final JTextArea transcript = new JTextArea(8, 56);
    private final JLabel status = new JLabel("Bereit");
    private final MicrophoneRecorder recorder = new MicrophoneRecorder();
    private final ExecutorService background = Executors.newSingleThreadExecutor(job -> {
        Thread worker = new Thread(job, "whisper-transcription");
        worker.setDaemon(true);
        return worker;
    });
    private final Path modelDirectory;
    private WhisperTranscriber transcriber;

    public static void main(String[] args) {
        Path model = locateModel(args);
        SwingUtilities.invokeLater(() -> new WhisperMicrophoneDemo(model).setVisible(true));
    }

    private WhisperMicrophoneDemo(Path modelDirectory) {
        super("Whisper – deutscher Mikrofontest");
        this.modelDirectory = modelDirectory;
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        initializeControls();
        pack();
        setLocationRelativeTo(null);
        addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
            public void windowClosed(java.awt.event.WindowEvent event) {
                recorder.close();
                background.shutdownNow();
            }
        });
    }

    private void initializeControls() {
        JPanel actions = new JPanel(new FlowLayout(FlowLayout.LEFT));
        actions.add(new JLabel("Mikrofon:"));
        for (MicrophoneRecorder.Device device : MicrophoneRecorder.availableDevices()) {
            devices.addItem(device);
        }
        actions.add(devices);
        actions.add(start);
        actions.add(stop);
        start.addActionListener(event -> beginRecording());
        stop.addActionListener(event -> finishRecording());

        transcript.setLineWrap(true);
        transcript.setWrapStyleWord(true);
        transcript.setEditable(false);
        transcript.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 18));
        transcript.setText("Wähle dein Mikrofon, starte die Aufnahme, sprich einen kurzen Satz "
                + "und stoppe die Aufnahme. Der erkannte Text erscheint hier.");
        stop.setEnabled(false);

        JPanel content = new JPanel(new BorderLayout(8, 8));
        content.setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));
        content.add(actions, BorderLayout.NORTH);
        content.add(new JScrollPane(transcript), BorderLayout.CENTER);
        content.add(status, BorderLayout.SOUTH);
        setContentPane(content);

        if (devices.getItemCount() == 0) {
            disableRecording("Kein Mikrofon gefunden. Prüfe die Windows-Mikrofonberechtigung.");
        } else if (!modelAvailable()) {
            disableRecording("Whisper-Tiny-Modell fehlt: " + modelDirectory.toAbsolutePath());
        }
    }

    private boolean modelAvailable() {
        return Files.isRegularFile(modelDirectory.resolve("model.safetensors"))
                && Files.isRegularFile(modelDirectory.resolve("config.json"))
                && Files.isRegularFile(modelDirectory.resolve("vocab.json"))
                && Files.isRegularFile(modelDirectory.resolve("tokenizer.json"));
    }

    private void disableRecording(String reason) {
        start.setEnabled(false);
        status.setText(reason);
        transcript.setText(reason);
    }

    private void beginRecording() {
        MicrophoneRecorder.Device selected = (MicrophoneRecorder.Device) devices.getSelectedItem();
        start.setEnabled(false);
        devices.setEnabled(false);
        status.setText("Mikrofon wird gestartet …");
        background.execute(() -> {
            try {
                recorder.start(selected);
                SwingUtilities.invokeLater(() -> {
                    transcript.setText("");
                    stop.setEnabled(true);
                    status.setText("Aufnahme läuft (maximal 12 Sekunden). Bitte sprechen.");
                });
            } catch (Exception exception) {
                showFailure("Mikrofon konnte nicht geöffnet werden", exception);
            }
        });
    }

    private void finishRecording() {
        stop.setEnabled(false);
        status.setText("Aufnahme beendet. Transkription läuft lokal …");
        background.execute(() -> {
            try {
                float[] audio = recorder.stop();
                if (transcriber == null) {
                    transcriber = new WhisperTranscriber(modelDirectory);
                }
                String recognized = transcriber.transcribeShortRecording(audio, 80);
                SwingUtilities.invokeLater(() -> {
                    transcript.setText(recognized.isBlank() ? "(Keine Sprache erkannt)" : recognized);
                    status.setText("Transkription abgeschlossen. Neue Aufnahme möglich.");
                    start.setEnabled(true);
                    devices.setEnabled(true);
                });
            } catch (Exception exception) {
                showFailure("Transkription fehlgeschlagen", exception);
            }
        });
    }

    private void showFailure(String description, Exception exception) {
        SwingUtilities.invokeLater(() -> {
            status.setText(description + ": " + exception.getMessage());
            transcript.setText(description + "\n" + exception);
            devices.setEnabled(true);
            start.setEnabled(modelAvailable() && devices.getItemCount() > 0);
            stop.setEnabled(false);
        });
    }

    private static Path locateModel(String[] args) {
        if (args.length > 0) {
            return Paths.get(args[0]).toAbsolutePath();
        }
        Path current = Paths.get("").toAbsolutePath();
        for (int index = 0; current != null && index < 5; index++, current = current.getParent()) {
            Path candidate = current.resolve("model").resolve("whisper-tiny");
            if (Files.isRegularFile(candidate.resolve("model.safetensors"))) {
                return candidate;
            }
        }
        return Paths.get("model", "whisper-tiny").toAbsolutePath();
    }
}

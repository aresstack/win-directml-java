package com.aresstack.windirectml.sidecar.protocol.validation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelValidatorTest {
    private static final Charset UTF8 = Charset.forName("UTF-8");

    @Test
    void acceptsMiniLmShape(@TempDir Path dir) throws Exception {
        writeConfig(dir, 384, 6, 12);
        writeTokenizer(dir, "WordPiece");
        Files.write(dir.resolve("model.safetensors"), new byte[]{1});
        ValidationReport report = ModelValidator.validate(dir.toFile(), ModelValidator.minilmExpectation());
        assertTrue(report.isOk(), report.format());
    }

    @Test
    void rejectsWrongMiniLmShape(@TempDir Path dir) throws Exception {
        writeConfig(dir, 768, 6, 12);
        writeTokenizer(dir, "WordPiece");
        Files.write(dir.resolve("model.safetensors"), new byte[]{1});
        ValidationReport report = ModelValidator.validate(dir.toFile(), ModelValidator.minilmExpectation());
        assertFalse(report.isOk(), report.format());
        assertTrue(report.format().contains("hidden_size"));
    }

    @Test
    void rejectsWrongTokenizerFamily(@TempDir Path dir) throws Exception {
        writeConfig(dir, 384, 6, 12);
        writeTokenizer(dir, "SentencePiece");
        Files.write(dir.resolve("model.safetensors"), new byte[]{1});
        ValidationReport report = ModelValidator.validate(dir.toFile(), ModelValidator.minilmExpectation());
        assertFalse(report.isOk(), report.format());
        assertTrue(report.format().contains("SentencePiece"));
    }

    @Test
    void classifiesBaseStsEnDeAsPlanned(@TempDir Path dir) throws Exception {
        writeConfig(dir, 768, 12, 12);
        writeTokenizer(dir, "SentencePiece");
        Files.write(dir.resolve("model.safetensors"), new byte[]{1});
        ValidationReport report = ModelValidator.validate(dir.toFile(), ModelValidator.e5Expectation("base-sts-en-de"));
        assertFalse(report.isOk(), report.format());
        assertTrue(report.format().contains("planned"));
        assertTrue(report.format().contains("SentencePiece"));
    }

    @Test
    void acceptsE5BaseV2WordPiece(@TempDir Path dir) throws Exception {
        writeConfig(dir, 768, 12, 12);
        writeTokenizer(dir, "WordPiece");
        Files.write(dir.resolve("pytorch_model.bin"), new byte[]{1});
        ValidationReport report = ModelValidator.validate(dir.toFile(), ModelValidator.e5Expectation("base-v2"));
        assertTrue(report.isOk(), report.format());
    }

    @Test
    void missingDirectoryIsNotOk(@TempDir Path dir) {
        ValidationReport report = ModelValidator.validate(dir.resolve("missing").toFile(), ModelValidator.minilmExpectation());
        assertFalse(report.isOk(), report.format());
        assertTrue(report.format().contains("does not exist"));
    }

    private static void writeConfig(Path dir, int hidden, int layers, int heads) throws Exception {
        String json = "{\n"
                + "  \"hidden_size\": " + hidden + ",\n"
                + "  \"num_hidden_layers\": " + layers + ",\n"
                + "  \"num_attention_heads\": " + heads + "\n"
                + "}\n";
        Files.write(dir.resolve("config.json"), json.getBytes(UTF8));
    }

    private static void writeTokenizer(Path dir, String type) throws Exception {
        String json = "{\"model\":{\"type\":\"" + type + "\"}}\n";
        Files.write(dir.resolve("tokenizer.json"), json.getBytes(UTF8));
    }
}

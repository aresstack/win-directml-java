package com.aresstack.windirectml.inference.t5;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class JsonT5TokenizerTest {
    @TempDir
    Path tempDir;

    @Test
    void loadsGoogleT5StyleTokenizerJson() throws Exception {
        writeTokenizerFiles(tempDir);

        T5TextTokenizer tokenizer = T5TokenizerLoader.load(tempDir);

        int[] encoded = tokenizer.encode("hello world");

        assertArrayEquals(new int[]{3, 4, 1}, encoded);
        assertEquals("hello world", tokenizer.decode(encoded));
    }

    static void writeTokenizerFiles(Path modelDir) throws Exception {
        Files.createDirectories(modelDir);
        Files.writeString(modelDir.resolve("tokenizer.json"), "{\n" +
                "  \"model\": {\n" +
                "    \"type\": \"Unigram\",\n" +
                "    \"vocab\": [\n" +
                "      [\"<pad>\", 0.0],\n" +
                "      [\"</s>\", 0.0],\n" +
                "      [\"<unk>\", 0.0],\n" +
                "      [\"▁hello\", -1.0],\n" +
                "      [\"▁world\", -1.0]\n" +
                "    ]\n" +
                "  }\n" +
                "}\n", StandardCharsets.UTF_8);
        Files.writeString(modelDir.resolve("spiece.model"), "placeholder", StandardCharsets.UTF_8);
    }
}

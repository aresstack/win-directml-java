package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.SourceTensor;
import com.aresstack.windirectml.inference.model.SourceTensorCatalog;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class SmolLM2TestFixtures {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private SmolLM2TestFixtures() {
    }

    static SmolLM2Config config135(boolean tiedEmbeddings) {
        return new SmolLM2Config("llama", List.of("LlamaForCausalLM"),
                8, 16, 1, 4, 2, 2, 32, 64,
                1.0e-5d, 10000.0d, "silu", false, false, 1, 2, null, tiedEmbeddings);
    }

    static SmolLM2Config config360(boolean tiedEmbeddings) {
        return new SmolLM2Config("llama", List.of("LlamaForCausalLM"),
                12, 24, 1, 6, 3, 2, 40, 128,
                1.0e-5d, 10000.0d, "silu", false, false, 1, 2, 0, tiedEmbeddings);
    }

    static SourceTensorCatalog completeCatalog(SmolLM2Config config, boolean includeLmHead) {
        List<SourceTensor> tensors = new ArrayList<>();
        add(tensors, "model.embed_tokens.weight", config.vocabSize(), config.hiddenSize());
        add(tensors, "model.norm.weight", config.hiddenSize());
        if (includeLmHead) {
            add(tensors, "lm_head.weight", config.vocabSize(), config.hiddenSize());
        }
        for (int layer = 0; layer < config.numHiddenLayers(); layer++) {
            add(tensors, "model.layers." + layer + ".input_layernorm.weight", config.hiddenSize());
            add(tensors, "model.layers." + layer + ".self_attn.q_proj.weight",
                    config.numAttentionHeads() * config.effectiveHeadDim(), config.hiddenSize());
            add(tensors, "model.layers." + layer + ".self_attn.k_proj.weight",
                    config.effectiveKeyValueHeads() * config.effectiveHeadDim(), config.hiddenSize());
            add(tensors, "model.layers." + layer + ".self_attn.v_proj.weight",
                    config.effectiveKeyValueHeads() * config.effectiveHeadDim(), config.hiddenSize());
            add(tensors, "model.layers." + layer + ".self_attn.o_proj.weight",
                    config.hiddenSize(), config.numAttentionHeads() * config.effectiveHeadDim());
            add(tensors, "model.layers." + layer + ".post_attention_layernorm.weight", config.hiddenSize());
            add(tensors, "model.layers." + layer + ".mlp.gate_proj.weight", config.intermediateSize(), config.hiddenSize());
            add(tensors, "model.layers." + layer + ".mlp.up_proj.weight", config.intermediateSize(), config.hiddenSize());
            add(tensors, "model.layers." + layer + ".mlp.down_proj.weight", config.hiddenSize(), config.intermediateSize());
        }
        return new SourceTensorCatalog(tensors);
    }

    static void writeModelDirectory(Path dir, SmolLM2Config config, boolean includeLmHead) throws Exception {
        Files.createDirectories(dir);
        Files.writeString(dir.resolve("config.json"), configJson(config));
        writeSafeTensors(dir.resolve("model.safetensors"), completeCatalog(config, includeLmHead));
    }


    static void writeTokenizerJson(Path file) throws Exception {
        Map<String, Object> model = new LinkedHashMap<>();
        Map<String, Integer> vocab = new LinkedHashMap<>();
        vocab.put("a", 0);
        vocab.put("b", 1);
        vocab.put("ab", 2);
        vocab.put("<|endoftext|>", 3);
        model.put("type", "BPE");
        model.put("vocab", vocab);
        model.put("merges", List.of(List.of("a", "b")));

        Map<String, Object> eosToken = new LinkedHashMap<>();
        eosToken.put("id", 3);
        eosToken.put("content", "<|endoftext|>");
        eosToken.put("special", true);

        Map<String, Object> root = new LinkedHashMap<>();
        root.put("model", model);
        root.put("added_tokens", List.of(eosToken));
        Files.writeString(file, MAPPER.writeValueAsString(root), StandardCharsets.UTF_8);
    }

    static String configJson(SmolLM2Config config) throws Exception {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("model_type", config.modelType());
        root.put("architectures", config.architectures());
        root.put("hidden_size", config.hiddenSize());
        root.put("intermediate_size", config.intermediateSize());
        root.put("num_hidden_layers", config.numHiddenLayers());
        root.put("num_attention_heads", config.numAttentionHeads());
        root.put("num_key_value_heads", config.numKeyValueHeads());
        root.put("head_dim", config.headDim());
        root.put("vocab_size", config.vocabSize());
        root.put("max_position_embeddings", config.maxPositionEmbeddings());
        root.put("rms_norm_eps", config.rmsNormEps());
        root.put("rope_theta", config.ropeTheta());
        root.put("hidden_act", config.hiddenAct());
        root.put("attention_bias", config.attentionBias());
        root.put("mlp_bias", config.mlpBias());
        root.put("bos_token_id", config.bosTokenId());
        root.put("eos_token_id", config.eosTokenId());
        if (config.padTokenId() != null) {
            root.put("pad_token_id", config.padTokenId());
        }
        root.put("tie_word_embeddings", config.tieWordEmbeddings());
        return MAPPER.writeValueAsString(root);
    }

    static void writeSafeTensors(Path file, SourceTensorCatalog catalog) throws Exception {
        Map<String, Object> header = new LinkedHashMap<>();
        long offset = 0L;
        for (SourceTensor tensor : catalog.entries().values()) {
            long byteLength = elementCount(tensor.dims()) * 4L;
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("dtype", "F32");
            entry.put("shape", tensor.dims());
            entry.put("data_offsets", List.of(offset, offset + byteLength));
            header.put(tensor.name(), entry);
            offset += byteLength;
        }
        byte[] headerBytes = MAPPER.writeValueAsBytes(header);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteBuffer length = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        length.putLong(headerBytes.length).flip();
        out.write(length.array());
        out.write(headerBytes);
        out.write(new byte[Math.toIntExact(offset)]);
        Files.write(file, out.toByteArray());
    }

    private static void add(List<SourceTensor> tensors, String name, long... dims) {
        tensors.add(SourceTensor.metadataOnly(name, 1, dims));
    }

    private static long elementCount(long[] dims) {
        long count = 1L;
        for (long dim : dims) {
            count *= dim;
        }
        return count;
    }
}

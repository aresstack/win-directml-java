package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyConfig;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGeneratedTokens;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGreedyTokenSelector;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyStopTokenPolicy;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyTokenizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Slice 4 — Qwen decoder-only adapter compatibility. Verifies Qwen plugs into the shared {@code decoderonly} seams
 * (config, tokenizer, stop policy, token selector) without touching the production {@code Qwen2Runtime} forward path.
 */
class QwenDecoderOnlyCompatibilityTest {

    @Test
    void qwen2ConfigIsADecoderOnlyConfigWithConsistentDimensions() {
        Qwen2Config config = new Qwen2Config(
                896,        // hiddenSize
                14,         // numAttentionHeads
                24,         // numHiddenLayers
                2,          // numKeyValueHeads (GQA)
                151936,     // vocabSize
                32768,      // maxPositionEmbeddings
                4864,       // intermediateSize
                1e-6f,      // rmsNormEps
                1_000_000f, // ropeTheta
                true);      // tieWordEmbeddings

        DecoderOnlyConfig view = config; // compiles only because Qwen2Config implements the shared contract
        assertEquals(24, view.numHiddenLayers());
        assertEquals(896, view.hiddenSize());
        assertEquals(14, view.numAttentionHeads());
        assertEquals(2, view.numKeyValueHeads());
        assertEquals(64, view.headDim()); // 896 / 14
        assertEquals(151936, view.vocabSize());
    }

    @Test
    void qwenTokenizerImplementsTheSharedTokenizerSeam() {
        // Instantiating the tokenizer needs a tokenizer.json fixture; the type binding is what matters here.
        assertTrue(DecoderOnlyTokenizer.class.isAssignableFrom(QwenTokenizer.class));
    }

    @Test
    void qwenStopPolicyStopsOnQwenTerminators() {
        DecoderOnlyStopTokenPolicy stopPolicy = QwenStopTokenPolicy.asDecoderOnlyPolicy();
        assertTrue(stopPolicy.shouldStop(QwenTokenizer.ENDOFTEXT_ID));
        assertTrue(stopPolicy.shouldStop(QwenTokenizer.IM_END_ID));
        assertFalse(stopPolicy.shouldStop(QwenTokenizer.IM_START_ID));
        assertFalse(stopPolicy.shouldStop(42));
    }

    @Test
    void qwenTokenSelectorMatchesTheGreedySelectorAndStopPolicy() {
        float penalty = 1.5f;
        DecoderOnlyStopTokenPolicy stopPolicy = QwenStopTokenPolicy.asDecoderOnlyPolicy();
        QwenTokenSelector selector = new QwenTokenSelector(stopPolicy, penalty);
        DecoderOnlyGreedyTokenSelector greedy = new DecoderOnlyGreedyTokenSelector(stopPolicy);

        DecoderOnlyGeneratedTokens generated = new DecoderOnlyGeneratedTokens(4);
        generated.add(2); // token 2 already generated → penalty applies

        // Repetition penalty mutates logits in place, so each path gets its own copy of identical input.
        float[] viaSeam = {1.0f, 3.0f, 5.0f, 2.0f};
        float[] viaGreedy = viaSeam.clone();
        assertEquals(
                greedy.selectNextToken(viaGreedy, generated, penalty),
                selector.selectNextToken(viaSeam, generated));

        assertTrue(selector.shouldStop(QwenTokenizer.ENDOFTEXT_ID));
        assertFalse(selector.shouldStop(5));
    }
}

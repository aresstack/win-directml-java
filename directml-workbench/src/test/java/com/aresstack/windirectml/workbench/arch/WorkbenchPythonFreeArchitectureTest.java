package com.aresstack.windirectml.workbench.arch;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.aresstack.windirectml.workbench.panels.SummarizerPanel;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * P8 guard: the workbench carries no Python/legacy Gemma remnants after the dormant per-family paths were
 * removed. Locks in that the deletions cannot silently return.
 */
class WorkbenchPythonFreeArchitectureTest {

    /** The removed per-family / native-vs-Python dispatch methods that must not reappear on the panel. */
    private static final Set<String> FORBIDDEN_METHODS = Set.of(
            "runQwenGeneration", "runT5Generation", "runSmolLm2Generation", "runGemma3Generation",
            "runGemma3NativeWarp", "runPhi3Summarizer", "gemmaUsesNativeDirectMl", "gemmaAdapterMode");

    @Test
    void gemma3PythonScriptResourceIsGone() {
        assertNull(WorkbenchPythonFreeArchitectureTest.class.getResource(
                        "/com/aresstack/windirectml/workbench/runtime/gemma3_generate.py"),
                "gemma3_generate.py must not ship in the workbench classpath");
    }

    @Test
    void gemma3ExternalRuntimeRunnerClassDoesNotExist() {
        assertThrows(ClassNotFoundException.class,
                () -> Class.forName("com.aresstack.windirectml.workbench.runtime.Gemma3ExternalRuntimeRunner"),
                "the external Python Gemma runner must be deleted");
    }

    @Test
    void summarizerPanelHasNoDeletedPerFamilyOrPythonMethods() {
        for (Method m : SummarizerPanel.class.getDeclaredMethods()) {
            assertTrue(!FORBIDDEN_METHODS.contains(m.getName()),
                    "SummarizerPanel must not declare the removed method '" + m.getName() + "'; the active path "
                            + "routes through the shared runtime. Present: "
                            + Arrays.toString(SummarizerPanel.class.getDeclaredMethods()));
        }
    }
}

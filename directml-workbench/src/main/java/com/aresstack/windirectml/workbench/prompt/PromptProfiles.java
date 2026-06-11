package com.aresstack.windirectml.workbench.prompt;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Built-in prompt profiles for the Workbench generation models.
 */
final class PromptProfiles {

    private PromptProfiles() {
    }

    static WorkbenchPromptProfile rawOnly() {
        return RawOnlyPromptProfile.INSTANCE;
    }

    static WorkbenchPromptProfile qwenCoder() {
        return QwenCoderPromptProfile.INSTANCE;
    }

    static WorkbenchPromptProfile smolLm2Instruct() {
        return SmolLm2InstructPromptProfile.INSTANCE;
    }

    static WorkbenchPromptProfile googleT5() {
        return GoogleT5PromptProfile.INSTANCE;
    }

    static WorkbenchPromptProfile flanT5() {
        return FlanT5PromptProfile.INSTANCE;
    }

    static WorkbenchPromptProfile codeT5Small() {
        return CodeT5SmallPromptProfile.INSTANCE;
    }

    static WorkbenchPromptProfile codeT5MultiSum() {
        return CodeT5MultiSumPromptProfile.INSTANCE;
    }

    private abstract static class AbstractPromptProfile implements WorkbenchPromptProfile {
        private final List<WorkbenchPromptTemplate> supportedTemplates;

        AbstractPromptProfile(WorkbenchPromptTemplate... supportedTemplates) {
            this.supportedTemplates = Collections.unmodifiableList(Arrays.asList(supportedTemplates));
        }

        @Override
        public final List<WorkbenchPromptTemplate> supportedTemplates() {
            return supportedTemplates;
        }

        @Override
        public RenderedWorkbenchPrompt render(WorkbenchPromptTemplate template, String inputText) {
            WorkbenchPromptTemplate safeTemplate = template == null ? WorkbenchPromptTemplate.NONE : template;
            String safeInput = inputText == null ? "" : inputText;
            if (!supportedTemplates.contains(safeTemplate)) {
                safeTemplate = WorkbenchPromptTemplate.NONE;
            }
            String systemPrompt = renderSystemPrompt(safeTemplate, safeInput);
            if (systemPrompt != null && !systemPrompt.isEmpty()) {
                return new RenderedWorkbenchPrompt(safeInput, systemPrompt);
            }
            return RenderedWorkbenchPrompt.userPromptOnly(renderUserPrompt(safeTemplate, safeInput));
        }

        protected abstract String renderUserPrompt(WorkbenchPromptTemplate template, String inputText);

        protected String renderSystemPrompt(WorkbenchPromptTemplate template, String inputText) {
            return "";
        }
    }

    private static final class RawOnlyPromptProfile extends AbstractPromptProfile {
        private static final RawOnlyPromptProfile INSTANCE = new RawOnlyPromptProfile();

        private RawOnlyPromptProfile() {
            super(WorkbenchPromptTemplate.NONE);
        }

        @Override
        protected String renderUserPrompt(WorkbenchPromptTemplate template, String inputText) {
            return inputText;
        }
    }

    private static final class QwenCoderPromptProfile extends AbstractPromptProfile {
        private static final QwenCoderPromptProfile INSTANCE = new QwenCoderPromptProfile();

        private QwenCoderPromptProfile() {
            super(WorkbenchPromptTemplate.NONE,
                    WorkbenchPromptTemplate.SUMMARIZE_DE,
                    WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN,
                    WorkbenchPromptTemplate.TRANSLATE_TO_ENGLISH,
                    WorkbenchPromptTemplate.EXPLAIN_CODE_DE);
        }

        @Override
        protected String renderUserPrompt(WorkbenchPromptTemplate template, String inputText) {
            if (template == WorkbenchPromptTemplate.SUMMARIZE_DE) {
                return "Fasse den folgenden Text kurz und sachlich auf Deutsch zusammen. "
                        + "Gib nur die Zusammenfassung aus.\n\n" + inputText;
            }
            if (template == WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN) {
                return "Übersetze den folgenden Text ins Deutsche. Gib nur die Übersetzung aus.\n\n" + inputText;
            }
            if (template == WorkbenchPromptTemplate.TRANSLATE_TO_ENGLISH) {
                return "Translate the following text to English. Output only the translation.\n\n" + inputText;
            }
            if (template == WorkbenchPromptTemplate.EXPLAIN_CODE_DE) {
                return "Erkläre den folgenden Code kurz und verständlich auf Deutsch.\n\n" + inputText;
            }
            return inputText;
        }
    }

    private static final class SmolLm2InstructPromptProfile extends AbstractPromptProfile {
        private static final SmolLm2InstructPromptProfile INSTANCE = new SmolLm2InstructPromptProfile();

        private SmolLm2InstructPromptProfile() {
            super(WorkbenchPromptTemplate.NONE,
                    WorkbenchPromptTemplate.SUMMARIZE_DE,
                    WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN,
                    WorkbenchPromptTemplate.TRANSLATE_TO_ENGLISH);
        }

        @Override
        protected String renderUserPrompt(WorkbenchPromptTemplate template, String inputText) {
            // For SmolLM2 chat models, the task instruction goes into the system prompt
            // (via renderSystemPrompt) so it lands in the <|im_start|>system block of the
            // ChatML template. The user prompt stays raw.
            // renderSystemPrompt returns a non-empty string for task templates, which
            // causes AbstractPromptProfile.render() to use systemPrompt + raw userPrompt.
            return inputText;
        }

        @Override
        protected String renderSystemPrompt(WorkbenchPromptTemplate template, String inputText) {
            if (template == WorkbenchPromptTemplate.SUMMARIZE_DE) {
                return "Fasse diesen Text kurz auf Deutsch zusammen. Gib nur die Zusammenfassung aus.";
            }
            if (template == WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN) {
                return "Translate to German. Output only the translation.";
            }
            if (template == WorkbenchPromptTemplate.TRANSLATE_TO_ENGLISH) {
                return "Translate to English. Output only the translation.";
            }
            return "";
        }
    }

    private static final class GoogleT5PromptProfile extends AbstractPromptProfile {
        private static final GoogleT5PromptProfile INSTANCE = new GoogleT5PromptProfile();

        private GoogleT5PromptProfile() {
            super(WorkbenchPromptTemplate.NONE,
                    WorkbenchPromptTemplate.SUMMARIZE_DE,
                    WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN,
                    WorkbenchPromptTemplate.TRANSLATE_TO_ENGLISH);
        }

        @Override
        protected String renderUserPrompt(WorkbenchPromptTemplate template, String inputText) {
            if (template == WorkbenchPromptTemplate.SUMMARIZE_DE) {
                return "summarize: " + inputText;
            }
            if (template == WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN) {
                return "translate English to German: " + inputText;
            }
            if (template == WorkbenchPromptTemplate.TRANSLATE_TO_ENGLISH) {
                return "translate German to English: " + inputText;
            }
            return inputText;
        }
    }

    private static final class FlanT5PromptProfile extends AbstractPromptProfile {
        private static final FlanT5PromptProfile INSTANCE = new FlanT5PromptProfile();

        private FlanT5PromptProfile() {
            super(WorkbenchPromptTemplate.NONE,
                    WorkbenchPromptTemplate.SUMMARIZE_DE,
                    WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN,
                    WorkbenchPromptTemplate.TRANSLATE_TO_ENGLISH);
        }

        @Override
        protected String renderUserPrompt(WorkbenchPromptTemplate template, String inputText) {
            if (template == WorkbenchPromptTemplate.SUMMARIZE_DE) {
                return "Summarize the following text in German:\n" + inputText;
            }
            if (template == WorkbenchPromptTemplate.TRANSLATE_TO_GERMAN) {
                return "Translate the following text to German:\n" + inputText;
            }
            if (template == WorkbenchPromptTemplate.TRANSLATE_TO_ENGLISH) {
                return "Translate the following text to English:\n" + inputText;
            }
            return inputText;
        }
    }

    private static final class CodeT5SmallPromptProfile extends AbstractPromptProfile {
        private static final CodeT5SmallPromptProfile INSTANCE = new CodeT5SmallPromptProfile();

        private CodeT5SmallPromptProfile() {
            super(WorkbenchPromptTemplate.NONE, WorkbenchPromptTemplate.EXPLAIN_CODE_DE);
        }

        @Override
        protected String renderUserPrompt(WorkbenchPromptTemplate template, String inputText) {
            if (template == WorkbenchPromptTemplate.EXPLAIN_CODE_DE) {
                return "explain java: " + inputText;
            }
            return inputText;
        }
    }

    private static final class CodeT5MultiSumPromptProfile extends AbstractPromptProfile {
        private static final CodeT5MultiSumPromptProfile INSTANCE = new CodeT5MultiSumPromptProfile();

        private CodeT5MultiSumPromptProfile() {
            super(WorkbenchPromptTemplate.NONE, WorkbenchPromptTemplate.SUMMARIZE_DE);
        }

        @Override
        protected String renderUserPrompt(WorkbenchPromptTemplate template, String inputText) {
            if (template == WorkbenchPromptTemplate.SUMMARIZE_DE) {
                return "summarize: " + inputText;
            }
            return inputText;
        }
    }
}

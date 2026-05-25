package com.aresstack.windirectml.sidecar;

import com.aresstack.windirectml.sidecar.client.validation.ModelExpectation;
import com.aresstack.windirectml.sidecar.client.validation.ModelValidator;
import com.aresstack.windirectml.sidecar.client.validation.ValidationReport;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

final class ModelValidationCli {
    private ModelValidationCli() {
    }

    static boolean requested(String[] args) {
        if (args != null) {
            for (String arg : args) {
                if ("--validate-models".equals(arg) || "--validate".equals(arg)) return true;
            }
        }
        String property = System.getProperty("model.validate");
        return "true".equalsIgnoreCase(property) || "1".equals(property);
    }

    static int run() {
        List<ValidationReport> reports = new ArrayList<ValidationReport>();
        String family = System.getProperty("embed.model", "minilm").trim().toLowerCase(java.util.Locale.ROOT);
        if (family.contains("e5")) {
            String variant = System.getProperty("e5.model", "base-v2");
            reports.add(validate(path("e5.modelDir"), ModelValidator.e5Expectation(variant)));
        } else {
            reports.add(validate(path("minilm.modelDir"), ModelValidator.minilmExpectation()));
        }
        File reranker = path("rerank.modelDir");
        if (reranker != null) {
            reports.add(validate(reranker, ModelValidator.rerankerExpectation()));
        }
        int errors = 0;
        for (ValidationReport report : reports) {
            System.out.println(report.format());
            System.out.println();
            if (!report.isOk()) errors++;
        }
        System.out.println("model validation reports: " + reports.size() + ", failed: " + errors);
        return errors == 0 ? 0 : 4;
    }

    private static ValidationReport validate(File directory, ModelExpectation expectation) {
        return ModelValidator.validate(directory, expectation);
    }

    private static File path(String property) {
        String value = System.getProperty(property);
        if (value == null || value.trim().isEmpty()) return null;
        return new File(value.trim());
    }
}

package com.aresstack.windirectml.inference.model;

import java.io.PrintStream;
import java.nio.file.Path;

/**
 * Command-line entry point for inspecting PyTorch state-dict checkpoints.
 */
public final class TorchCheckpointInspectTool {
    private TorchCheckpointInspectTool() {
    }

    public static void main(String[] args) {
        int exitCode = run(args, System.out, System.err);
        if (exitCode != 0) {
            System.exit(exitCode);
        }
    }

    static int run(String[] args, PrintStream out, PrintStream err) {
        try {
            if (args.length != 1 || "--help".equals(args[0])) {
                printUsage(out);
                return args.length == 1 && "--help".equals(args[0]) ? 0 : 2;
            }
            TorchCheckpointInspection inspection = TorchCheckpointInspector.inspect(Path.of(args[0]));
            printInspection(out, inspection);
            return inspection.hasMissingStorageEntries() ? 3 : 0;
        } catch (Exception e) {
            err.println("Torch checkpoint inspection failed: " + e.getMessage());
            return 2;
        }
    }

    static void printInspection(PrintStream out, TorchCheckpointInspection inspection) {
        out.println("Torch checkpoint: " + inspection.checkpoint());
        out.println("Archive prefix: " + (inspection.archivePrefix().isBlank() ? "<root>" : inspection.archivePrefix()));
        out.println("Tensors: " + inspection.tensorCount());
        out.println("Declared tensor bytes: " + inspection.declaredTensorBytes());
        out.println("Storage bytes: " + inspection.storageBytes());
        for (TorchCheckpointTensor tensor : inspection.tensors()) {
            out.println("- " + tensor.name()
                    + " dtype=" + tensor.dataType().name()
                    + " shape=" + tensor.shapeText()
                    + " stride=" + tensor.strideText()
                    + " storage=" + tensor.storageKey()
                    + " offset=" + tensor.storageOffset()
                    + " bytes=" + tensor.tensorByteLength()
                    + (tensor.storageEntryPresent() ? "" : " MISSING_STORAGE"));
        }
    }

    private static void printUsage(PrintStream out) {
        out.println("Usage:");
        out.println("  TorchCheckpointInspectTool <pytorch_model.bin>");
    }
}

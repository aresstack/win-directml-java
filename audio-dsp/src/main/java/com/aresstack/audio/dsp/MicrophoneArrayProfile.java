package com.aresstack.audio.dsp;

/**
 * Geometry of a microphone array: the 3-D position of each microphone in millimetres (channel-ordered),
 * plus identity and supported sample rates. Beamforming requires a valid geometry — it must never invent
 * default microphone positions. Parsed from a compact string form so it can also live in a block parameter
 * until a persisted array-profile repository exists.
 */
public final class MicrophoneArrayProfile {

    private final String id;
    private final String name;
    private final double[][] positionsMm; // [channel][x,y,z]

    public MicrophoneArrayProfile(String id, String name, double[][] positionsMm) {
        if (positionsMm == null || positionsMm.length < 2) {
            throw new IllegalArgumentException("A microphone array needs at least two microphone positions.");
        }
        for (double[] p : positionsMm) {
            if (p == null || p.length != 3) {
                throw new IllegalArgumentException("Each microphone position needs three coordinates (x,y,z).");
            }
        }
        this.id = id;
        this.name = name;
        this.positionsMm = deepCopy(positionsMm);
    }

    /**
     * Parse positions from "x0,y0,z0; x1,y1,z1; ..." (millimetres). Returns null when the text is empty or
     * malformed, so callers can reject beamforming instead of guessing a geometry.
     */
    public static MicrophoneArrayProfile parse(String id, String name, String text) {
        if (text == null || text.trim().length() == 0) {
            return null;
        }
        String[] mics = text.split(";");
        if (mics.length < 2) {
            return null;
        }
        double[][] positions = new double[mics.length][3];
        for (int i = 0; i < mics.length; i++) {
            String[] coords = mics[i].trim().split(",");
            if (coords.length != 3) {
                return null;
            }
            for (int j = 0; j < 3; j++) {
                try {
                    positions[i][j] = Double.parseDouble(coords[j].trim());
                } catch (NumberFormatException ex) {
                    return null;
                }
            }
        }
        try {
            return new MicrophoneArrayProfile(id, name, positions);
        } catch (IllegalArgumentException ex) {
            return null;
        }
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public int getMicrophoneCount() {
        return positionsMm.length;
    }

    public double[] position(int channel) {
        return positionsMm[channel].clone();
    }

    private static double[][] deepCopy(double[][] source) {
        double[][] copy = new double[source.length][];
        for (int i = 0; i < source.length; i++) {
            copy[i] = source[i].clone();
        }
        return copy;
    }
}

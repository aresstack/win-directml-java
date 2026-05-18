# Head-Layout-Konvention (MiniLM/BERT-Encoder)

> **Status:** verbindlich seit `feat(layout): head-layout kernel and convention doc`.
>
> Diese Datei beantwortet die einzige Frage, die zwischen einem
> funktionierenden und einem stillschweigend kaputten Attention-Pfad
> entscheidet: **In welcher Reihenfolge liegen die Bytes nach den
> Q/K/V-Projektionen im GPU-Buffer?**

## TL;DR

```
[S, hidden]  ─ DirectMlLinearKernel ─►  [S, hidden]            (Linear bleibt 2-D)
                                          ║   identisch zu
                                          ▼
                                      [S, H, D]                 (sequence-major)
                                          │
                                          ▼  DirectMlHeadLayoutKernel.seqMajorToHeadMajor
                                          ▼
                                      [1, H, S, D]              (head-major)
                                          │
                                          ▼  DirectMlAttentionKernel
                                          ▼
                                      [1, H, S, D]              (head-major, neu beschrieben)
                                          │
                                          ▼  DirectMlHeadLayoutKernel.headMajorToSeqMajor
                                          ▼
                                      [S, H, D]                 (sequence-major)
                                          ║   identisch zu
                                          ▼
                                      [S, hidden]               (Linear-Output-Projection erwartet das wieder)
```

Die beiden Layouts sind **nicht** dieselbe Speicherreihenfolge.
Reinterpretieren ohne den Layout-Kernel produziert „irgendwelche"
Vektoren – Cosine-Similarity gegen die CPU-Referenz fällt typisch auf
`< 0.3` ab, ohne irgendwo zu crashen.

## Symbole und ihre Bedeutung

| Symbol     | Bedeutung                                                                                     |
| ---------- | --------------------------------------------------------------------------------------------- |
| `S`        | Sequenzlänge (Anzahl Tokens nach Tokenizer + Padding)                                         |
| `H`        | Anzahl Attention-Heads (MiniLM-L6-v2: `H = 12`)                                               |
| `D`        | Head-Dimension (MiniLM-L6-v2: `D = 32`)                                                       |
| `hidden`   | Modell-Hidden-Size; per Vertrag `hidden = H * D` (MiniLM-L6-v2: `384 = 12 * 32`)              |
| `B`        | Batch-Größe; für den Encoder-Pfad aktuell immer `1` (eine Anfrage = eine Sequenz)             |

## „Sequence-major" Layout `[S, H, D]`

* Physischer Offset eines Elements: `idx(s, h, d) = s · H · D + h · D + d`
* Aliasiert bit-identisch mit dem 2-D-Layout `[S, hidden]`:
  Spalte `j ∈ [0, hidden)` zerfällt in `h = j / D` und `d = j % D`.
* Direkter Output von `DirectMlLinearKernel` für die Q/K/V- und
  Output-Projektion.
* Direkter Input des Residual-`ADD` und der `LayerNorm` (beide
  operieren pro Zeile = pro Token, völlig unabhängig vom Head-Split).

## „Head-major" Layout `[B, H, S, D]`

* Physischer Offset: `idx(b, h, s, d) = ((b · H + h) · S + s) · D + d`
* Erwartete Eingangsform für `DirectMlAttentionKernel`. Der Grund:
  `DML_OPERATOR_GEMM` interpretiert bei 4-D-Tensoren die beiden
  führenden Dimensionen als unabhängige Batch-Achsen. Damit wird
  `Q · Kᵀ` und `probs · V` in einem einzigen Dispatch über alle
  `(batch, head)`-Kombinationen ausgeführt – ohne diese Umordnung
  wäre pro `(b, h)` ein separater GEMM-Dispatch nötig.
* `B = 1` für den Encoder-Pfad: der äußere Stride ist nominell
  `H · S · D`, in der Praxis irrelevant.

## Konvertierung: `DirectMlHeadLayoutKernel`

Implementiert über `DML_OPERATOR_ELEMENT_WISE_IDENTITY` (Op 1, FL 1.0)
mit nicht-default-Strides auf dem Input-Tensor und Default-Strides
(= contiguous) auf dem Output. DirectML kopiert dann Element-für-Element
in der vom Output diktierten Reihenfolge – materialisiert also die
Permutation.

| Richtung       | Logische Shape | Input-Strides         | Output-Strides                |
| -------------- | -------------- | --------------------- | ----------------------------- |
| `SEQ_TO_HEAD`  | `[1, H, S, D]` | `[S·H·D, D, H·D, 1]`  | `null` (= `[H·S·D, S·D, D, 1]`) |
| `HEAD_TO_SEQ`  | `[1, S, H, D]` | `[S·H·D, D, S·D, 1]`  | `null` (= `[S·H·D, H·D, D, 1]`) |

Die beiden Stride-Vektoren sind exakte Inverse zueinander, formal
verifiziert durch den Unit-Test `roundTripIsIdentity()` mit
Toleranz `0.0f` (Identity-Op preserviert Float-Bits exakt).

## Eigenheiten, die sonst Bugs werden

1. **Linear-Bias.** Die Q/K/V-Linears in MiniLM haben einen Bias der
   Form `[hidden]`. Dieser Bias adressiert **`hidden = H·D`** Elemente
   – also den sequence-major-Indexraum. Er muss **vor** der
   Layout-Konvertierung addiert werden. `DirectMlLinearKernel` macht
   das bereits in einem Schritt; danach erst kommt der Layout-Kernel.
2. **Output-Projection.** Genauso: erst Layout-Konvertierung von
   `[1, H, S, D]` zurück nach `[S, hidden]`, dann erst die `Wo`-Linear-
   Projektion mit ihrem Bias `[hidden]`.
3. **Residual-Connection.** Findet im sequence-major-Raum
   `[S, hidden]` statt. Das heißt: Save eine Kopie der Input-Activations
   vor der Attention (oder vor dem FFN), wende den Block an, addiere
   die Kopie auf den Output. `LayerNorm` läuft auf der summierten
   Aktivierung.
4. **Padding-Mask.** Form `[B, S]` (additiv, `-1e9f` an Padding-
   Positionen, `0` sonst). Die Maske ist Layout-agnostisch: sie wird
   im `DirectMlAttentionKernel` per Strides `[S, 0, 0, 1]` von
   `[B, 1, 1, S]` auf `[B, H, S, S]` ausgedehnt. Sie wird **nicht**
   durch den `DirectMlHeadLayoutKernel` geschleust.
5. **GELU im FFN-Pfad.** GELU operiert elementweise und ist
   Layout-agnostisch. Reihenfolge im Encoder-Block:
   `LayerNorm → Linear (hidden → 4·hidden) → GELU → Linear (4·hidden → hidden) → Residual`.
   Das gesamte FFN bleibt sequence-major.

## Wo der Konverter eingesetzt wird

```
Input [S, hidden]
  ├──► (Q-Linear, B=hidden) ──► [S, H, D] ─SEQ_TO_HEAD─► [1, H, S, D] ─┐
  ├──► (K-Linear, B=hidden) ──► [S, H, D] ─SEQ_TO_HEAD─► [1, H, S, D] ─┼─► Attention ─► [1, H, S, D]
  └──► (V-Linear, B=hidden) ──► [S, H, D] ─SEQ_TO_HEAD─► [1, H, S, D] ─┘                  │
                                                                                          ▼
                                              [S, hidden] ◄─ (Wo-Linear) ◄─ [S, H, D] ◄─HEAD_TO_SEQ
```

Pro MiniLM-Layer also: **3× `SEQ_TO_HEAD`** vor Attention, **1× `HEAD_TO_SEQ`**
nach Attention. Die Kernel sind formgebunden – ein Kernel pro Richtung
genügt für den gesamten 6-Layer-Encoder.


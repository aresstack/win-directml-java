# Problems / Blockers — Engine Cleanup Plan abarbeitung

Dieser Log sammelt Punkte aus `doc/todo.md`, die nicht (vollständig) erledigt werden konnten,
mit Grund und Lösungsalternativen. Erledigte Punkte stehen kurz mit Commit-Hash; offene mit Begründung.

Reihenfolge entspricht `doc/todo.md`.

---

## Item 3 — Fused Projection heap-light: T5-Fusion noch float[] (Teil offen)

**Stand:** Der **decoder-only/SmolLM2**-Teil ist erledigt (Commit folgt): SmolLM2 `qkv` und `gate_up` nutzen jetzt
`DecoderOnlyWarpFusedDenseProjection.fromWeightSourceParts` → bei all-FP32-ByteBuffer-Teilen einen
**Device-Region-Stack** (`MatMulNBitsKernel.fromFusedFp32ByteBuffers`, mehrere `CopyBufferRegion` in einen Ziel-Buffer,
kein Java-Heap-`float[]`-Fused-Array). Byte-identisch zum float[]-Concat verifiziert + SmolLM2-Native grün.

**Offen:** Die **T5-Fusion** (`T5FusedSelfAttentionProjection` / `T5FusedCrossAttentionMemoryProjection`) baut weiterhin
ein Host-`float[]` und verpackt es in `T5TensorData.reference(...)`; danach baut `T5WarpLinearProjectionFactory` daraus
eine Einzel-Projektion. Dadurch greift der ByteBuffer-Vorteil für T5-fused (Q/K/V, Cross-Attn K/V) noch nicht.

**Warum offen gelassen:** Die T5-Fusion ist über `T5WarpLinearProjectionFactory.createSelfAttentionProjection` /
`createCrossAttentionMemoryProjection` mit den T5-Attention-Projektionstypen verzahnt und T5 wird in den Items 6/7
ohnehin überarbeitet. Eine Umstellung jetzt würde die T5-Attention-Verdrahtung anfassen, bevor der T5-Engine-/
Status-Umbau (Item 6/7) entschieden ist — Risiko ohne Mehrwert.

**Lösungsalternative:** Im Zuge von Item 7 (T5-Engine) `T5WarpLinearProjectionFactory` so umbauen, dass die fused
Self-/Cross-Attn-Projektionen eine `List<WarpWeightSource>` aus den zugrunde liegenden `T5TensorData`-FP32-Quellen
bilden und `WarpDenseProjection.fromFusedFp32(...)` nutzen (statt `T5TensorData.reference(float[])`). Die Infrastruktur
(`fromFusedFp32`, `fromWeightSourceParts`) steht bereits und ist getestet.

---

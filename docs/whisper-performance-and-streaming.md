# Whisper: SIMD performance and streaming transcription

Status: plan for the next steps after the guarded decoder of 0.3.2.
Order of work: **1. SIMD → 2. streaming after stop (A) → 3. transcription while recording (B)**.
**C (live preview of the unfinished window) is optional and not planned.**

## Starting point (0.3.2)

- Every entry point uses the guarded decoder: 30-second windows padded to 3000 log-Mel frames,
  timestamp decoding, temperature fallback, compression-ratio, log-probability and no-speech checks,
  segment-based window stitching.
- `AskAiWhisperPipeline` applies DC removal, an 80-Hz high-pass and a conservative VAD that only skips
  silence of about eight seconds, with a whole-recording fallback when it detects nothing.
- Batched encoder operations (projection, layer norm, feed-forward, attention, convolution) and the
  decoder's vocabulary projection run in parallel over rows on the common fork-join pool.
- All arithmetic is scalar with `double` accumulation. `jdk.incubator.vector` is not used by Whisper.

Measured with Whisper Tiny on a 24-core CPU (default pipeline):

| Recording | Audio | CPU time | Real-time factor |
|---|---|---|---|
| Single word (synthetic) | 3.3 s | 1.1 s | 0.34 |
| Continuous speech (synthetic) | 64.6 s | 7.1 s | 0.11 |
| Real dictation, fairy tale | 117 s | 16–17 s | 0.14 |
| Real dictation, five minutes | 305 s | 37 s | 0.12 |

One 30-second window costs about 1–2 s; before row parallelization it cost about 14 s.

## 1. SIMD for the Whisper math

### Goal

Use the existing optional-SIMD infrastructure (`simd.FloatMathOps`, `SimdMath.provider()`) in the
Whisper hot paths, so machines with fewer cores also transcribe faster than real time.

### Scope

| Hot path | Change |
|---|---|
| `WhisperMath.project` (all linear layers) | Row dot product via `FloatMathOps.dot` |
| `WhisperTextDecoder` vocabulary projection | `FloatMathOps.dot` per vocabulary row |
| `WhisperMath.attention` | Scores via `dot`, weighted values via `axpy` |
| `WhisperMath.convolution` | Kernel transposed once to `[channel][tap][input]`; three contiguous `dot`s per output |

The single-token decoder path (Tiny: 384-wide vectors) stays sequential; it is too small for threads,
but benefits from SIMD in the same functions.

### Constraints

- **Optional module:** `SimdMath` loads the Vector API reflectively. Without
  `--add-modules jdk.incubator.vector` the scalar `ScalarFloatMathOps` is used automatically, so
  consumers such as Space Harvester keep working; they gain speed by adding the flag.
- **Numerics:** SIMD accumulates in `float` with FMA instead of `double`. Results change in the last
  bits. Acceptance is therefore by transcription quality, not bit equality.

### Acceptance

- `WhisperDictationRegressionTest` passes unchanged: word error rate ≤ 0.15, no repetition loop,
  empty output for silence and noise, quiet-word and language cases.
- The real recordings keep their word error rate within ±0.01.
- CPU time per window is measured before and after, with and without the incubator module.

### Result (implemented)

The Whisper hot paths use `SimdMath.provider()`. Measuring them exposed a library-wide defect:
`VectorFloatMathOps` kept its `VectorSpecies` in an instance field, so C2 never compiled the Vector API
to SIMD instructions and every call took the allocating fallback, about five times slower than scalar
code. The species is now a `static final` constant; this also speeds up the Phi-3, Qwen and
decoder-only CPU paths that share the provider.

Whisper Tiny, same audio, word error rate unchanged (0.071 and 0.321):

| Measurement | Scalar | SIMD | Speed-up |
|---|---|---|---|
| 64.6 s speech, all 24 cores | 7.4 s | 3.5 s | 2.1× |
| 64.6 s speech, 4 cores | 16.1 s | 6.1 s | 2.6× |
| 117 s real dictation | 14.5 s | 7.0 s | 2.1× |

`-Dwindirectml.whisper.simd=false` forces the scalar path for comparison.

## 2. A: streaming after stop

### Goal

After the user stops recording, show the transcript window by window (about every 3 s) instead of all
at once at the end.

### Design

- The library already reports every committed window through
  `AskAiWhisperPipeline.transcribe(source, transcriber, Consumer<TranscriptionResult>)`.
- Space Harvester forwards each committed window to the composer on the event thread and keeps
  "Erkenne Sprache …" visible until the last window.
- Cancel (✕) stops at the next window and keeps the text recognized so far.
- No library change is required.

### Acceptance

- Text appears in the composer while later windows are still being decoded.
- The final text equals the text of a non-streaming run of the same recording.
- Cancel keeps the committed text and leaves no running decoder thread.

## 3. B: transcription while recording

### Goal

Decode completed 30-second windows while the user is still speaking, so after stop only the remaining
audio has to be processed (a few seconds instead of up to a minute).

### Library: incremental transcription API

A new public entry point in `directml-inference`, for example `WhisperStreamingTranscription`:

- `append`-side: the caller supplies a growing `WhisperAudioSource` (or notifies the API about the
  current sample count).
- `decodeAvailable()`: while at least one full 30-second window lies beyond the committed position,
  decode it as a **non-final** window, commit only closed segments and move the committed position to
  the end of the last committed segment (the same rules as `WhisperTimestampWindowStitcher`).
- `finish()`: decode the remaining audio with final-window semantics and return the complete result.
- Callbacks report every committed window with absolute timestamps; cancellation interrupts between
  windows.

The window rules, safeguards and options are identical to the batch path, so batch and incremental
transcription of the same audio produce the same text.

Speech regions: the energy VAD needs the whole recording for its noise-floor calibration and region
merging. While recording, the incremental path decodes the audio without VAD skipping; the guarded
decoder handles pauses inside a window (measured: noisy pauses of 6 and 8 s, noise only → empty text).
Skipping long silence while recording can be added later with a causal VAD.

### Space Harvester

- `ChunkedPcmAudioSource` allows reads of completed audio while recording continues. Reads and appends
  are synchronized; spilled chunks are immutable, so only the in-memory chunks need locking.
- A single background worker runs `decodeAvailable()` whenever the recording has grown by a window.
- Recording never waits for the decoder. If the CPU is slower than real time, the backlog grows and is
  processed after stop; memory stays bounded by the existing two-minute chunk spilling.
- Committed windows go to the composer immediately (A); stop triggers `finish()`.

### Acceptance

- For a recording of several minutes the committed text during recording plus the text after stop
  equals the batch transcript of the same recording (same windows, same safeguards).
- After stop only the last, incomplete window is decoded.
- Recording continues without gaps while the decoder runs; the long-recording probe (12 numbered
  paragraphs across two chunk spills, 16 kHz and 48 kHz) finds every paragraph.
- Cancel during recording discards audio and pending decoding; no thread keeps running.

## 4. C (optional, not planned): live preview of the unfinished window

Re-decode the current, still growing window every 3–5 s and show it as a grey preview that is replaced
by the committed text. This needs a full window decode every few seconds (1–2 s CPU each on 24 cores,
much more on laptops) and the preview flickers when Whisper revises earlier words. Because B already
shows text paragraph by paragraph during recording, C is documented as an optional feature and is not
implemented now.

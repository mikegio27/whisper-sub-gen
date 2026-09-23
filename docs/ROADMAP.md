# Subtitle quality roadmap

Living tracker for the timing/accuracy overhaul started 2026-09-22. Update the status
boxes, the decision log and the results table in the same change as the code.

## Why

A generated sub (Harry Potter and the Order of the Phoenix, sha-2fae320, large-v3-turbo on CPU)
measured against the [subtitling.net timing standards](https://subtitling.net/standards/subtitle-timing):

| Problem | Count (of 1,430 cues) | Standard |
|---|---|---|
| Cue starts exactly when the previous ends | 841 | ~2-frame gap |
| On screen > 7 s | 135 (96 > 10 s, worst 6 min 12 s) | 7 s max |
| On screen < 0.83 s | 240 | 5/6 s min |
| Duration exactly 1.000/2.000/3.000 s | 610 | (quantised timestamps, not measured) |
| Reading speed > 20 cps | 473 | 17 target, 20 max |
| One line > 42 chars, no break | 249 | 2 x 42 |
| Same text twice in a row | 24 | |

Root causes, all in the old `transcriber.py`:

1. Whisper segment timestamps are predicted tokens, not measurements. faster-whisper chains each
   segment's start to the previous one's end.
2. VAD glues speech chunks together, so music and silence get absorbed into whichever cue is open
   (this causes the multi-minute cues).
3. Raw segments were written verbatim: no segmentation, line breaking or reading-speed rules.
4. `condition_on_previous_text=True` (the default) lets one hallucination repeat.
5. The 5.1 mix was downmixed to mono, so music and SFX compete with the dialogue, which sits
   mostly in the center channel.

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-09-22 | Run on the GPU (CUDA image, one time-slice of the RTX 4070 Super) | Owner OK. Alignment and the LLM pass need it. As of 2026-09-22, 3 of 4 slices are used (immich-ml, immich-server, jellyfin) |
| 2026-09-22 | A second model for timing (CTC forced aligner) instead of trusting Whisper's timestamps | Whisper timestamps can't be fixed by tuning, only by alignment |
| 2026-09-22 | Word corrections come from a **local** LLM on the 4070 only, with no paid APIs | Owner: subs are free, and this is a convenience |
| 2026-09-22 | No OpenSubtitles fetching | Owner already downloads human subs. This service is only the automated fallback |
| 2026-09-22 | Our own cue composer (`app/cues.py`) applies the standards to word timings | Pure and unit-testable. Doesn't depend on the ASR/aligner |
| 2026-09-22 | Evaluate against human subs already in the library | Owner has many. Timing and WER are measured, not eyeballed |
| 2026-09-22 | **VAD off by default** | On a 4 min OotP clip Silero cut 2:45 of 4:03 as non-speech, lost whole lines ("Come on, Dudley, let's go. What's going on?") and smeared word times across chunk boundaries ("What are you doing?" spread over 9 s). With VAD off, the text was right ("Don't put away your wand", "Is she dead, Potter?") and the words were coherent. VAD only buys speed; the GPU covers that |
| 2026-09-22 | P2 aligner: `ctc-forced-aligner` (MahmoudAshraf97, **from git, not PyPI**: the PyPI name is a different ONNX fork) + MMS-300m-1130 | CTC aligners are within a few ms of each other on FA-Bench (~46 ms clean / ~57 ms noisy word MAE); MMS covers ~1,130 languages; `<star>` token absorbs untranscribed speech. WhisperX pins torch~=2.8 + pyannote; NeMo too heavy; MFA needs Kaldi. Weights are CC-BY-NC (private use OK) |
| 2026-09-22 | P3 LLM: Ollama as its own homelab Deployment, `qwen3.5:4b` Q4_K_M (~3.4 GB), `keep_alive: 0` | Schema-constrained JSON (`format`), load/unload on demand, and no CUDA build of llama-cpp-python to maintain. The LLM proposes **edits to flagged words only**; code accepts one only if it's phonetically close or an exact cast/character name. Fails open |
| 2026-09-22 | Stages run one after another and free VRAM in between | A 12 GB card shared with Jellyfin NVENC + immich: peak ~4 GB sequential vs ~9–12.6 GB if everything stays loaded |

## Architecture (target)

```
video ─► audio.py      ffmpeg → 16 kHz mono, center channel (FC) when the track is 5.1/7.1
      ─► transcriber   faster-whisper, word_timestamps, no text conditioning, hallucination guards
      ─► align.py      [P2] CTC forced alignment of the words → accurate word start/end
      ─► correct.py    [P3] local LLM fixes low-confidence words (text only, timings untouched)
      ─► cues.py       words → cues per standards.CueRules (segmentation, line breaks, cps, gaps)
      ─► srt.py        render → .srt.part → atomic rename
qa.py        scores any cue list against CueRules (existing library subs included)
evaluate.py  hyp vs reference (human) subs: onset error, WER; `python -m app.evaluate`
```

Pure modules (no model/ffmpeg/settings): `srt`, `standards`, `cues`, `qa`, `evaluate` (the
metric part). Keep them that way so the tests stay fast and GPU-free.

## Phases

### Phase 0: Measure
- [x] `srt.py` (Cue, format/parse, lenient for human subs) and `standards.py` (CueRules)
- [x] `qa.py`: per-file standards score + `python -m app.qa FILE.srt [--json]`.
      `violations_per_100` = too_short + too_long + cps_over_max + touching_or_overlap +
      line_too_long + too_many_lines + consecutive_duplicate_text, per 100 cues
- [x] `evaluate.py`: `python -m app.evaluate HYP.srt REF.srt [--json]`. Onset/offset error after
      removing the global (release) offset, % within 100/250/500 ms, drift, `wer_approx`.
      Caveats: onset n covers only cues both subs start on the same word (read n with the %),
      and human subs paraphrase, so WER vs a human ref has a floor well above 0
- [ ] Eval set: The Big Lebowski, Fargo, Snatch, The Rock, Interstellar (human sidecars;
      talky / accents / fast speech / action / score-heavy). Run locally on the RTX 5090 against
      a read-only NFS mount of the library (this host is in the export's allow list)
- [ ] Baseline numbers for the old pipeline recorded below

### Phase 1: Cheap wins (no new model)
- [x] `audio.py`: dialogue-track pick (skips commentary), center channel for 5.1/7.1 with a
      silent-center fallback, one in-memory decode
- [x] faster-whisper: `word_timestamps=True`, `condition_on_previous_text=False`,
      `hallucination_silence_threshold=2.0`, **VAD off** (see decisions)
- [x] `cues.py` composer wired into the transcriber (`PIPELINE_VERSION = 1`)
- [x] QA score stored per file in sqlite (`qa_violations`, `qa_json`) +
      `subgen_qa_violations_per_100_cues` histogram
- [x] Pipeline version in state + opt-in `REGENERATE_OUTDATED` (own output only: size+mtime
      fingerprint, ctime for the 1,847 pre-v1 rows, never when another sub sits next to it)
- [x] Independent review of the wiring. It found and fixed a real hazard: a human sub
      downloaded *during* a job could be overwritten (same name) or deleted (language change).
      Output is now placed with `os.link` (atomic no-clobber). Anything already at the path is
      replaced only after a last-moment ownership re-check (`tests/test_regen.py`). Failed regens
      keep their row (`regen_attempts`, capped at `MAX_RETRIES`), and `/process force` upgrades
      a queued regen job instead of being swallowed
- [ ] Verify on the NAS (ZFS over NFS) that `mv`-ing a file over another updates ctime. The
      pre-v1 ownership check relies on it
- [x] GPU-safety guard (from the homelab review): on CUDA the model is loaded and smoke-tested
      (1 s of silence end to end, since cuBLAS loads lazily) at startup. A failure sets
      `/healthz` to 503, so the pod restarts, and jobs are dropped without a state row, so a
      driver mismatch can't mark the whole library `failed`
- [ ] Ship, in order:
      1. commit whisper-sub-gen (owner) → CI builds `sha-<X>` + `sha-<X>-cuda`
      2. homelab: the prepared GPU switch, with the image set to `sha-<X>-cuda` (not the
         `sha-2fae320-cuda` placeholder in the working tree) → push → watch for
         "model loaded and smoke-tested" in the logs (that doubles as the driver-550 canary)
      3. eval on the dev box; record v0 vs v1 below
      4. `REGENERATE_OUTDATED=true` in the ConfigMap (1,847 existing outputs)

### Phase 2: Forced alignment (GPU)
- [x] Choose the aligner: ctc-forced-aligner @ `64293cc` + MMS-300m-1130 (see decisions)
- [ ] **Driver check.** The k3s host runs NVIDIA **550.163.01 (CUDA 12.4)**. CTranslate2 >= 4.6.3
      wheels are CUDA 12.8 builds, and the recommended torch is `2.13.0+cu129`. Both should run on
      550 via CUDA minor-version compatibility (sm_89 SASS, no PTX JIT), but that's unverified.
      Canary: the P1 CUDA image in prod first. Fallbacks: upgrade the host driver to >= 575, or a
      cu124/cu126 torch build
- [ ] Prod on the CUDA image + GPU slice (homelab change), `WHISPER_COMPUTE_TYPE=int8_float16`
      (less VRAM on the shared card)
- [ ] `align.py`: align **per Whisper segment window (±0.5 s)**, not the whole film (the Viterbi
      pass isn't windowed). Expand numbers with `num2words` (MMS drops digits). Fall back to Whisper
      word times when a window's alignment score is low (hallucinated text derails alignment).
      Free the model before the next stage
- [ ] Image: torch from the PyTorch index (cu129 for CUDA / cpu otherwise) + `transformers~=5.17`,
      `uroman`, the aligner wheel built `--no-deps` in a builder stage (needs g++). The separate
      `nvidia-cudnn` pip line can go (CTranslate2 >= 4.6.3 is built without cuDNN; torch brings
      its own). Expect roughly +4 GB compressed
- [ ] Dev box note: RTX 5090 (sm_120) — use `int8_float16` (reports of fp16 instability on
      Blackwell), torch cu128/cu129 wheels
- [ ] Optional: vocal separation for action/music-heavy films. Shot-change snapping (ffmpeg `scdet`)
- [ ] Optional: vocal separation for action/music-heavy films. Shot-change snapping (ffmpeg `scdet`)

### Phase 3: Word validation (local LLM)
- [x] Pick the serving option and model: Ollama v0.34.x as `homelab/apps/ollama`, `qwen3.5:4b`
      Q4_K_M (A/B `qwen3.5:9b` / Gemma-4-E4B if the eval shows headroom). Env
      `OLLAMA_MAX_LOADED_MODELS=1 NUM_PARALLEL=1 CONTEXT_LENGTH=8192 FLASH_ATTENTION=1
      KV_CACHE_TYPE=q8_0`; requests `think:false temperature:0`, JSON schema in `format`.
      Needs the nvidia-device-plugin time-slicing raised from 4 to 6 (accounting only)
- [ ] Estimated cost: ~1.5–2.5 min per 2 h film on a 4070S when only flagged windows are sent
- [ ] Flag low-confidence words (whisper prob + aligner score)
- [ ] `correct.py`: LLM edits text only, with Jellyfin/TMDB context (title, cast, character names).
      Must fail open (LLM down → keep the ASR text)
- [ ] Optional: second ASR for disagreement flags: parakeet-tdt-0.6b-v3 via `onnx-asr` on **CPU**
      (onnxruntime is already a faster-whisper dep; no NeMo, no VRAM; ~4 min per film). Text only,
      its timings are worse than the aligner's
- Seen on the OotP clip, P3 targets: "Patrona" (Patronum), "Dumbledore, Austin?" (asked you),
  "Dementors, a little whinging" (in Little Whinging)

## Results

Evaluated with `python -m app.evaluate` on the eval set. Lower is better except "within".

| Pipeline | Median onset err | Within 250 ms | WER | QA violations / 100 cues | Notes |
|---|---|---|---|---|---|
| v0 (sha-2fae320) | | | | 140.1 (OotP full film) | baseline; OotP has no human ref |
| v1 (P1, local CPU) | | | | 5.1 (OotP 4 min clip) | 39 cues; clip = 1:30–5:30 of the film; VAD off |

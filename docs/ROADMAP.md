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
| 2026-09-23 | P2 aligner, revised: **no aligner package.** MMS-300m-1130 via `transformers` + our own numpy Viterbi (per segment window, `<star>` column at the edges). Fallback to whisper times when a segment's mean token log-prob < −5.0 | Nothing to compile, and nothing that breaks on py3.14. The ctc-forced-aligner C++ ext is only the Viterbi, and windows are a few hundred frames: 1.8 s of numpy per film. The threshold is calibrated on 1,850 Lebowski segments (whisper text median −1.5, wrong text median −4.7). It only catches clear failures (0.5%, e.g. a credits hallucination at −7.5) |
| 2026-09-22 | P2 aligner (superseded above): `ctc-forced-aligner` (MahmoudAshraf97, **from git, not PyPI**: the PyPI name is a different ONNX fork) + MMS-300m-1130 | CTC aligners are within a few ms of each other on FA-Bench (~46 ms clean / ~57 ms noisy word MAE); MMS covers ~1,130 languages; `<star>` token absorbs untranscribed speech. WhisperX pins torch~=2.8 + pyannote; NeMo too heavy; MFA needs Kaldi. Weights are CC-BY-NC (private use OK) |
| 2026-09-22 | P3 LLM: Ollama as its own homelab Deployment, `qwen3.5:4b` Q4_K_M (~3.4 GB), `keep_alive: 0` | Schema-constrained JSON (`format`), load/unload on demand, and no CUDA build of llama-cpp-python to maintain. The LLM proposes **edits to flagged words only**; code accepts one only if it's phonetically close or an exact cast/character name. Fails open |
| 2026-09-23 | LLM edits to **context names need closeness too** (lev ≤ 0.6), not a free pass | Live qwen3.5:4b replaced "dead", "What", "We're", "getting", "Let's", "No" with "Dementors" under the free-pass rule |
| 2026-09-23 | LLM correction stays **off** (`LLM_CORRECT=false`) pending the eval set | OotP clip, same guardrails: 4b accepted 1 edit, and it was wrong ("on your own" → "own"); 9b and 27b (5090) proposed 0 acceptable edits. Most proposals echo the word with new casing. The known misses need multi-word or semantic edits ("Austin" → "asked you"), which the closeness test is there to block |
| 2026-09-23 | Hallucination filter (`app/hallucination.py`) instead of VAD | VAD off means Whisper runs over score, where it invents "Thank you.", "PIANO PLAYS", "THE END", "¶¶". no_speech_prob was 0.00 on every segment and the fakes' word probs overlap real lines. Isolation + SDH form do separate them |
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
- [x] Eval set: The Big Lebowski, Fargo, Snatch, The Rock, Interstellar (human sidecars;
      talky / accents / fast speech / action / score-heavy). Run locally on the RTX 5090 against
      a read-only NFS mount of the library (this host is in the export's allow list)
- [x] Baseline numbers for the old pipeline recorded below

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
- [x] Verified on the NAS (ZFS over NFS, 2026-09-23): `mv` of a 2020-mtime file over ours gives
      ctime = now, so the pre-v1 ctime ownership check holds
- [x] GPU-safety guard (from the homelab review): on CUDA the model is loaded and smoke-tested
      (1 s of silence end to end, since cuBLAS loads lazily) at startup. A failure sets
      `/healthz` to 503, so the pod restarts, and jobs are dropped without a state row, so a
      driver mismatch can't mark the whole library `failed`
- [x] Ship, in order:
      1. commit whisper-sub-gen (owner) → CI builds `sha-<X>` + `sha-<X>-cuda`
      2. homelab: the prepared GPU switch, with the image set to `sha-<X>-cuda` (not the
         `sha-2fae320-cuda` placeholder in the working tree) → push → watch for
         "model loaded and smoke-tested" in the logs (that doubles as the driver-550 canary)
      3. eval on the dev box; record v0 vs v1 below
      4. `REGENERATE_OUTDATED=true` in the ConfigMap (1,847 existing outputs)

### Phase 2: Forced alignment (GPU)
- [x] Choose the aligner: ctc-forced-aligner @ `64293cc` + MMS-300m-1130 (see decisions)
- [x] **Driver check.** (verified 2026-09-23: CTranslate2 12.8 and torch cu129 both run on 550) The k3s host runs NVIDIA **550.163.01 (CUDA 12.4)**. CTranslate2 >= 4.6.3
      wheels are CUDA 12.8 builds, and the recommended torch is `2.13.0+cu129`. Both should run on
      550 via CUDA minor-version compatibility (sm_89 SASS, no PTX JIT), but that's unverified.
      Canary: the P1 CUDA image in prod first. Fallbacks: upgrade the host driver to >= 575, or a
      cu124/cu126 torch build
- [x] Prod on the CUDA image + GPU slice (homelab change), `WHISPER_COMPUTE_TYPE=int8_float16`
      (less VRAM on the shared card)
- [x] `align.py`: per-segment windows (±0.5 s, ≤0.25 s into a neighbour), num2words for digits,
      uroman for non-Latin scripts, per-segment fallback, `free()` after each job
      (`ALIGN_FREE_AFTER_JOB`). OotP clip: 45/45 segments aligned, 2.1 s, ~2 GB VRAM peak (torch),
      ~4 GB process total
- [x] Composer retune for aligned times: `CueRules.min_linger = 0.7` (the cue stays ≥ 0.7 s after
      speech unless the next cue needs the room). Median |end error| was 422–571 ms with no linger,
      277–467 ms at 0.4, 234–420 ms at 0.7. Onsets unchanged
- [x] Pre-ship review fixes (2026-09-23):
      - num2words limited to 20 verified languages + digit cap, and all errors caught (it hung forever
        on Amharic 7-digit numbers, inside the aligner lock)
      - hallucination drops gated on confidence (were dropping "OK. OK.", "HELP! HELP!", and
        confident isolated lines)
      - failed or mostly-fallback alignment records pipeline 1, not 2, so regen redoes it later; new
        metrics `subgen_align_segments_total{result}`, `subgen_align_errors_total`,
        `subgen_hallucinations_dropped_total`
      - `free()` also clears the cuBLAS workspace (~0.4 GB was left reserved)
      - the LLM stage is wrapped fail-open
- [x] Image: torch 2.13.0 from the PyTorch index (cu129 / cpu) **before** requirements.txt, plus
      `transformers~=5.17`, `uroman`, `num2words` in requirements. No builder stage. CI frees
      runner disk for the CUDA leg
- [x] Dev box note: RTX 5090 (sm_120) — use `int8_float16` (reports of fp16 instability on
      Blackwell), torch cu128/cu129 wheels
- [x] Vocal separation (rejected, see Polish) and shot-change snapping (shipped in v4)

### Ops findings after v2 went live (2026-09-23)
- [x] **OOMKilled on long films.** With VAD off, faster-whisper builds the mel spectrogram for the
      whole input in one pass (~3.3 GB per hour, measured). The Godfather Part II (3 h 18 min) died
      at 12Gi, and an OOM kill writes no state row, so it would be retried after every restart.
      Hotfix: limit 24Gi. Fix: `ASR_CHUNK_S=1200`, ~20 min chunks cut at the quietest 0.5 s near
      each mark, language fixed from the first chunk. Godfather II peak RSS is now 4.3 GB, 78x
      realtime, with clean cuts (checked cues at every boundary)
- [x] **Whisper's no-punctuation mode** (`app/punctuation.py`, `PUNCT_REPAIR`): stretches came out
      all lowercase with no punctuation ("i'm gonna leave here tonight"), 0.5–2.5% of cues per film.
      Detected segments (≥ 4 words, no capitals or punctuation, adjacent ones merged up to 28 s) are
      re-decoded with a punctuated `initial_prompt`. The result is kept only if it has the same words
      (≥ 0.8 similarity) and is now punctuated. Edge words borrowed from neighbours are trimmed.
      Eval: lowercase-"i" cues went from 8–43 per film to 0–1, with onset and s+d unchanged. Cost +6%
      runtime (The Rock: 33/34 stretches repaired, 1 rejected as different words). **Pipeline v3**:
      regen redoes the v2 files too
- [x] **RSS growth across jobs** (2026-09-24): prod climbed ~3.5 → 11 GB over ~180 sequential jobs
      and was OOMKilled mid-job (3 restarts in 12 h). It wasn't reproducible locally (9 varied films
      on a worker thread: ~3.2 GB, or ~2.2 GB with `malloc_trim`), which points at glibc heap
      fragmentation on the image's glibc. Fixed in layers (`app/memory.py`):
      - `gc` + `malloc_trim(0)` after every job
      - `MALLOC_ARENA_MAX=2` in the image
      - `RECYCLE_MEMORY_FRACTION=0.7`: when RSS ends a job above 70% of the limit, a clean SIGTERM
        restart happens between jobs
      - metric `subgen_memory_recycles_total`
      To do: confirm the RSS slope in prod via Mimir after deploy

### Polish (2026-09-24)
- [x] **Reading speed: already at human parity, no change.** Of our cues over 20 cps, 74% are
      inherently fast (can't fit even using all the time up to the next line's speech), 22% could be
      merged, and 4% have time left over. More telling, the human subs on the same films have *more*
      fast cues: 9–21% vs our 6–13%, with the same median cps (13–15). Humans condense dialogue,
      which verbatim subs can't. A lead-in (starting before speech) was rejected: it would trade real
      onset accuracy for this metric
- [x] **`min_linger` 0.7 → 1.0**: better on every film. Median |end error| went Lebowski 249→233,
      Fargo 420→387, Snatch 239→175, Rock 345→313, Interstellar 238→157 ms, and the signed bias is now
      around 0; onsets/QA/cps unchanged. Shipped in v4
- [x] **Shot-change snapping** (`app/shots.py`, `SHOT_SNAP=true`). Cues move onto nearby cuts per the
      Netflix Timed Text Style Guide: a start within 12 frames after / 4 before a cut moves to it; an
      end within 12 frames of cut−2f goes to cut−2f. A snap is skipped if it would break
      gap/duration/cps. Detection: ffmpeg `scdet` at threshold 5, with clustered candidates within
      0.5 s dropped unless the top score is ≥ 2x the runner-up (motion and flashes). It runs in a
      background thread next to the ASR, with NVDEC (`-hwaccel cuda`, `scale_cuda`) at ~100x realtime
      and a CPU fallback (~700–1300 CPU-s per film), always `nice 19`. Cost ≈ +20% per job.
      Eval (v3 base → snapped): ends better on 4/5 (Snatch 241→194 ms, ≤250 ms 52→59%), QA better
      on 4/5, onsets ±7 ms. Onsets look slightly worse under `onset_local` only because it
      subtracts the human lead-in; measured without offset on the films whose refs follow cuts
      (Snatch, Interstellar), every figure improved.
      **Prod needs `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`** for NVDEC (libnvcuvid); without
      it, decoding falls back to the CPU
- [x] **Vocal isolation: rejected** (experiment, 2026-09-24). Mel-RoFormer (Kim) and htdemucs_ft
      (vocals sub-model), fed to the aligner only or to whisper + aligner, on The Rock / Fargo /
      Interstellar: nothing beat run-to-run noise (±16 ms onset, ±0.5 pp s+d). htdemucs_ft into whisper
      was harmful (a whole 20 min Fargo chunk transcribed empty, s+d 36%). Cost: ~3–4.5 extra min per
      2 h film on the 4070S. The premise doesn't hold either: Fargo's center channel is +30 dB speech
      over background yet scores 271 ms, while Interstellar is +10 dB and scores 83 ms. The center
      channel already does most of the work. Scripts and data: session scratchpad `vocals/`
- [x] **Eval set quality**: `scripts/run_eval.py` now accepts `embedded:<stream>` refs. A text sub
      stream inside the video is the *same release*, so there's no cut or framerate mismatch. It is
      extracted into the eval output cache, never committed. 136 library films have a clean English
      text stream. v3 on five of them (2026-09-24):

      | Film | onset (local) | ≤250 ms | s+d | global offset | drift |
      |---|---|---|---|---|---|
      | John Wick | 65 ms | 86.6% | 10.8% | +189 ms | +5 ms |
      | Die Hard | 70 ms | 91.8% | 9.2% | +103 ms | +43 ms |
      | Hot Fuzz | 72 ms | 91.5% | 6.7% | +85 ms | −42 ms |
      | Moon | 59 ms | 89.2% | 9.7% | +8 ms | −14 ms |
      | Hereditary | 104 ms | 83.5% | 8.5% | +225 ms | −81 ms |

      So The Rock (~360 ms) and Fargo (~280 ms) were mostly measuring their references. These five
      join `eval/set.tsv` once the shot-snapping eval finishes (so that eval's set doesn't change
      mid-run)
- [x] **Lead-in**: with same-release refs the *global* offset is meaningful, and ours is +8 to
      +225 ms (median ~+100): our cues appear later than the human ones. Pro subs typically set the
      in-time a frame or two before speech, and CTC onsets tend to be slightly late. Test a small
      `lead_in` (0.1–0.2 s, never into the previous cue's min_gap) against the raw (not local)
      onset error on the embedded-ref films.
      Prototyped as a post-process on the v3 outputs (raw median |onset error|, none → 0.10 → 0.15 s):
      Hereditary 230→170→147, Hot Fuzz 96→75→79, John Wick 194→111→83, Die Hard 111→80→92,
      Moon 57→94→128 ms. Moon's ref sits on speech. QA improved on all five.
      **Decision: `lead_in = 0.1`** (≈2 frames, the pro "a frame or two early" convention; better
      on 4/5, and costs Moon the least). Shipped in v4 (`cues._timed`, never into the previous cue's
      gap)

- [x] **v4 eval** (10 films, 2026-09-24; v3 → v4 = shots + linger 1.0 + lead-in 0.1):
      - Raw onset improved where refs share our timeline: Snatch 286→215, Interstellar 230→166,
        John Wick 194→115, Die Hard 111→92, Hot Fuzz 96→88 ms.
      - Ends improved on 6/10. QA improved on all 10 (e.g. Lebowski 11.1→10.1, The Rock 12.7→11.3).
      - Moon and Hereditary looked 100–180 ms worse in the first v4 run, but clean reruns matched v3
        (Moon 72/73 ms, Hereditary 105 ms). That run was uniformly 0.34 s early on just those two
        films, during heavy GPU contention, and took 720 s, which suggests audio lost in the decode.
        Hence the decode guard below.
- [x] **Decode-shortfall guard** (`audio._decode_checked`): a healthy decode matches the audio
      stream's duration to the millisecond (stream `duration`, or the MKV `DURATION` tag; checked on 5
      films). If the decode is > 0.25 s short, it is decoded again once, and logged as an error if still
      short. ffmpeg can drop unreadable packets (NFS) and still exit 0, and lost audio at the start
      shifts every cue early.
- [x] **LLM calls via dozai** (spec: `dozai/docs/integrations/whisper-sub-gen.md`): `OLLAMA_API_KEY`
      is sent as a Bearer token. `OLLAMA_URL=http://dozai.dozai.svc.cluster.local:8080/ollama/auto`
      picks the 5090 when up, else the 4070, records usage as client `whisper-sub-gen`, and owns model
      loading (our `keep_alive: 0` unload no longer evicts chat users). The `X-Dozai-Backend`
      header is recorded per film (`corrected.backends`). No key = direct Ollama, unchanged.
      `LLM_CORRECT` stays false (Phase 3 decision).

- [x] **v4 live** (sha-c619e9f-cuda, 2026-09-24). The pod runs with NVIDIA_DRIVER_CAPABILITIES incl.
      `video`; shot detection runs on NVDEC in prod (The Net: 1327 cuts, 92 starts / 234 ends
      snapped). dozai path verified from the pod: 200 with the token, 401 without; `dozai client
      list` shows `whisper-sub-gen` LAST USED.
- [x] **Shot detection cost on the 4070S** (watch only): 178 s for a 1 h 54 min film (5090: ~60 s), so the job
      waited 79 s for it: 103 → 178 s per film (+73%, not the +20% seen on the 5090). Tolerable for
      the one-off regen backlog. Watch whether it competes with Jellyfin transcodes for NVDEC. If
      it does, options: skip snapping when detection isn't done by compose time, run detection
      only for new files, or `SHOT_DECODE=cpu` at nice 19.
      Follow-up: on typical content, detection keeps pace. DS9 episodes waited 0.5–5.7 s,
      ~60x realtime overall. Only long 1080p films wait noticeably

### Next (open, 2026-09-24)
- [ ] Confirm the v4 regeneration backlog (~1,840 files) finishes with flat memory
      (`subgen_memory_recycles_total` stays ~0) and doesn't disturb Jellyfin transcodes
- [ ] 162 files never subtitled: `Permission denied` writing into American Dad! / Futurama
      folders on TrueNAS (uid 1000). Owner-side permission fix, then they get picked up by the next scan
- [ ] Owner decision: foreign-language films, `transcribe` (original language, today) or
      `translate` (English). Mixed-language films are also imperfect, because the language is fixed
      from the first chunk
- [ ] Faster pickup: a Jellyfin "item added" webhook calling `/process`, plus a Jellyfin library
      refresh after each write (today it's the 6 h scan + Jellyfin's own scan)
- [ ] `API_KEY` for the service API (SealedSecret). Only matters if something outside the
      cluster ever calls it
- [ ] *(nice-to-have)* Speaker-change dashes need diarization; cosmetic, sizeable

### Phase 3: Word validation (local LLM)
- [x] Pick the serving option and model: Ollama v0.34.x as `homelab/apps/ollama`, `qwen3.5:4b`
      Q4_K_M (A/B `qwen3.5:9b` / Gemma-4-E4B if the eval shows headroom). Env
      `OLLAMA_MAX_LOADED_MODELS=1 NUM_PARALLEL=1 CONTEXT_LENGTH=8192 FLASH_ATTENTION=1
      KV_CACHE_TYPE=q8_0`; requests `think:false temperature:0`, JSON schema in `format`.
      Needs the nvidia-device-plugin time-slicing raised from 4 to 6 (accounting only)
- [x] ~~Estimated cost: ~1.5–2.5 min per 2 h film~~ moot, LLM correction is off on a 4070S when only flagged windows are sent
- [x] Ollama deployed in-cluster (`homelab/apps/ollama`, qwen3.5:4b pulled on start), time-slicing
      4 → 6 (2026-09-23)
- [x] Flag low-confidence words: prob < 0.6 (≈10% of words on the OotP clip), plus mid-sentence
      capitalised non-names and word loops
- [x] `correct.py` + `context.py`: window planning, schema-constrained edits, code-side acceptance
      (flagged index, 1–3 words, lev ≤ 0.5 or same Metaphone, names lev ≤ 0.6), fails open,
      per-film budget. Jellyfin context lookup is written but **untested live**: needs a Jellyfin
      API key as a SealedSecret
- [x] Decided on the eval set: **off**. v2 + qwen3.5:27b (5090) gave s+d 10.5 / 11.7 / 11.1% vs
      10.8 / 11.4 / 11.2% without it (Lebowski / Fargo / Snatch), within run-to-run noise, at 3–4x
      the runtime. The code stays (`LLM_CORRECT`) for a better model or a names-heavy use case
- [ ] *(deferred, low value)* Optional: second ASR for disagreement flags: parakeet-tdt-0.6b-v3 via `onnx-asr` on **CPU**
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
| v1 prod (4070S) | | | | 7.3 (OotP full film) | 85 s for 2 h 18 min (98x realtime), ~1.1 GB VRAM, 2026-09-23 |

**Eval set, 2026-09-23** (`scripts/run_eval.py`, 5090, int8_float16). Onset = median |error| against
the *local* offset (rolling median of ±25 anchors; The Rock's ref is for another cut, drifting 225 s,
and Fargo's ref is incomplete, 609 cues). s+d = substitutions+deletions / ref words, which ignores
insertions because refs are condensed or incomplete. `wer~` is also in results.json.

| Film | v0 onset | v1 onset | v0 s+d | v1 s+d | v0 QA/100 | v1 QA/100 |
|---|---|---|---|---|---|---|
| The Big Lebowski | 429 ms | 230 ms | 14.2% | 11.3% | 136.3 | 8.9 |
| Fargo | 317 ms | 306 ms | 14.2% | 11.5% | 119.7 | 12.4 |
| Snatch | 551 ms | 130 ms | 12.2% | 11.0% | 139.6 | 8.6 |
| The Rock | 522 ms | 386 ms | 27.2% | 11.7% | 167.8 | 10.7 |
| Interstellar | 128 ms | 142 ms | 20.4% | 10.4% | 125.9 | 6.6 |

| Film | v2 onset | ≤250 ms | v2 s+d | v2 QA/100 |
|---|---|---|---|---|
| The Big Lebowski | 163 ms | 62.0% | 10.6% | 10.5 |
| Fargo | 280 ms | 45.1% | 11.0% | 14.3 |
| Snatch | 100 ms | 85.6% | 11.7% | 9.3 |
| The Rock | 358 ms | 35.8% | 11.6% | 12.8 |
| Interstellar | 82 ms | 91.9% | 10.3% | 7.7 |

v2 = v1 + forced alignment + hallucination filter; `min_linger` 0.7 on top leaves onsets as they are
and improves ends (see Phase 2). QA/100 rose from v1 (8.6–12.4) because aligned words expose real
pauses, so there are more, faster cues (`cps_over_max`).

v1 also adds hallucinated lines over score-only stretches ("Thank you.", "PIANO PLAYS", "THE END"
in the Fargo/Interstellar intros): the cost of VAD off. See `app/hallucination.py`.

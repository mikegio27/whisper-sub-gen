# whisper-sub-gen

A FastAPI service plus a background worker. It scans media libraries and writes a sidecar
`<stem><tag>.<lang>.srt` next to any video that has no subtitles, using faster-whisper
(CTranslate2). It runs on the homelab k3s cluster in namespace `whisper-sub-gen` and writes into the
same NFS export that Jellyfin serves (`10.0.10.20:/mnt/tank/k3s/jellyfin-media`, mounted at `/media`).
Jellyfin picks up new .srt files on its next library scan. The manifests are in
`../homelab/apps/whisper-sub-gen/`, not here.

**Subtitle quality work is tracked in `docs/ROADMAP.md`**: phases, decision log, eval results.
Update it in the same change as the code.

## Commands

CI (`.github/workflows/build.yaml`) runs a `test` job (ruff check, ruff format --check, unittest on
Python 3.12) and only builds/pushes the images if it passes. Run the same gates locally before
committing:

```bash
.venv/bin/python -m unittest discover -s tests -v   # stdlib unittest, ~125 tests, <1s
ruff check . && ruff format --check .               # config in pyproject.toml; CI pins ruff==0.16.1
```

`.claude/settings.json` provides hooks that run ruff (fix + format) on each edited Python file and the
unittest suite when a turn stops, so failures surface before you commit.

```bash
# Local venv (.venv, Python 3.14; the image uses 3.12)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Smoke check: importing app.main constructs the Worker, which opens the sqlite db.
# The defaults (/data, /models, /media) aren't writable locally, so override them:
STATE_DIR=./data MODEL_DIR=./models MEDIA_DIRS=/path/to/test-media \
  .venv/bin/python -c "import app.main"

# Run locally (API on :8000). The first job downloads the model (~1.6GB) into MODEL_DIR.
STATE_DIR=./data MODEL_DIR=./models MEDIA_DIRS=/path/to/test-media RUN_MODE=manual \
  .venv/bin/python -m app.main

docker build -t whisper-sub-gen:local .                        # CPU image
docker build --build-arg WITH_CUDA=true -t whisper-sub-gen:cuda .   # adds ~1.5GB of cuBLAS/cuDNN
```

## Tests and lint

- `tests/` is stdlib `unittest` only (no pytest, no dev requirements). It covers `in_work_window`
  (incl. across midnight), `Settings` validation, `StateStore` skip/retry/migration/ownership on a
  temp sqlite, audio track picking, `check_needs_subtitles` against real temp sidecar files
  (ffprobe mocked), `_validate_media_path`, and the pure pipeline modules (`srt`, `cues`, `qa`,
  `evaluate`, with property tests over synthetic word streams). Nothing loads the model, calls
  ffmpeg or needs a GPU; keep it that way.
- Tests override config by `mock.patch.object(settings, "<field>", value)` on the singleton, since
  `settings` is built once at import. `tests/test_api.py` sets `settings.state_dir` to a temp dir
  *before* importing `app.api`, because that import constructs the `Worker` (see gotchas).
- Ruff config lives in `pyproject.toml`: rules `E,F,I,B,UP`, line-length 100, target py312. The
  code is ruff-formatted; the ffmpeg/ffprobe argv lists are fenced with `# fmt: off/on` on purpose.

- `./data` and `./models` are gitignored. Never point `MEDIA_DIRS` at the real library during local
  testing, because the service writes files next to the media.
- Host `ffmpeg`/`ffprobe` must be on PATH for a local run. They are used for probing, duration and
  the audio-extraction fallback.

## Layout

```
app/main.py         FastAPI app; lifespan binds gauges, starts the worker, then stop() + join() on shutdown
app/config.py       pydantic-settings Settings (every field = env var, case-insensitive) + in_work_window()
app/worker.py       Worker singleton: in-memory job queue, 1 "worker" thread, 1 "scanner" thread (continuous mode)
app/scanner.py      find_videos (oldest first, ignore globs, min-age), sidecar/ffprobe skip checks, output_path
app/transcriber.py  pipeline driver: audio -> faster-whisper words -> cues -> qa -> .srt.part -> rename;
                    lazy model load, live Progress, cancel flag, PIPELINE_VERSION
app/audio.py        pick the dialogue track (skip commentary), ffmpeg -> 16 kHz mono float32 in memory,
                    center channel only for 5.1/7.1 (downmix fallback when FC is silent)
app/cues.py         PURE words -> cues composer (segmentation, line breaks, cps, gaps) per CueRules
app/standards.py    CueRules: the subtitle timing/readability limits (subtitling.net / Netflix)
app/srt.py          PURE Cue model, format_ts, render_srt, lenient parse_srt (human subs too)
app/qa.py           PURE standards score of any cue list; CLI `python -m app.qa FILE.srt`
app/evaluate.py     PURE hyp-vs-human-sub timing/WER metrics; CLI `python -m app.evaluate HYP REF`
app/align.py        CTC forced alignment (MMS-300m via transformers + our numpy Viterbi) per whisper
                    segment window; falls back to whisper word times per segment (score < -5.0)
app/hallucination.py PURE drop of non-speech fakes (isolated "Thank you.", SDH caps, music marks)
app/correct.py      local-LLM (Ollama) proofreading of flagged words; code-side acceptance test; off by default
app/context.py      title/year/cast from the path (+ Jellyfin API when JELLYFIN_URL/API_KEY set)
scripts/run_eval.py eval-set runner (eval/set.tsv, human refs on the Jellyfin share); not in the image
app/state.py        sqlite at $STATE_DIR/whisper-sub-gen.db, keyed on path; skip if same size+mtime and
                    done/skipped; pipeline version + fingerprint of the sub we wrote + qa score
app/api.py          /healthz /metrics (no auth); /status /history /scan /process DELETE /history (API_KEY if set)
app/metrics.py      subgen_* Prometheus metrics
```

## Config

The full env var table is in `README.md`. It is sourced from `Settings` in `app/config.py`, so keep
both in sync when adding a field (the table currently lists every field).
`OVERWRITE_EXISTING_OUTPUT=true` only helps together with `SKIP_IF_EXTERNAL_SUBS=false`, because
our own .srt also counts as an external sub. Prod values live in `../homelab/apps/whisper-sub-gen/configmap.yaml`:
`large-v3-turbo`, `WHISPER_DEVICE=cuda` + `int8_float16` on one GPU time-slice (the `-cuda` image),
`REGENERATE_OUTDATED=true`, memory limit 12Gi (safe because of `ASR_CHUNK_S` chunking), `CPU_THREADS=12` (capped so Jellyfin transcodes aren't
starved), `SCAN_INTERVAL_MINUTES=360`, `SHUTDOWN_MODE=abort`, and no `WORK_WINDOW`. There is no
`API_KEY` yet (the secretRef is commented out, waiting on a SealedSecret). There is no ingress: the
API is cluster-internal only (the unused IngressRoute was removed 2026-09-24). In-cluster callers use
`http://whisper-sub-gen.whisper-sub-gen.svc.cluster.local:8000`.

## Release -> deploy loop

1. Push to `main`. After the `test` job passes, `.github/workflows/build.yaml` builds a cpu/cuda
   matrix and pushes `ghcr.io/mikegio27/whisper-sub-gen` with the tags `sha-<short>` (note the
   `sha-` prefix, unlike RoboDoze) and `latest` for the CPU image, and `sha-<short>-cuda` and
   `cuda` for the `WITH_CUDA=true` image. A `v*` git tag also adds `<semver>` / `<semver>-cuda`.
   PRs run tests and build but don't push.
2. In `../homelab/apps/whisper-sub-gen/deployment.yaml`, set
   `image: ghcr.io/mikegio27/whisper-sub-gen:sha-<short>`. Commit and push homelab `main`, and Flux
   reconciles. Pin the sha, because `imagePullPolicy: IfNotPresent`.
3. The deployment uses `Recreate` with 1 replica. The state/model PVC is RWO `local-path`, and the
   worker isn't safe to run twice against the same library.
4. To verify, `kubectl -n whisper-sub-gen logs deploy/whisper-sub-gen`, `GET /status`, and the Grafana
   dashboard `../homelab/apps/grafana/dashboards/whisper-sub-gen.json`.

**GPU variant:** to switch prod to the commented GPU variant in `deployment.yaml`
(`runtimeClassName: nvidia`, `nodeSelector: {gpu: "true"}`, `nvidia.com/gpu: 1` of the 4
time-slices, `WHISPER_DEVICE=cuda` in the ConfigMap), set the image to `sha-<short>-cuda`. The
plain `sha-<short>` image has no cuBLAS/cuDNN, and with `WHISPER_DEVICE=cuda` it fails at model
load.

## Invariants and gotchas

- **Import has side effects.** `worker = Worker()` runs at import time and opens sqlite in
  `STATE_DIR`. Any test or script that imports `app.worker`, `app.api` or `app.main` needs a writable
  `STATE_DIR`. `app.config`, `app.scanner` and `app.state` are safe to import for unit tests.
- The container runs as uid 1000 (the pod `securityContext` matches). The NFS media share and the
  state volume must be writable by uid 1000. `StateStore` raises a hint about exactly this.
- **Output is atomic.** The composed cues are written to `<target>.srt.part` once the ASR finishes,
  which is renamed on success and unlinked on any `BaseException`. A `.part` older than 1h that a hard kill left behind is deleted
  by the next scan. Don't write the final .srt directly.
- **Shutdown semantics (`SHUTDOWN_MODE`).** `abort` sets `transcriber.cancel`. The segment loop
  raises `TranscriptionCancelled`, and no state row is written, so the file is retried after the
  restart. `main` joins for up to 60s, inside the pod's `terminationGracePeriodSeconds: 120`. With
  `finish` it joins with no timeout, so the grace period (SIGKILL) is the only ceiling. Raise it to
  about 3600 in homelab if you switch. Queued jobs are always dropped, and the next scan re-finds
  them.
- The queue is in-memory only. Durable truth is the sqlite `files` table
  (`done|failed|skipped`, `attempts`). Failures stop being retried after `MAX_RETRIES` attempts
  until the file's size or mtime changes, or until `DELETE /history?path=` clears the row.
- The skip logic runs twice, at scan time and again right before processing (subs may have
  appeared meanwhile), unless `force`. `/process` with `force: true` calls `store.forget()` and
  bypasses the work window.
- Audio is always decoded by ffmpeg (`app/audio.py`) into memory (~0.7 GB float32 for 3 h), bounded
  by `EXTRACT_TIMEOUT_S`. No temp files. The same array feeds every stage.
- The image installs **torch from the PyTorch index before requirements.txt** (cu129 for the CUDA
  image, cpu otherwise). Installing it after would let transformers drag PyPI's CUDA torch into
  the CPU image. Tests never import torch. Locally, CUDA runs need the venv's `nvidia/*/lib` dirs
  on `LD_LIBRARY_PATH` (`scripts/run_eval.py` sets that itself).
- Stage order in `transcribe()`: decode → whisper → hallucination filter → align → (LLM) → compose.
  Alignment and the LLM **fail open** (they keep the previous words and log); only decode/ASR errors
  fail a job.
- **VAD is off by default on purpose** (see the comment in `config.py`): on film audio Silero drops
  music-backed dialogue and garbles word timestamps. Don't turn it back on for speed; use the GPU.
- **Never touch a sub we didn't write.** Regeneration (`REGENERATE_OUTDATED`) goes through
  `StateStore.outdated_output`, which only returns our own output while it's byte-for-byte what we
  wrote (size+mtime fingerprint; ctime for pre-v1 rows) and no other sub sits next to the video.
  Bump `transcriber.PIPELINE_VERSION` when output quality changes materially.
- The model loads lazily on the first job, not at startup, so the first job is slow and
  `/healthz` stays green while the model downloads.
- `/process` and `/scan` paths must resolve inside `MEDIA_DIRS` (`_validate_media_path`). Keep that
  check on any new endpoint that takes a path.
- **Metric names are a contract** with the homelab Grafana dashboard (`subgen_*`). Rename them
  there in the same change. Alloy scrapes via the pod annotations on port 8000.
- `requirements.txt` uses `~=` (compatible-release), so rebuilds can pick up new minor versions.
  The local `.venv` already has fastapi 0.139 and uvicorn 0.50.

## Conventions

- Plain stdlib `logging` with `%s` formatting. Thread names appear in the log format (`worker`,
  `scanner`, `api-scan`). Threads are used for concurrency, not asyncio. The API routes are plain
  sync `def` functions, and the worker state they read is guarded by `Worker._lock`.
- Only `main` exists, with no PRs. Early history is short and informal; newer commits use
  conventional `type(scope): summary`. Keep pure `ruff format` churn in its own commit.
- The image tag is `sha-<short>` (CPU) or `sha-<short>-cuda` (GPU).

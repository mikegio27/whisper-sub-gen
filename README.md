# whisper-sub-gen

A containerized app that scans media libraries (e.g. a Jellyfin NFS mount) and
generates sidecar `.srt` subtitles for anything that doesn't have them, using
Whisper. Runs continuously on a schedule, inside a time window, or purely
on-demand via its HTTP API.

## Engine choice

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2)
instead of the reference `openai/whisper` — roughly 4x faster on CPU with int8
quantization at identical accuracy. The default model is **`large-v3-turbo`**:
near-`large-v3` quality at ~6x the speed, which is the sweet spot for CPU-bound
batch transcription. Set `WHISPER_MODEL=large-v3` if you want maximum accuracy
and don't mind the wait, or `WHISPER_DEVICE=cuda` to use a GPU (see below).

With VAD filtering (on by default) silence and music are skipped, so a typical
movie transcribes in a fraction of its runtime on a modern many-core CPU.

## How it works

1. **Scan** — walks `MEDIA_DIRS` for video files (skipping trailers/extras/
   samples and files modified in the last `FILE_MIN_AGE_MINUTES`).
2. **Skip check** — a file is skipped if it already has external sidecar
   subtitles, embedded subtitle streams (ffprobe), or a previous run already
   handled this exact file (tracked in a small sqlite db, keyed on
   path+size+mtime, so nothing is re-probed on every scan).
3. **Transcribe** — faster-whisper decodes the media directly (ffmpeg audio
   extraction as a fallback), auto-detects language unless `LANGUAGE` is set,
   and streams segments into `Movie Name.en.srt` next to the media file
   (written atomically via a `.part` rename). Jellyfin picks the sidecar up on
   its next library scan.

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `MEDIA_DIRS` | `/media` | Comma-separated roots to scan |
| `VIDEO_EXTENSIONS` | `mkv,mp4,...` | Extensions treated as video |
| `IGNORE_PATTERNS` | extras/trailers/samples/trickplay | Comma-separated globs to skip |
| `FILE_MIN_AGE_MINUTES` | `10` | Ignore files newer than this (still importing) |
| `SKIP_IF_EXTERNAL_SUBS` | `true` | Skip when any sidecar sub file exists |
| `SKIP_IF_EMBEDDED_SUBS` | `true` | Skip when the container has subtitle streams |
| `EMBEDDED_TEXT_SUBS_ONLY` | `false` | Only text streams count; bitmap PGS/VobSub won't block generation |
| `MAX_RETRIES` | `2` | Attempts per file before it's parked as failed |
| `WHISPER_MODEL` | `large-v3-turbo` | Any faster-whisper model id (`large-v3`, `medium`, ...) |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `auto` | `auto` = int8 on CPU, float16 on GPU |
| `CPU_THREADS` | `0` | Inference threads; 0 = all cores |
| `BEAM_SIZE` | `5` | Lower (1-2) trades a little accuracy for speed |
| `LANGUAGE` | *(auto)* | Force a language (`en`) instead of per-file detection |
| `TASK` | `transcribe` | Or `translate` (any language → English subs) |
| `VAD_FILTER` | `true` | Skip silence/music — big speedup |
| `SUBTITLE_TAG` | *(empty)* | Extra tag in output name: `<stem><tag>.<lang>.srt` |
| `RUN_MODE` | `continuous` | `continuous` (periodic scans) or `manual` (API-only) |
| `SCAN_INTERVAL_MINUTES` | `60` | Scan cadence in continuous mode |
| `SCAN_ON_STARTUP` | `true` | Scan immediately when the container starts |
| `WORK_WINDOW` | *(empty)* | e.g. `23:00-07:00` — only transcribe inside this window (may cross midnight); the queue pauses outside it |
| `MANUAL_BYPASS_WINDOW` | `true` | API-triggered jobs run immediately regardless of window |
| `API_KEY` | *(empty)* | If set, required as `X-Api-Key` or `Bearer` on all endpoints except `/healthz` and `/metrics` |
| `PORT` | `8000` | API port |
| `STATE_DIR` | `/data` | sqlite db + temp audio |
| `MODEL_DIR` | `/models` | Model download cache (mount a volume!) |
| `WEBHOOK_URL` | *(empty)* | POSTed a JSON summary after every processed file |
| `LOG_LEVEL` | `INFO` | |

## API

| Endpoint | Description |
|---|---|
| `GET /healthz` | Liveness (no auth) |
| `GET /metrics` | Prometheus metrics (no auth) |
| `GET /status` | Current job with live % progress, queue, window state, totals |
| `GET /history?status=done&limit=100` | Processing history from the state db |
| `POST /scan` | Trigger a scan; optional body `{"paths": ["/media/movies"]}` |
| `POST /process` | Queue one file or dir: `{"path": "/media/movies/X.mkv", "force": false}` — `force: true` regenerates |
| `DELETE /history?path=...` | Forget a file so the next scan reconsiders it |

Example — trigger from a Jellyfin post-import script/webhook:

```bash
curl -X POST http://whisper-sub-gen.whisper-sub-gen.svc.cluster.local:8000/process \
  -H 'Content-Type: application/json' \
  -d '{"path": "/media/movies/New Movie (2026)/New Movie (2026).mkv"}'
```

The [Jellyfin Webhook plugin](https://github.com/jellyfin/jellyfin-plugin-webhook)
"Item Added" notification pointed at `POST /scan` also works and needs no
scripting.

## Running standalone

```bash
docker compose up -d          # edit docker-compose.yml volumes first
```

## Deploying to k3s (homelab)

Manifests in [`deploy/k8s/`](deploy/k8s/) follow the homelab conventions:
media PV/PVC against `10.0.10.20:/mnt/tank/k3s/jellyfin-media` (same pattern
as the arr suite), model/state on a `local-path` PVC, config via ConfigMap.

1. Build & push (CI does this on push to main/tags, or use
   `homelab/build-images.sh` with `REGISTRY=ghcr.io/mikegio27`).
2. Copy `deploy/k8s/` to `homelab/apps/whisper-sub-gen/` and add
   `whisper-sub-gen/` to `homelab/apps/kustomization.yaml`; Flux does the rest.
3. Optional: enable `ingressroute.yaml` (Authentik-gated
   `subgen.dozydelta.com`) — remember to add the namespace to the reflector
   allow-list in `cert-manager/certificate.yaml`. In-cluster callers can just
   use the Service directly.
4. If you want an `API_KEY`, create it as a SealedSecret named
   `whisper-sub-gen-secret` and uncomment the `secretRef` in the deployment.

The container runs as uid 1000 and needs write access to the media export to
drop `.srt` files (the NFS export already allows this for the arr apps).

### Using the 4070S instead

CPU is the default and deliberately capped (`CPU_THREADS=12`) so transcodes
aren't starved. If throughput on a big backlog matters, GPU is ~10x faster:

1. Build with `--build-arg WITH_CUDA=true`.
2. In the deployment: set `WHISPER_DEVICE=cuda`, add
   `runtimeClassName: nvidia`, `nodeSelector: {gpu: "true"}` and
   `resources.limits."nvidia.com/gpu": 1` (one of the four time-slices).

A pragmatic pattern: run CPU day-to-day, switch to GPU temporarily for the
initial library backfill.

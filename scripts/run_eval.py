"""Run a pipeline version over the eval set and score it against human subs.

    python scripts/run_eval.py --label v1 --media-root /mnt/jf --out ../eval-out
    python scripts/run_eval.py --label v0 --app-root /path/to/old/checkout ...

Transcription runs in a child process that imports `app` from --app-root, so an
old checkout (e.g. a `git worktree` of an earlier commit) can be measured with
today's metrics. Outputs go to OUT/<label>/ and are reused if present (pass
--redo to regenerate), never next to the media. Needs a GPU for sane runtimes.
Not part of the service image.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.evaluate import compare  # noqa: E402
from app.qa import load_srt, score  # noqa: E402

# Runs inside the child: import the pipeline under test from sys.argv[1] and
# redirect its output path into the eval dir.
_CHILD = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
video, out = Path(sys.argv[2]), Path(sys.argv[3])
import app.transcriber as t
t.output_path = lambda v, lang: out
r = t.Transcriber().transcribe(video)
keep = ("elapsed_s", "language", "aligned", "corrected", "hallucinations_dropped", "punct_repair")
print(json.dumps({k: r[k] for k in keep if k in r}, default=str))
"""


def load_set(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            video, ref = line.split("\t")
            pairs.append((video, ref))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--media-root", type=Path, default=Path("/mnt/jf"))
    ap.add_argument("--out", type=Path, default=REPO.parent / "whisper-eval-out")
    ap.add_argument("--app-root", type=Path, default=REPO)
    ap.add_argument("--set", type=Path, default=REPO / "eval" / "set.tsv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="int8_float16")
    ap.add_argument("--only", help="substring filter on the video path")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    outdir = args.out / args.label
    outdir.mkdir(parents=True, exist_ok=True)
    # CTranslate2 dlopens libcublas.so.12; in a dev venv it comes from the
    # nvidia-* wheels (pulled in by torch), which aren't on the loader path.
    import site

    nvidia_libs = [str(p) for sp in site.getsitepackages() for p in Path(sp).glob("nvidia/*/lib")]
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": ":".join(nvidia_libs + [os.environ.get("LD_LIBRARY_PATH", "")]),
        "STATE_DIR": str(args.out / ".state"),
        "MODEL_DIR": str(REPO / "models"),
        "WHISPER_DEVICE": args.device,
        "WHISPER_COMPUTE_TYPE": args.compute_type,
    }
    results = {}
    for video_rel, ref_rel in load_set(args.set):
        if args.only and args.only not in video_rel:
            continue
        name = Path(video_rel).stem
        hyp_path = outdir / f"{name}.srt"
        meta = {}
        if args.redo or not hyp_path.exists():
            # Our own previous eval output; the pipeline refuses to clobber files.
            hyp_path.unlink(missing_ok=True)
            print(f"[{args.label}] transcribing {name} ...", flush=True)
            proc = subprocess.run(
                [sys.executable, "-c", _CHILD, str(args.app_root),
                 str(args.media_root / video_rel), str(hyp_path)],
                env=env, capture_output=True, text=True, check=False,
            )  # fmt: skip
            if proc.returncode != 0:
                print(proc.stderr[-2000:], file=sys.stderr)
                continue
            meta = json.loads(proc.stdout.strip().splitlines()[-1])
        hyp, ref = load_srt(hyp_path), load_srt(args.media_root / ref_rel)
        c = compare(hyp, ref)
        results[name] = {"meta": meta, "compare": c, "qa": score(hyp)}
        on = c["onset_local"]
        qa_v = results[name]["qa"]["violations_per_100"]
        print(
            f"[{args.label}] {name[:40]:40} onset med {(on['median_abs'] or 0) * 1000:5.0f} ms  "
            f"<=250ms {on['within_250'] or 0:5.1f}%  n {on['n']:4}  "
            f"wer~ {c['wer_approx'] * 100:5.1f}%  s+d {c['sub_del_rate'] * 100:5.1f}%  "
            f"qa {qa_v:5.1f}" + (f"  ({meta['elapsed_s']:.0f}s)" if meta else ""),
            flush=True,
        )
    prev = outdir / "results.json"
    merged = json.loads(prev.read_text()) if prev.exists() else {}
    merged.update(results)
    prev.write_text(json.dumps(merged, indent=1, default=str))


if __name__ == "__main__":
    main()

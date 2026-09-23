"""app.evaluate: alignment-based WER and timing metrics on synthetic hyp/ref pairs."""

from __future__ import annotations

import random
import unittest

from app.evaluate import clean_text, compare, main, normalize_word, tokenize
from app.srt import Cue, parse_srt, render_srt

_VOCAB = (
    "harry ron hermione wand spell castle dark lord owl letter broom quidditch potion "
    "house elf ministry order phoenix prophecy dream scar forest train platform wizard "
    "school friend enemy door window night morning lesson teacher office secret truth"
).split()


def make_ref(n_cues: int = 90, seed: int = 7) -> tuple[list[Cue], list[list[str]]]:
    """Reference cues of 3..8 random words, 2 s long with a 1 s gap. Returns the cues and each
    cue's word list (so a hyp can reuse the exact same words)."""
    rng = random.Random(seed)
    cues, words = [], []
    for i in range(n_cues):
        ws = [rng.choice(_VOCAB) for _ in range(rng.randint(3, 8))]
        start = 10.0 + 3.0 * i
        cues.append(Cue(start, start + 2.0, " ".join(ws).capitalize() + "."))
        words.append(ws)
    return cues, words


def resegment(
    ref: list[Cue], words: list[list[str]], offset: float, jitter: float, seed: int = 3
) -> list[Cue]:
    """The same words with a different segmentation: every 4th ref cue is merged with the next
    one, every 5th is split in two. Cue starts/ends get `offset` plus uniform +-`jitter`."""
    rng = random.Random(seed)

    def j() -> float:
        return offset + rng.uniform(-jitter, jitter)

    out, i = [], 0
    while i < len(ref):
        c, ws = ref[i], words[i]
        if i % 4 == 0 and i + 1 < len(ref):
            nxt = ref[i + 1]
            text = " ".join(ws) + "\n" + " ".join(words[i + 1])
            out.append(Cue(c.start + j(), nxt.end + j(), text))
            i += 2
            continue
        if i % 5 == 0 and len(ws) >= 4:
            half = len(ws) // 2
            mid = c.start + c.duration / 2
            out.append(Cue(c.start + j(), mid + j(), " ".join(ws[:half])))
            out.append(Cue(mid + 0.1 + j(), c.end + j(), " ".join(ws[half:])))
        else:
            out.append(Cue(c.start + j(), c.end + j(), " ".join(ws)))
        i += 1
    return out


class NormalizeTest(unittest.TestCase):
    def test_words(self):
        self.assertEqual(normalize_word("Don't"), ["dont"])
        self.assertEqual(normalize_word("it’s"), ["its"])
        self.assertEqual(normalize_word("21"), ["twenty", "one"])
        self.assertEqual(
            normalize_word("1995"), ["one", "thousand", "nine", "hundred", "ninety", "five"]
        )
        self.assertEqual(normalize_word("Um"), [])
        self.assertEqual(normalize_word("OK"), ["okay"])

    def test_clean_text_strips_sdh(self):
        self.assertEqual(clean_text("[DOOR SLAMS]\nMAN: Get out!"), "Get out!")
        self.assertEqual(clean_text("- Hello.\n- (whispering) Hi."), "Hello. Hi.")
        self.assertEqual(clean_text("♪ Hogwarts, Hogwarts ♪\nSing it"), "Sing it")
        self.assertEqual(clean_text("MRS. WEASLEY: Ron!"), "Ron!")
        # mixed-case "Look:" is speech, not a speaker label
        self.assertEqual(clean_text("Look: over there."), "Look: over there.")
        self.assertEqual(clean_text("It cost 1,000 galleons"), "It cost 1000 galleons")

    def test_tokenize_times(self):
        toks = tokenize([Cue(10, 12, "aaaa bbbb"), Cue(20, 21, "[NOISE]")])
        self.assertEqual([t.word for t in toks], ["aaaa", "bbbb"])
        self.assertEqual(toks[0].time, 10.0)
        self.assertAlmostEqual(toks[1].time, 10 + 2 * 5 / 9)
        self.assertTrue(toks[0].first and not toks[0].last)
        self.assertTrue(toks[1].last and not toks[1].first)


class CompareTest(unittest.TestCase):
    def test_identical(self):
        ref, _ = make_ref()
        r = compare(ref, ref)
        self.assertEqual(r["wer_approx"], 0.0)
        self.assertEqual(r["matched_ratio"], 1.0)
        self.assertEqual(r["global_offset"], 0.0)
        self.assertEqual(r["onset"]["n"], len(ref))
        self.assertEqual(r["onset"]["corrected"]["within_100"], 100.0)

    def test_resegmented_offset_and_jitter(self):
        ref, words = make_ref()
        hyp = resegment(ref, words, offset=1.5, jitter=0.05)
        r = compare(hyp, ref)
        self.assertEqual(r["wer_approx"], 0.0)
        self.assertEqual(r["matched_ratio"], 1.0)
        self.assertAlmostEqual(r["global_offset"], 1.5, delta=0.05)
        on = r["onset"]
        # Merged/split cues don't share a first word, so only part of the cues are measured.
        self.assertGreater(on["n"], len(ref) // 3)
        self.assertLess(on["n"], len(ref))
        self.assertLess(on["corrected"]["median_abs"], 0.06)
        self.assertLessEqual(on["corrected"]["p90_abs"], 0.1)
        self.assertEqual(on["corrected"]["within_100"], 100.0)
        # Without offset removal everything is ~1.5 s off.
        self.assertEqual(on["raw"]["within_500"], 0.0)
        self.assertAlmostEqual(on["raw"]["median_signed"], 1.5, delta=0.05)
        # End errors use the same global offset.
        self.assertGreater(r["offset"]["n"], 0)
        self.assertLess(r["offset"]["corrected"]["median_abs"], 0.06)
        self.assertAlmostEqual(r["drift"]["drift"], 0.0, delta=0.08)

    def test_drift_detected(self):
        ref, words = make_ref()
        # hyp clock runs 25/23.976 fast: offset grows linearly with time
        scale = 25 / 23.976
        hyp = [
            Cue(c.start * scale, c.end * scale, " ".join(w))
            for c, w in zip(ref, words, strict=True)
        ]
        r = compare(hyp, ref)
        self.assertGreater(r["drift"]["drift"], 5.0)

    def test_substitutions_and_deletions(self):
        ref = [Cue(0, 2, "one two three four five"), Cue(3, 5, "six seven eight nine ten")]
        hyp = [Cue(0, 2, "one two three four five"), Cue(3, 5, "six sevens eight ten")]
        r = compare(hyp, ref)
        self.assertEqual(r["substitutions"], 1)
        self.assertEqual(r["deletions"], 1)
        self.assertEqual(r["insertions"], 0)
        self.assertAlmostEqual(r["wer_approx"], 0.2)
        self.assertAlmostEqual(r["matched_ratio"], 0.8)

    def test_insertions(self):
        ref = [Cue(0, 2, "one two three four")]
        hyp = [Cue(0, 2, "one two extra three four")]
        r = compare(hyp, ref)
        self.assertEqual((r["substitutions"], r["deletions"], r["insertions"]), (0, 0, 1))
        self.assertAlmostEqual(r["wer_approx"], 0.25)

    def test_sdh_in_ref_is_ignored(self):
        # Through render/parse, as a real file would be read (that strips the <i> tags).
        ref = parse_srt(
            render_srt(
                [
                    Cue(0, 2, "[DOOR SLAMS]\nHARRY: Where are you going?"),
                    Cue(3, 5, "- I'm off.\n- <i>Wait</i> for me!"),
                    Cue(6, 8, "♪ la la la ♪"),
                    Cue(9, 11, "(sighs) Fine."),
                ]
            )
        )
        hyp = [
            Cue(0, 2, "Where are you going?"),
            Cue(3, 5, "Im off. Wait for me!"),
            Cue(9, 11, "Fine."),
        ]
        r = compare(hyp, ref)
        self.assertEqual(r["wer_approx"], 0.0)
        self.assertEqual(r["global_offset"], 0.0)

    def test_empty(self):
        r = compare([], [])
        self.assertEqual(r["wer_approx"], 0.0)
        self.assertIsNone(r["global_offset"])
        self.assertEqual(r["onset"]["n"], 0)
        self.assertIsNone(r["onset"]["corrected"]["median_abs"])
        self.assertIsNone(r["drift"]["drift"])
        r = compare([], [Cue(0, 1, "hello there")])
        self.assertEqual(r["wer_approx"], 1.0)
        self.assertEqual(r["deletions"], 2)


class CliTest(unittest.TestCase):
    def test_runs_on_files(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path

        ref, words = make_ref(20)
        hyp = resegment(ref, words, offset=0.5, jitter=0.0)
        with tempfile.TemporaryDirectory() as d:
            hp, rp = Path(d) / "hyp.srt", Path(d) / "ref.srt"
            hp.write_text(render_srt(hyp), encoding="utf-8")
            # Human downloads are often cp1252; the loader must not choke on it.
            rp.write_bytes(render_srt(ref).replace("Harry", "Harry’s").encode("cp1252"))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main([str(hp), str(rp), "--json"]), 0)
            data = json.loads(out.getvalue())
            self.assertAlmostEqual(data["compare"]["global_offset"], 0.5, delta=0.01)
            self.assertEqual(set(data["qa"]), {"hyp", "ref"})
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main([str(hp), str(rp)])
            self.assertIn("wer_approx", out.getvalue())
            self.assertIn("violations_per_100", out.getvalue())


class LocalOffsetTest(unittest.TestCase):
    """A reference from a different cut: the offset jumps mid-film."""

    def test_local_residuals_follow_a_recut(self):
        from app.srt import Cue

        ref, hyp = [], []
        for i in range(200):
            t = 10.0 + i * 5
            shift = 1.0 if i < 100 else 61.0  # 60 s scene removed from the ref's cut
            words = f"line number {i} says something different"
            ref.append(Cue(t, t + 2, words))
            hyp.append(Cue(t + shift + (0.05 if i % 2 else -0.05), t + shift + 2, words))
        r = compare(hyp, ref)
        self.assertGreater(r["onset"]["corrected"]["median_abs"], 5)  # global offset fails
        local = r["onset_local"]
        self.assertLess(local["median_abs"], 0.2)
        self.assertGreater(local["within_100"], 90)

    def test_sub_del_rate_ignores_insertions(self):
        from app.srt import Cue

        ref = [Cue(1, 2, "hello there friend")]
        hyp = [Cue(1, 2, "hello there friend"), Cue(5, 6, "extra words nobody subtitled")]
        r = compare(hyp, ref)
        self.assertGreater(r["wer_approx"], 1.0)
        self.assertEqual(r["sub_del_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

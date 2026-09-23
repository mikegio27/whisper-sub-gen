"""app.qa: each violation type on small synthetic cue lists."""

from __future__ import annotations

import unittest

from app.qa import COUNT_KEYS, format_report, percentile, score
from app.srt import Cue
from app.standards import CueRules

# Comfortable on every rule: 2 s, 10 chars (5 cps), one short line.
OK = "Fine text."


def cues_at(*spans: tuple[float, float], text: str = OK) -> list[Cue]:
    return [Cue(s, e, text) for s, e in spans]


class ScoreTest(unittest.TestCase):
    def counts(self, cues, **kw):
        return score(cues, **kw)["counts"]

    def test_clean_file_has_no_violations(self):
        s = score([Cue(0, 2.2, "First line."), Cue(3, 5.2, "Second line."), Cue(6, 8.2, "Third.")])
        self.assertEqual(s["cues"], 3)
        self.assertEqual(set(s["counts"]), set(COUNT_KEYS))
        self.assertTrue(all(v == 0 for k, v in s["counts"].items()))
        self.assertEqual(s["violations_per_100"], 0.0)

    def test_empty_input(self):
        s = score([])
        self.assertEqual(s["cues"], 0)
        self.assertEqual(s["violations_per_100"], 0.0)
        self.assertTrue(all(v == 0.0 for v in s["rates"].values()))
        self.assertEqual(s["stats"]["median_duration"], 0.0)
        self.assertIn("cues", format_report(s))

    def test_too_short(self):
        c = self.counts([Cue(0, 0.5, "Hi."), Cue(1, 1.833, "Hi.")])
        self.assertEqual(c["too_short"], 1)  # exactly min_duration is allowed

    def test_too_long_and_very_long(self):
        c = self.counts(cues_at((0, 7.0), (10, 17.5), (20, 31)))
        self.assertEqual(c["too_long"], 2)
        self.assertEqual(c["very_long"], 1)

    def test_cps(self):
        # 36 chars: 2 s -> 18 cps (over target only), 1.5 s -> 24 cps (over both)
        text = "x" * 36
        c = self.counts([Cue(0, 2, text), Cue(5, 6.5, text), Cue(10, 13, text)])
        self.assertEqual(c["cps_over_target"], 2)
        self.assertEqual(c["cps_over_max"], 1)

    def test_zero_duration_counts_as_unreadable(self):
        c = self.counts([Cue(1, 1, "Hi.")])
        self.assertEqual(c["cps_over_max"], 1)
        self.assertEqual(c["too_short"], 1)

    def test_gaps(self):
        # gaps: 0 (touching), 0.083 (ok), 0.05 (too small), -0.5 (overlap); last has no next
        c = self.counts(cues_at((0, 2), (2, 4), (4.083, 6), (6.05, 8), (7.5, 9)))
        self.assertEqual(c["touching_or_overlap"], 3)
        self.assertEqual(c["overlap"], 1)

    def test_unsorted_input_is_sorted_first(self):
        c = self.counts(cues_at((3, 5), (0, 2)))
        self.assertEqual(c["touching_or_overlap"], 0)

    def test_line_length_and_line_count(self):
        long_line = "y" * 43
        cues = [
            Cue(0, 5, long_line),
            Cue(6, 11, "y" * 42 + "\nshort"),
            Cue(12, 17, "one\ntwo\nthree"),
        ]
        c = self.counts(cues)
        self.assertEqual(c["line_too_long"], 1)
        self.assertEqual(c["too_many_lines"], 1)

    def test_consecutive_duplicates_are_normalized(self):
        cues = [
            Cue(0, 2, "He's got Padfoot."),
            Cue(3, 5, "he's got  padfoot"),
            Cue(6, 8, "Something else."),
            Cue(9, 11, "He's got Padfoot."),
        ]
        self.assertEqual(self.counts(cues)["consecutive_duplicate_text"], 1)

    def test_whole_second_durations(self):
        c = self.counts(cues_at((0.5, 1.5), (2, 4.0005), (5, 6.5), (7.1, 10.1)))
        self.assertEqual(c["whole_second_durations"], 3)

    def test_rates_and_violations_per_100(self):
        # 4 cues: one too short, one touching its successor -> 2 violations / 4 cues
        cues = [Cue(0, 0.5, "Hi."), Cue(1, 3, "A."), Cue(3, 5, "B."), Cue(6, 8, "C.")]
        s = score(cues)
        self.assertAlmostEqual(s["rates"]["too_short"], 0.25)
        self.assertAlmostEqual(s["violations_per_100"], 50.0)

    def test_custom_rules(self):
        c = self.counts(cues_at((0, 5)), rules=CueRules(max_duration=4.0))
        self.assertEqual(c["too_long"], 1)

    def test_stats(self):
        s = score(cues_at((0, 1), (2, 4), (5, 8)))
        self.assertAlmostEqual(s["stats"]["median_duration"], 2.0)
        self.assertAlmostEqual(s["stats"]["max_duration"], 3.0)
        self.assertAlmostEqual(s["stats"]["median_cps"], 5.0)

    def test_format_report_lists_every_count(self):
        report = format_report(score(cues_at((0, 2))))
        for key in COUNT_KEYS:
            self.assertIn(key, report)


class PercentileTest(unittest.TestCase):
    def test_interpolates(self):
        self.assertEqual(percentile([], 90), 0.0)
        self.assertEqual(percentile([3.0], 95), 3.0)
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], 50), 3.0)
        self.assertAlmostEqual(percentile([0, 10], 95), 9.5)


if __name__ == "__main__":
    unittest.main()

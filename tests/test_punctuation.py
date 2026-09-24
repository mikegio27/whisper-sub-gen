"""Re-decoding whisper's no-punctuation stretches (examples from The Rock, 2026-09-24)."""

from __future__ import annotations

import unittest

from app.cues import Word
from app.punctuation import accept, groups, is_unpunctuated, repair


class S:
    def __init__(self, start, end, text):
        toks = text.split()
        step = (end - start) / max(len(toks), 1)
        self.start, self.end = start, end
        self.words = [
            Word(start + i * step, start + (i + 1) * step, " " + t) for i, t in enumerate(toks)
        ]


def words(start, end, text):
    return S(start, end, text).words


class DetectTest(unittest.TestCase):
    def test_signature(self):
        self.assertTrue(is_unpunctuated("what's your news baby i'm pregnant i'm sorry"))
        self.assertFalse(is_unpunctuated("What's your news, baby?"))
        self.assertFalse(is_unpunctuated("yeah okay"))  # too short to judge
        self.assertFalse(is_unpunctuated("the rock is the most famous prison."))

    def test_groups_merge_adjacent_within_a_window(self):
        segs = [
            S(0, 3, "Hello there."),
            S(3, 8, "oh okay you go first just some"),
            S(8, 12, "which had to be neutralized before"),
            S(12, 14, "Fine."),
            S(20, 25, "systems up possible penetration point"),
        ]
        self.assertEqual(groups(segs), [[1, 2], [4]])
        self.assertEqual(groups(segs, max_window_s=6), [[1], [2], [4]])


class AcceptTest(unittest.TestCase):
    def test_same_words_now_punctuated(self):
        old = words(0, 3, "what's your news baby i'm pregnant i'm sorry i'm pregnant")
        new = words(0, 3, "What's your news, baby? I'm pregnant. I'm sorry? I'm pregnant.")
        self.assertIsNone(accept(old, new))

    def test_rejections(self):
        old = words(0, 3, "you will not be detained one minute longer")
        self.assertEqual(accept(old, []), "empty")
        self.assertEqual(
            accept(old, words(0, 3, "you will not be detained one minute longer")),
            "still unpunctuated",
        )
        self.assertEqual(
            accept(old, words(0, 3, "Welcome to Alcatraz. Watch your step, please.")),
            "different words",
        )


class RepairTest(unittest.TestCase):
    def test_replaces_run_and_drops_neighbour_words(self):
        segs = [
            S(0, 4, "I brought goodies."),
            S(4.0, 8.0, "which had to be neutralized before blowing up the office"),
            S(8.5, 10, "Right."),
        ]

        def decode(a, b):
            # The padded window caught the tail of the previous line.
            return words(a, a + 0.3, "Goodies,") + words(
                4.0, 8.0, "which had to be neutralized before blowing up the office."
            )

        out, stats = repair(segs, decode)
        self.assertEqual(stats.repaired, 1)
        self.assertEqual(len(out), 3)
        text = " ".join(w.text.strip() for w in out[1].words)
        self.assertEqual(text, "which had to be neutralized before blowing up the office.")
        self.assertEqual((out[1].start, out[1].end), (4.0, 8.0))

    def test_failed_or_bad_decode_keeps_original(self):
        segs = [S(0, 4, "oh okay you go first just some terrorists")]

        def boom(a, b):
            raise RuntimeError("cuda")

        out, stats = repair(segs, boom)
        self.assertIs(out[0], segs[0])
        self.assertEqual(stats.reasons, {"decode error": 1})
        out, stats = repair(segs, lambda a, b: words(0, 4, "Something else entirely here."))
        self.assertIs(out[0], segs[0])
        self.assertEqual(stats.reasons, {"different words": 1})

    def test_nothing_to_do(self):
        segs = [S(0, 2, "Hello, there.")]
        out, stats = repair(segs, lambda a, b: self.fail("decode called"))
        self.assertEqual((out, stats.candidates), (segs, 0))


if __name__ == "__main__":
    unittest.main()

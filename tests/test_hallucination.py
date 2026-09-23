"""Non-speech hallucination filter (real examples from the Fargo/Interstellar intros)."""

from __future__ import annotations

import unittest

from app.cues import Word
from app.hallucination import Segment, filter_segments, is_caps_line, is_descriptor


def seg(start: float, text: str, dur: float = 1.5, prob: float = 0.95) -> Segment:
    words = text.split()
    step = dur / max(len(words), 1)
    return Segment(
        start,
        start + dur,
        tuple(
            Word(start + i * step, start + (i + 1) * step, " " + w, prob)
            for i, w in enumerate(words)
        ),
    )


class DescriptorTest(unittest.TestCase):
    def test_music_marks_and_brackets(self):
        for t in ("¶¶", "♪ ♪", "[MUSIC]", "(GUNSHOT)"):
            self.assertTrue(is_descriptor(t), t)

    def test_real_lines_are_not_descriptors(self):
        for t in ("NO!", "PIANO PLAYS", "Yeah.", "I'm, uh, Jerry Lundegaard.", "Dad?"):
            self.assertFalse(is_descriptor(t), t)

    def test_caps_lines(self):
        self.assertTrue(is_caps_line("PIANO PLAYS"))
        self.assertTrue(is_caps_line("EXPLOSION"))
        self.assertFalse(is_caps_line("NO!"))
        self.assertFalse(is_caps_line("FBI!"))
        for t in ("OK. OK.", "OK, OK!", "I... I...", "NO! NO!"):
            self.assertFalse(is_caps_line(t), t)
        self.assertTrue(is_caps_line("STOP IT!"))  # a caps line; confidence decides


class FilterTest(unittest.TestCase):
    def test_fargo_intro(self):
        segs = [
            seg(30.2, "Thank you.", prob=0.56),
            seg(111.0, "PIANO PLAYS", prob=0.73),
            seg(135.7, "¶¶"),
            seg(151.6, "Thank you.", prob=0.54),
            seg(186.5, "THE END", prob=0.36),
            seg(222.9, "I'm, uh, Jerry Lundegarden."),
            seg(225.8, "You're Jerry Lundegarden?"),
        ]
        kept, dropped = filter_segments(segs)
        self.assertEqual([s.text for s in kept], [s.text for s in segs[5:]])
        self.assertEqual(len(dropped), 5)

    def test_thank_you_inside_a_conversation_is_kept(self):
        segs = [seg(10.0, "Here are the keys."), seg(12.0, "Thank you."), seg(14.0, "Sure.")]
        kept, dropped = filter_segments(segs)
        self.assertEqual(len(kept), 3)
        self.assertEqual(dropped, [])

    def test_isolated_real_line_that_isnt_stock_is_kept(self):
        kept, _ = filter_segments([seg(10.0, "Whoa!"), seg(80.0, "Get in.")])
        self.assertEqual(len(kept), 2)

    def test_shouted_caps_line_in_a_scene_is_kept(self):
        segs = [seg(10.0, "Get down!"), seg(11.6, "NO! NO!", prob=0.97), seg(13.2, "Run!")]
        kept, dropped = filter_segments(segs)
        self.assertEqual(len(kept), 3)
        self.assertEqual(dropped, [])

    def test_help_help_at_film_confidence_is_kept(self):
        segs = [seg(10.0, "Get down!"), seg(11.6, "HELP! HELP!", prob=0.75), seg(13.2, "Run!")]
        self.assertEqual(filter_segments(segs)[1], [])

    def test_confident_isolated_lines_are_kept(self):
        # Sparse-dialogue films: a lone, clearly heard line is real.
        segs = [seg(10.0, "Thank you.", prob=0.9), seg(80.0, "Bye.", prob=0.9)]
        segs += [seg(160.0, "STOP IT!", prob=0.97)]
        kept, dropped = filter_segments(segs)
        self.assertEqual((len(kept), dropped), (3, []))

    def test_low_confidence_caps_line_in_a_scene_is_dropped(self):
        segs = [seg(10.0, "Get down!"), seg(11.6, "GUNFIRE CONTINUES", prob=0.4), seg(13.2, "Run!")]
        kept, dropped = filter_segments(segs)
        self.assertEqual(dropped, ["GUNFIRE CONTINUES"])

    def test_empty(self):
        self.assertEqual(filter_segments([]), ([], []))


if __name__ == "__main__":
    unittest.main()

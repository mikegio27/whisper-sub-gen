"""app.srt: the lenient parser has to cope with real-world (human, downloaded) SRTs."""

from __future__ import annotations

import unittest

from app.srt import Cue, format_ts, parse_srt, render_srt


class ParseSrtTest(unittest.TestCase):
    def test_basic(self):
        cues = parse_srt("1\n00:00:01,000 --> 00:00:02,500\nHello\n")
        self.assertEqual(cues, [Cue(1.0, 2.5, "Hello")])

    def test_bom_and_crlf(self):
        text = (
            "\ufeff1\r\n00:00:01,000 --> 00:00:02,000\r\nHi\r\n\r\n"
            "2\r\n00:00:03,000 --> 00:00:04,000\r\nThere\r\n"
        )
        cues = parse_srt(text)
        self.assertEqual([c.text for c in cues], ["Hi", "There"])
        self.assertEqual(cues[1].start, 3.0)

    def test_missing_index_line(self):
        cues = parse_srt("00:00:01,000 --> 00:00:02,000\nNo index\n")
        self.assertEqual(cues, [Cue(1.0, 2.0, "No index")])

    def test_dot_millisecond_separator_and_short_ms(self):
        cues = parse_srt("1\n00:00:01.5 --> 00:00:02.25\nDots\n")
        self.assertAlmostEqual(cues[0].start, 1.5)
        self.assertAlmostEqual(cues[0].end, 2.25)

    def test_tags_stripped_by_default(self):
        text = "1\n00:00:01,000 --> 00:00:02,000\n{\\an8}<i>Quiet</i> <b>now</b>\n"
        self.assertEqual(parse_srt(text)[0].text, "Quiet now")
        self.assertEqual(
            parse_srt(text, strip_tags=False)[0].text, "{\\an8}<i>Quiet</i> <b>now</b>"
        )

    def test_multi_line_body(self):
        cues = parse_srt("1\n00:00:01,000 --> 00:00:02,000\n  Line one  \nLine two\n")
        self.assertEqual(cues[0].text, "Line one\nLine two")
        self.assertEqual(cues[0].lines, ["Line one", "Line two"])
        self.assertEqual(cues[0].chars, len("Line oneLine two"))

    def test_blank_body_and_tag_only_cues_dropped(self):
        text = (
            "1\n00:00:01,000 --> 00:00:02,000\n\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n<i></i>\n\n"
            "3\n00:00:05,000 --> 00:00:06,000\nKept\n"
        )
        self.assertEqual([c.text for c in parse_srt(text)], ["Kept"])

    def test_sorted_by_start(self):
        text = "00:00:05,000 --> 00:00:06,000\nB\n\n00:00:01,000 --> 00:00:02,000\nA\n"
        self.assertEqual([c.text for c in parse_srt(text)], ["A", "B"])

    def test_empty_input(self):
        self.assertEqual(parse_srt(""), [])
        self.assertEqual(parse_srt("\ufeff\r\n"), [])


class FormatTsTest(unittest.TestCase):
    def test_round_trip_through_parser(self):
        for secs in (0.0, 0.001, 1.5, 59.999, 3661.234, 100 * 3600 + 0.5):
            line = f"{format_ts(secs)} --> {format_ts(secs + 1)}\nx\n"
            cue = parse_srt(line)[0]
            self.assertAlmostEqual(cue.start, secs, places=6)
            self.assertAlmostEqual(cue.end, secs + 1, places=6)


class RenderSrtTest(unittest.TestCase):
    def test_render_parse_round_trip(self):
        cues = [
            Cue(1.0, 2.5, "Hello there."),
            Cue(3.0, 5.25, "Two lines,\nsecond one."),
            Cue(3725.125, 3727.0, "Late in the film."),
        ]
        self.assertEqual(parse_srt(render_srt(cues)), cues)

    def test_numbering_starts_at_one(self):
        out = render_srt([Cue(0, 1, "a"), Cue(2, 3, "b")])
        self.assertTrue(out.startswith("1\n00:00:00,000 --> 00:00:01,000\na\n"))
        self.assertIn("\n2\n00:00:02,000 --> 00:00:03,000\nb\n", out)

    def test_render_empty(self):
        self.assertEqual(render_srt([]), "")


class CueTest(unittest.TestCase):
    def test_cps_zero_duration_is_inf(self):
        self.assertEqual(Cue(1, 1, "abc").cps, float("inf"))
        self.assertAlmostEqual(Cue(0, 2, "ab\ncd").cps, 2.0)


if __name__ == "__main__":
    unittest.main()

import random
import re
import unittest

from app.cues import Word, compose
from app.standards import DEFAULT_RULES, CueRules

R = DEFAULT_RULES
EPS = 1e-6


def words_from(text: str, start: float = 0.0, per_word: float = 0.3, gap: float = 0.05):
    """Evenly timed whisper-style words (leading spaces) for one stretch of speech."""
    out, t = [], start
    for tok in text.split():
        out.append(Word(t, t + per_word, " " + tok))
        t += per_word + gap
    return out


def norm_tokens(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


class CueInvariants:
    """Property checks every compose() output must satisfy."""

    def check(self, cues, rules: CueRules = R):
        for c in cues:
            self.assertGreater(c.end, c.start, c)
            self.assertLessEqual(c.duration, rules.max_duration + EPS, c)
            self.assertLessEqual(len(c.lines), rules.max_lines, c)
            for line in c.lines:
                if " " in line:  # a single over-long word may exceed the limit
                    self.assertLessEqual(len(line), rules.max_line_chars, c)
            self.assertEqual(c.text, c.text.strip())
            self.assertNotIn("  ", c.text)
            # ms rounding
            self.assertAlmostEqual(c.start * 1000, round(c.start * 1000), places=6)
            self.assertAlmostEqual(c.end * 1000, round(c.end * 1000), places=6)
        for a, b in zip(cues, cues[1:], strict=False):
            self.assertGreaterEqual(b.start - a.end, rules.min_gap - EPS, (a, b))
            # A cue may only be shorter than min_duration when the next cue squeezed it
            # (it then ends min_gap before the next one, give or take the 1 ms that keeps
            # float comparisons honest).
            if a.duration < rules.min_duration - EPS:
                self.assertAlmostEqual(b.start - a.end, rules.min_gap, delta=0.0015, msg=(a, b))
        if cues:
            self.assertGreaterEqual(cues[-1].duration, rules.min_duration - EPS)


class PropertyTest(CueInvariants, unittest.TestCase):
    def generated_stream(self, seed: int, n: int = 2000):
        rng = random.Random(seed)
        vocab = (
            "the a of to and harry you I don't know what we're going ministry wizard "
            "dumbledore professor Mr. Weasley extraordinarily-long-compound-word be it"
        ).split()
        ws, t = [], 0.0
        for _ in range(n):
            tok = rng.choice(vocab)
            r = rng.random()
            if r < 0.08:
                tok += "."
            elif r < 0.12:
                tok += ","
            elif r < 0.14:
                tok += "?"
            dur = rng.uniform(0.08, 0.6)
            ws.append(Word(round(t, 3), round(t + dur, 3), " " + tok, rng.random()))
            p = rng.random()
            t += dur + (
                rng.uniform(0.6, 4.0) if p < 0.05 else rng.uniform(0.0, 0.5) if p < 0.3 else 0.0
            )
        return ws

    def test_generated_streams(self):
        for seed in (1, 2, 3):
            ws = self.generated_stream(seed)
            cues = compose(ws)
            self.check(cues)
            # Expected text: the input with runs of >3 identical words cut to 3.
            expected, run = [], 0
            for i, w in enumerate(ws):
                key = norm_tokens(w.text)
                run = run + 1 if i and norm_tokens(ws[i - 1].text) == key else 1
                if run <= 3:
                    expected += key
            got = [t for c in cues for t in norm_tokens(c.text)]
            # The documented repeat-cue drop may remove whole short cues, nothing else: the
            # output is an in-order subsequence that keeps almost everything.
            it = iter(expected)
            self.assertTrue(all(tok in it for tok in got))
            self.assertGreater(len(got), 0.97 * len(expected))

    def test_generated_stream_exact_preservation(self):
        # Unique tokens: nothing can count as a loop, so every word must come through in order.
        ws = [
            Word(w.start, w.end, f" w{i}{w.text[-1] if w.text[-1] in '.,?' else ''}")
            for i, w in enumerate(self.generated_stream(7))
        ]
        cues = compose(ws)
        self.check(cues)
        got = [t for c in cues for t in norm_tokens(c.text)]
        self.assertEqual(got, [f"w{i}" for i in range(len(ws))])

    def test_junk_input_still_valid(self):
        ws = [
            Word(5.0, 4.0, " backwards"),
            Word(1.0, 1.2, "   "),
            Word(0.0, 0.3, " Hello"),
            Word(0.3, 0.2, ","),
            Word(0.4, 90.0, " stretched"),
            Word(0.35, 0.5, " early"),
            Word(6.0, 6.1, " ♪"),
            Word(6.1, 6.5, " la"),
            Word(6.5, 6.6, " ♪"),
            Word(7.0, 7.0, " zero"),
            Word(7.0, 7.0, " zero2"),
        ]
        cues = compose(ws)
        self.check(cues)
        text = " ".join(c.text for c in cues)
        self.assertIn("Hello,", text)
        self.assertIn("♪ la ♪", text)
        self.assertLess(text.index("early"), text.index("stretched"))

    def test_other_rules(self):
        rules = CueRules(max_line_chars=30, max_lines=3, min_duration=1.0, max_duration=5.0)
        cues = compose(self.generated_stream(11, 500), rules)
        self.check(cues, rules)


class TargetedTest(CueInvariants, unittest.TestCase):
    def test_empty(self):
        self.assertEqual(compose([]), [])
        self.assertEqual(compose([Word(0, 1, " "), Word(1, 2, "")]), [])

    def test_pause_splits(self):
        first = words_from("I am here")
        ws = first + words_from("and so", start=first[-1].end + R.split_pause)
        cues = compose(ws)
        self.check(cues)
        self.assertEqual([c.text for c in cues], ["I am here", "and so"])

    def test_short_pause_does_not_split_short_text(self):
        ws = words_from("I am here") + words_from("and so are you.", start=1.05 + 0.3)
        self.assertEqual([c.text for c in compose(ws)], ["I am here and so are you."])

    def test_sentence_end_preferred(self):
        text = (
            "We have to leave the castle tonight before anyone notices. "
            "The ministry is already watching every single one of us."
        )
        cues = compose(words_from(text))
        self.check(cues)
        self.assertEqual(len(cues), 2)
        self.assertTrue(cues[0].text.endswith("notices."), cues)

    def test_stretched_word_across_silence(self):
        # ROADMAP: "Mrs. Fig?" stayed up from 2:37 to 4:40 because whisper stretched the
        # last word across the music.
        ws = [
            Word(156.14, 156.4, " I'm"),
            Word(156.45, 156.6, " not"),
            Word(156.65, 156.9, " doing"),
            Word(156.95, 157.1, " anything."),
            Word(157.14, 157.5, " Mrs."),
            Word(157.5, 280.9, " Fig?"),
            Word(280.92, 281.2, " Well,"),
            Word(281.25, 281.5, " you're"),
            Word(281.55, 281.9, " warned."),
        ]
        cues = compose(ws)
        self.check(cues)
        fig = next(c for c in cues if "Fig?" in c.text)
        self.assertLess(fig.duration, 5.0)
        self.assertLessEqual(fig.end, 157.5 + R.max_word_duration + R.max_linger + EPS)
        self.assertTrue(cues[-1].text.startswith("Well,"))
        self.assertAlmostEqual(cues[-1].start, 280.92)

    def test_weak_word_not_at_cue_end(self):
        # 88 chars: must be two cues. A char-greedy split lands right after "the".
        text = (
            "Harry and Ron walked all the way down to the old castle gates and waited for the rest"
        )
        cues = compose(words_from(text, per_word=0.2, gap=0.02))
        self.check(cues)
        self.assertGreater(len(cues), 1)
        weak = {"the", "a", "of", "to", "and", "for"}
        for c in cues[:-1]:
            self.assertNotIn(c.text.split()[-1].lower(), weak, cues)

    def test_line_balance_60_chars(self):
        text = "I never thought I would see you standing here in the hall."
        self.assertEqual(len(text), 58)
        cues = compose(words_from(text, per_word=0.35))
        self.assertEqual(len(cues), 1)
        top, bottom = cues[0].lines
        self.assertLessEqual(abs(len(top) - len(bottom)), 10, cues[0].lines)
        self.assertNotIn(top.split()[-1], {"the", "a", "in"})

    def test_line_break_after_punctuation(self):
        cues = compose(words_from("Harry, listen to me, you can't go back there tonight alone."))
        self.assertEqual(
            cues[0].lines, ["Harry, listen to me,", "you can't go back there tonight alone."]
        )

    def test_single_line_when_it_fits(self):
        cues = compose(words_from("Mr. Weasley, it's you."))
        self.assertEqual(cues[0].text, "Mr. Weasley, it's you.")

    def test_joining(self):
        ws = [
            Word(0.0, 0.2, " I"),
            Word(0.2, 0.4, " don"),
            Word(0.4, 0.5, "'t"),
            Word(0.5, 0.7, " know"),
            Word(0.7, 0.8, "..."),
            Word(0.8, 1.0, " Harry"),
            Word(1.0, 1.1, " 's"),
            Word(1.1, 1.3, " wand"),
            Word(1.3, 1.4, " ?"),
            Word(1.4, 1.5, " Mr."),
            Word(1.5, 1.7, " Weasley"),
            Word(1.7, 1.8, ","),
            Word(1.8, 2.0, " well"),
            Word(2.0, 2.2, "-known"),
            Word(2.2, 2.3, " ("),
            Word(2.3, 2.4, "ahem"),
            Word(2.4, 2.5, ")"),
        ]
        cues = compose(ws)
        self.assertEqual(
            " ".join(c.text.replace("\n", " ") for c in cues),
            "I don't know... Harry's wand? Mr. Weasley, well-known (ahem)",
        )

    def test_unspaced_stream_is_not_glued(self):
        # Aligner output without leading spaces must not be read as sub-word pieces.
        ws = [Word(i * 0.3, i * 0.3 + 0.25, t) for i, t in enumerate(["Hello", "there", "Harry."])]
        self.assertEqual(compose(ws)[0].text, "Hello there Harry.")

    def test_cps_extends_end_toward_target(self):
        text = "You have no idea how much trouble you are in."
        ws = words_from(text, per_word=0.12, gap=0.0)  # spoken fast: 1.2 s for 45 chars
        cues = compose(ws)
        self.assertEqual(len(cues), 1)
        c = cues[0]
        spoken_end = ws[-1].end
        self.assertGreater(c.end, spoken_end)
        self.assertAlmostEqual(c.end, spoken_end + R.max_linger, places=3)  # linger cap
        slow = compose(words_from("Yes.", per_word=0.3))[0]
        self.assertAlmostEqual(slow.duration, max(R.min_duration, 0.3 + R.min_linger), places=3)
        mws = words_from("It's in the cupboard under the stairs.", per_word=0.25)
        mid = compose(mws)[0]
        # Whichever is later: reading time at target_cps, or speech end + min_linger.
        want = max(len(mid.text) / R.target_cps, mws[-1].end + R.min_linger - mid.start)
        self.assertAlmostEqual(mid.duration, want, delta=0.002)

    def test_min_linger_holds_a_cue_after_speech(self):
        ws = words_from("Go now.", per_word=0.5)
        c = compose(ws)[0]
        self.assertAlmostEqual(c.end, ws[-1].end + R.min_linger, places=3)
        # ...unless the next cue needs the room: then min_gap wins.
        a = words_from("Go now.", per_word=0.5)
        b = words_from(
            "Come back tomorrow and bring the whole family along.",
            start=a[-1].end + R.min_linger + 0.3,
        )
        first, second = compose(a + b)[:2]
        self.assertAlmostEqual(first.end, a[-1].end + R.min_linger, places=3)
        c = words_from("Come back tomorrow.", start=a[-1].end + 0.3, per_word=0.3)
        cues = compose(a + c)
        if len(cues) == 2:
            self.assertGreaterEqual(round(cues[1].start - cues[0].end, 3), R.min_gap)
            self.assertLess(cues[0].end, a[-1].end + R.min_linger)

    def test_min_duration_beats_linger(self):
        rules = CueRules(max_linger=0.1)
        c = compose([Word(0.0, 0.1, " Oh.")], rules)[0]
        self.assertAlmostEqual(c.duration, rules.min_duration, places=3)

    def test_gap_and_nudge(self):
        # Room for min_duration before the next cue: nothing moves.
        ws = words_from("Oh.", per_word=0.1) + words_from("Who is there?", start=1.3)
        cues = compose(ws)
        self.check(cues)
        self.assertAlmostEqual(cues[0].duration, R.min_duration, places=3)
        self.assertAlmostEqual(cues[1].start, 1.3, places=3)
        # 0.75 - 0.083 = 0.667 s free, 0.833 needed: the next cue's start is pushed by at
        # most 0.1 s, then min_gap wins and this cue is squeezed.
        ws = words_from("Oh.", per_word=0.1) + words_from("Who is there?", start=0.75)
        cues = compose(ws)
        self.check(cues)
        self.assertEqual(len(cues), 2)
        self.assertAlmostEqual(cues[1].start, 0.85, places=3)
        self.assertAlmostEqual(cues[0].end, 0.85 - R.min_gap, delta=0.0015)
        # Only a little short: the nudge covers it fully.
        ws = words_from("Oh.", per_word=0.1) + words_from("Who is there?", start=0.88)
        cues = compose(ws)
        self.assertAlmostEqual(cues[0].duration, R.min_duration, places=3)
        self.assertAlmostEqual(cues[1].start, 0.833 + R.min_gap, places=3)

    def test_max_duration_cap(self):
        # 81 chars fits one cue by length, but it takes 8.8 s to say.
        text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen."
        cues = compose(words_from(text, per_word=0.6, gap=0.05))
        self.check(cues)
        self.assertGreater(len(cues), 1)

    def test_orphan_merged(self):
        cues = compose(words_from("I think we should all go to bed now. Ron."))
        self.assertEqual(len(cues), 1)

    def test_word_loop_collapsed(self):
        ws = words_from("no no no no no no no stop")
        self.assertEqual(" ".join(c.text for c in compose(ws)), "no no no stop")

    def test_repeated_cue_dropped(self):
        ws = words_from("Thank you.") + words_from("Thank you.", start=0.7 + R.split_pause)
        self.assertEqual([c.text for c in compose(ws)], ["Thank you."])
        far = words_from("Thank you.") + words_from("Thank you.", start=3.0)
        self.assertEqual(len(compose(far)), 2)


if __name__ == "__main__":
    unittest.main()

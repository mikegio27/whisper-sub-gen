"""Pure parts of app.align: no torch, no model, no GPU."""

import importlib.util
import sys
import unittest

import numpy as np

from app import align
from app.align import (
    AlignSegment,
    align_window,
    enforce_order,
    frames_for_samples,
    min_frames,
    normalize_word,
    plan_batches,
    plan_windows,
    spans_to_words,
    token_spans,
    viterbi,
)
from app.cues import Word

HAS_UROMAN = importlib.util.find_spec("uroman") is not None
HAS_NUM2WORDS = importlib.util.find_spec("num2words") is not None

BLANK = 0
HI, LO = np.log(0.9), np.log(0.1 / 4)


def peaky(frames: list[int], vocab: int = 5) -> np.ndarray:
    """A (T, V) log-prob matrix where frame t strongly prefers token frames[t]."""
    m = np.full((len(frames), vocab), LO)
    for t, tok in enumerate(frames):
        m[t, tok] = HI
    return m


class ImportTest(unittest.TestCase):
    def test_import_does_not_pull_torch(self):
        # The pipeline imports app.align at startup; the model stack must stay lazy.
        self.assertNotIn("transformers", sys.modules)
        self.assertTrue(hasattr(align, "Aligner"))


class ViterbiTest(unittest.TestCase):
    def test_known_path(self):
        # blank blank 1 1 blank 2 blank 3 3 blank
        m = peaky([0, 0, 1, 1, 0, 2, 0, 3, 3, 0])
        path = viterbi(m, [1, 2, 3], BLANK)
        self.assertEqual(path.tolist(), [-1, -1, 0, 0, -1, 1, -1, 2, 2, -1])
        self.assertEqual(token_spans(path, 3), [(2, 3), (5, 5), (7, 8)])

    def test_repeated_token_needs_a_blank_between(self):
        # "ll": the audio has the token on frames 1-2 and again on 4. A path must put a
        # blank between the two tokens, even where the emissions don't favour one.
        m = peaky([0, 1, 1, 1, 0])
        path = viterbi(m, [1, 1], BLANK)
        spans = token_spans(path, 2)
        self.assertLess(spans[0][1] + 1, spans[1][0], "a blank separates the two")
        # And with exactly min_frames frames there is only one path: tok blank tok.
        path = viterbi(peaky([1, 1, 1]), [1, 1], BLANK)
        self.assertEqual(path.tolist(), [0, -1, 1])

    def test_different_tokens_may_touch(self):
        path = viterbi(peaky([1, 2]), [1, 2], BLANK)
        self.assertEqual(path.tolist(), [0, 1])

    def test_too_few_frames(self):
        self.assertEqual(min_frames([1, 1, 2]), 4)
        self.assertIsNone(viterbi(peaky([1, 2, 0]), [1, 1, 2], BLANK))
        self.assertIsNone(viterbi(peaky([1]), [1, 2], BLANK))
        self.assertIsNone(viterbi(peaky([1, 2]), [], BLANK))

    def test_long_sequence(self):
        # > 127 states: the int8 back-pointers must not leak into the state arithmetic.
        tokens = [1 + (k % 4) for k in range(150)]
        frames = [f for tok in tokens for f in (tok, 0)]
        path = viterbi(peaky(frames), tokens, BLANK)
        self.assertEqual(token_spans(path, 150)[-1], (298, 298))
        self.assertEqual(token_spans(path, 150)[0], (0, 0))

    def test_matches_brute_force(self):
        # Every CTC path for tokens [1, 2, 1] in 6 frames, scored exhaustively.
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(6, 4))
        m = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        tokens = [1, 2, 1]
        best, best_path = -np.inf, None
        for labels in np.ndindex(*(4,) * 6):
            collapsed = [
                k for i, k in enumerate(labels) if k != 0 and (i == 0 or labels[i - 1] != k)
            ]
            if collapsed != tokens:
                continue
            score = sum(m[t, k] for t, k in enumerate(labels))
            if score > best:
                best, best_path = score, labels
        path = viterbi(m, tokens, BLANK)
        emitted = [tokens[k] if k >= 0 else 0 for k in path]
        self.assertAlmostEqual(sum(m[t, k] for t, k in enumerate(emitted)), best)
        self.assertEqual(tuple(emitted), best_path)


class AlignWindowTest(unittest.TestCase):
    def test_word_spans(self):
        # word "ab" (1,2) then word "c" (3), with blanks around.
        m = peaky([0, 1, 2, 0, 0, 3, 3, 0])
        wa = align_window(m, [[1, 2], [], [3]], blank=BLANK, star=False)
        self.assertIsNone(wa.reason)
        self.assertEqual(wa.spans, [(1, 2), None, (5, 6)])
        self.assertAlmostEqual(wa.score, HI)

    def test_star_absorbs_untranscribed_edges(self):
        # Speech the text doesn't cover (tokens 4 4) sits before the word; the star eats it
        # instead of stretching the word over it.
        m = peaky([4, 4, 0, 1, 2, 0, 4])
        wa = align_window(m, [[1, 2]], blank=BLANK, star=True)
        self.assertIsNone(wa.reason)
        self.assertEqual(wa.spans, [(3, 4)])

    def test_no_tokens(self):
        wa = align_window(peaky([0, 0]), [[], []], blank=BLANK)
        self.assertEqual(wa.reason, "no_tokens")

    def test_window_too_short(self):
        wa = align_window(peaky([1, 2]), [[1, 2, 3]], blank=BLANK, star=True)
        self.assertEqual(wa.reason, "too_short")
        self.assertIsNone(wa.spans)

    def test_low_score_falls_back(self):
        # The audio says 1 1 1 1; the text claims 2 3: a forced but terrible alignment.
        m = np.full((8, 5), np.log(1e-4))
        m[:, 1] = np.log(0.9996)
        wa = align_window(m, [[2, 3]], blank=BLANK, star=True, min_score=-5.0)
        self.assertEqual(wa.reason, "low_score")
        self.assertLess(wa.score, -5.0)
        self.assertIsNone(wa.spans)
        ok = align_window(m, [[2, 3]], blank=BLANK, star=True, min_score=-20.0)
        self.assertIsNone(ok.reason)


class TimesTest(unittest.TestCase):
    def test_spans_to_seconds(self):
        words = [Word(0, 1, " Hello", 0.9), Word(1, 2, " 42,", 0.8), Word(2, 3, " there.", 0.7)]
        out = spans_to_words(words, [(5, 14), None, (30, 39)], 10.0, frame_s=0.02)
        self.assertAlmostEqual(out[0].start, 10.10)
        self.assertAlmostEqual(out[0].end, 10.30)
        # the unalignable word fills the gap between its neighbours
        self.assertAlmostEqual(out[1].start, 10.30)
        self.assertAlmostEqual(out[1].end, 10.60)
        self.assertAlmostEqual(out[2].end, 10.80)
        self.assertEqual([w.text for w in out], [" Hello", " 42,", " there."])
        self.assertEqual([w.prob for w in out], [0.9, 0.8, 0.7])

    def test_end_clamped_to_window(self):
        out = spans_to_words([Word(0, 1, "x")], [(0, 99)], 0.0, frame_s=0.02, window_end=1.5)
        self.assertAlmostEqual(out[0].end, 1.5)

    def test_unalignable_edges(self):
        words = [Word(0, 1, "1"), Word(1, 2, "b"), Word(2, 3, "2")]
        out = spans_to_words(words, [None, (10, 19), None], 0.0, frame_s=0.02)
        self.assertEqual((out[0].start, out[0].end), (0.2, 0.2))
        self.assertEqual((out[2].start, out[2].end), (0.4, 0.4))

    def test_enforce_order(self):
        out = enforce_order([Word(0.0, 1.2, "a"), Word(1.0, 1.1, "b"), Word(1.5, 2.0, "c")])
        self.assertEqual([(w.start, w.end) for w in out], [(0.0, 1.2), (1.2, 1.2), (1.5, 2.0)])

    def test_frames_for_samples(self):
        kernels, strides = [10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]
        self.assertEqual(frames_for_samples(16000, kernels, strides), 49)
        self.assertEqual(frames_for_samples(400, kernels, strides), 1)
        self.assertEqual(frames_for_samples(399, kernels, strides), 0)


class WindowsTest(unittest.TestCase):
    def seg(self, start, end):
        return AlignSegment(start, end, [Word(start, end, "x")])

    def test_padding_and_clamping(self):
        wins = plan_windows([self.seg(0.2, 2.0), self.seg(10.0, 12.0)], 12.3)
        self.assertEqual(wins, [(0.0, 2.5), (9.5, 12.3)])

    def test_neighbours_limit_overlap(self):
        # 0.1 s gap: each window may only reach 0.25 s into the other's core, not the full pad.
        a, b = self.seg(1.0, 5.0), self.seg(5.1, 8.0)
        (a_lo, a_hi), (b_lo, b_hi) = plan_windows([a, b], 100)
        self.assertAlmostEqual(a_lo, 0.5)
        self.assertAlmostEqual(a_hi, 5.1 + 0.25)
        self.assertAlmostEqual(b_lo, 5.0 - 0.25)
        self.assertAlmostEqual(b_hi, 8.5)

    def test_never_cuts_own_core(self):
        # Whisper segments that overlap each other: the window still covers the whole core.
        a, b = self.seg(1.0, 5.0), self.seg(4.0, 8.0)
        (_, a_hi), (b_lo, _) = plan_windows([a, b], 100)
        self.assertGreaterEqual(a_hi, 5.0)
        self.assertLessEqual(b_lo, 4.0)

    def test_core_covers_words(self):
        seg = AlignSegment(2.0, 3.0, [Word(1.8, 2.5, "a"), Word(2.5, 3.4, "b")])
        self.assertEqual(plan_windows([seg], 100), [(1.3, 3.9)])

    def test_plan_batches(self):
        lengths = [10, 50, 20, 45, 200]
        batches = plan_batches(lengths, budget=100)
        self.assertEqual(sorted(i for b in batches for i in b), [0, 1, 2, 3, 4])
        for b in batches:
            self.assertTrue(len(b) == 1 or len(b) * max(lengths[i] for i in b) <= 100)
        self.assertEqual(batches[0], [4])  # longer than the budget: alone
        self.assertEqual(plan_batches([5] * 10, budget=1000, max_items=4)[0], [0, 1, 2, 3])


class NormalizeTest(unittest.TestCase):
    def test_punctuation_and_case(self):
        self.assertEqual(normalize_word(" Hello,"), "hello")
        self.assertEqual(normalize_word(' "Mr.'), "mr")
        self.assertEqual(normalize_word("..."), "")
        self.assertEqual(normalize_word(" -"), "")
        self.assertEqual(normalize_word(" well-known"), "wellknown")

    def test_apostrophes(self):
        self.assertEqual(normalize_word(" don't"), "don't")
        self.assertEqual(normalize_word(" Don’t!"), "don't")
        self.assertEqual(normalize_word(" 'cause"), "cause")
        self.assertEqual(normalize_word(" dancin'"), "dancin")

    def test_accents(self):
        self.assertEqual(normalize_word(" Café"), "cafe")
        self.assertEqual(normalize_word(" naïve"), "naive")

    @unittest.skipUnless(HAS_NUM2WORDS, "num2words not installed")
    def test_numbers(self):
        self.assertEqual(normalize_word(" 42"), "fortytwo")
        self.assertEqual(normalize_word(" 1984,"), "nineteeneightyfour")
        self.assertEqual(normalize_word(" 21st"), "twentyfirst")
        self.assertEqual(normalize_word(" 1,000"), "onethousand")
        self.assertEqual(normalize_word(" 3.5"), "threepointfive")
        self.assertEqual(normalize_word(" 42", "fr"), "quarantedeux")
        # A language num2words doesn't know: digits stay, and are dropped (unalignable).
        self.assertEqual(normalize_word(" 42", "xx"), "")

    def test_numbers_outside_the_allowlist_never_reach_num2words(self):
        # num2words hangs forever on Amharic 7-digit numbers (review, 2026-09-23).
        orig = align._num2words
        try:
            align._num2words = lambda: self.fail("num2words called for 'am'")
            self.assertEqual(normalize_word(" 1234567", "am"), "")
        finally:
            align._num2words = orig

    def test_num2words_errors_of_any_type_keep_digits(self):
        orig = align._num2words

        def boom(*a, **k):
            raise TypeError("converter bug")

        try:
            align._num2words = lambda: boom
            self.assertEqual(normalize_word(" 42"), "")
            self.assertEqual(normalize_word(" 1234567890123"), "")  # over the digit cap
        finally:
            align._num2words = orig

    def test_numbers_without_num2words(self):
        orig = align._num2words
        try:
            align._num2words = lambda: None
            self.assertEqual(normalize_word(" 42"), "")
            self.assertEqual(normalize_word(" 4x4"), "x")
        finally:
            align._num2words = orig

    @unittest.skipUnless(HAS_UROMAN, "uroman not installed")
    def test_non_latin_through_uroman(self):
        self.assertEqual(normalize_word(" Привет,"), "privet")
        self.assertEqual(normalize_word("Ελλάδα"), "ellada")
        self.assertEqual(normalize_word(" Straße"), "strasse")

    def test_alphabet_restricts(self):
        self.assertEqual(normalize_word(" hello", alphabet="helo"), "hello")
        self.assertEqual(normalize_word(" hi", alphabet="h"), "h")


if __name__ == "__main__":
    unittest.main()

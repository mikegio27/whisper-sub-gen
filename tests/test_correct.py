import io
import json
import threading
import unittest
import urllib.error
from unittest import mock

from app import context as ctxmod
from app import correct as C
from app.context import MediaContext, media_context, parse_path
from app.cues import Word

HP = MediaContext(
    title="Harry Potter and the Order of the Phoenix",
    year=2007,
    names=("Harry Potter", "Albus Dumbledore", "Arabella Figg", "Dudley Dursley"),
    terms=("Hogwarts", "Voldemort"),
    source="jellyfin",
)


def W(text: str, start: float = 0.0, per_word: float = 0.3, probs: dict | None = None):
    """Whisper-style words; `probs` maps a token position to its probability."""
    out, t = [], start
    for n, tok in enumerate(text.split()):
        out.append(Word(t, t + per_word, " " + tok, (probs or {}).get(n, 0.99)))
        t += per_word + 0.05
    return out


def reply(edits) -> io.BytesIO:
    content = edits if isinstance(edits, str) else json.dumps({"edits": edits})
    return io.BytesIO(json.dumps({"message": {"role": "assistant", "content": content}}).encode())


class Flagging(unittest.TestCase):
    def test_low_prob(self):
        ws = W("is she dead Potter", probs={2: 0.28})
        self.assertEqual(C.flag_words(ws, 0.6, ["Harry Potter"]), {2})

    def test_capitalised_mid_sentence_unknown_name(self):
        ws = W("No, no, Patrona! Don't put away your wand, Harry.")
        self.assertEqual(C.flag_words(ws, 0.6, ["Harry Potter"]), {2})
        # Without the names list Harry is flagged too; sentence starts never are.
        self.assertEqual(C.flag_words(ws, 0.6), {2, 8})

    def test_caps_exceptions(self):
        # Titles, "I", "God", initials, and the name after "Mrs." is still checked.
        ws = W("Oh my God. I think Mrs. Figg saw Big D. and Mr. Smith")
        flags = {ws[i].text.strip() for i in C.flag_words(ws, 0.6, ["Arabella Figg"])}
        self.assertEqual(flags, {"Big", "Smith"})

    def test_repeated_name_is_learned(self):
        ws = W("come on, Dudley. go on, Dudley. run, Dudley, run")
        self.assertEqual(C.flag_words(ws, 0.6), set())

    def test_word_loop(self):
        ws = W("no no no no stop")
        self.assertEqual(C.flag_words(ws, 0.6), {0, 1, 2, 3})
        self.assertEqual(C.flag_words(W("no, no! stop"), 0.6), set())

    def test_punctuation_only_never_flagged(self):
        ws = [Word(0, 0.1, " ...", 0.01), Word(0.2, 0.4, " yes", 0.99)]
        self.assertEqual(C.flag_words(ws, 0.6), set())


class Windows(unittest.TestCase):
    def speech(self, n_lines: int, flag_lines=()):
        words, flags, t = [], set(), 0.0
        for li in range(n_lines):
            line = W(f"this is line number {li} of the film.", start=t)
            if li in flag_lines:
                flags.add(len(words) + 1)
            words += line
            t = line[-1].end + 0.2
        return words, flags

    def test_one_flag_one_window_with_context(self):
        ws, flags = self.speech(40, {20})
        (win,) = C.plan_windows(ws, flags)
        self.assertEqual(len(win.lines), C.WINDOW_LINES)
        self.assertEqual(len(win.before), 2)
        self.assertEqual(len(win.after), 2)
        self.assertEqual(win.flags, sorted(flags))

    def test_near_flags_merge(self):
        ws, flags = self.speech(60, {10, 14})
        (win,) = C.plan_windows(ws, flags)
        self.assertLessEqual(len(win.lines), C.WINDOW_MAX_LINES)
        self.assertEqual(win.flags, sorted(flags))

    def test_far_flags_split_and_cover_every_flag_once(self):
        ws, flags = self.speech(200, {5, 6, 50, 51, 120, 190})
        wins = C.plan_windows(ws, flags)
        self.assertEqual(len(wins), 4)
        got = [f for w in wins for f in w.flags]
        self.assertEqual(sorted(got), sorted(flags))
        for w in wins:
            self.assertLessEqual(len(w.lines), C.WINDOW_MAX_LINES)

    def test_dense_flags_never_overlap(self):
        ws, flags = self.speech(100, set(range(0, 100, 3)))
        wins = C.plan_windows(ws, flags)
        got = [f for w in wins for f in w.flags]
        self.assertEqual(sorted(got), sorted(flags))
        self.assertEqual(len(got), len(set(got)))

    def test_budget_trims_then_splits(self):
        ws, flags = self.speech(40, {15, 18})
        budget = 400
        wins = C.plan_windows(ws, flags, budget_chars=budget)
        self.assertEqual(sorted(f for w in wins for f in w.flags), sorted(flags))
        for w in wins:
            size = sum(C._line_chars(ws, r) for r in w.before + w.lines + w.after)
            self.assertLessEqual(size, budget)

    def test_real_budget_on_prompt(self):
        ws, flags = self.speech(300, set(range(0, 300, 2)))
        fixed = len(C.SYSTEM_PROMPT) + len(C._header(HP)) + 300
        for w in C.plan_windows(ws, flags, budget_chars=C.TOKEN_BUDGET * 4 - fixed):
            body = C.build_request(ws, w, HP, model="m")
            chars = sum(len(m["content"]) for m in body["messages"])
            self.assertLessEqual(chars / C.CHARS_PER_TOKEN, C.TOKEN_BUDGET)


class Prompt(unittest.TestCase):
    def setUp(self):
        self.ws = W("No, no, Patrona! Mrs. Figg.", probs={2: 0.6})
        self.win = C.plan_windows(self.ws, {2})[0]

    def test_shape(self):
        body = C.build_request(self.ws, self.win, HP, model="qwen3.5:4b")
        self.assertEqual(body["model"], "qwen3.5:4b")
        self.assertIs(body["stream"], False)
        self.assertIs(body["think"], False)
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertNotIn("keep_alive", body)
        sysmsg, user = body["messages"]
        self.assertEqual(sysmsg["role"], "system")
        self.assertIn("Title: Harry Potter and the Order of the Phoenix (2007)", user["content"])
        self.assertIn("Names: Harry Potter, Albus Dumbledore", user["content"])
        self.assertIn("[1:Patrona]!", user["content"])
        edits = body["format"]["properties"]["edits"]
        self.assertEqual(edits["maxItems"], 1)
        self.assertEqual(edits["items"]["properties"]["i"]["maximum"], 1)

    def test_last_request_frees_vram(self):
        body = C.build_request(self.ws, self.win, None, model="m", last=True)
        self.assertEqual(body["keep_alive"], 0)
        self.assertNotIn("Title:", body["messages"][1]["content"])

    def test_episode_header(self):
        ctx = MediaContext(title="Breaking Bad", year=2008, season=2, episode=3, episode_title="X")
        self.assertIn('Breaking Bad (2008), S02E03 "X"', C._header(ctx))


class Acceptance(unittest.TestCase):
    def test_patronum(self):
        self.assertTrue(C.judge_edit(" Patrona!", "Patronum").ok)  # close spelling
        v = C.judge_edit(" Patrona!", "Patronum", names=["Patronum"])
        self.assertTrue(v.ok)
        self.assertTrue(v.reason.startswith("context name"))

    def test_context_name_still_needs_to_be_close(self):
        self.assertTrue(C.judge_edit(" Fig?", "Figg", names=HP.names).ok)
        # Expanding to the full name changes what was said.
        self.assertFalse(C.judge_edit(" Fig?", "Arabella Figg", names=HP.names).ok)

    def test_live_regression_name_spray_rejected(self):
        # qwen3.5:4b on the OotP clip (2026-09-23) proposed all of these.
        names = ["Dementors", "Harry Potter", "Dudley Dursley"]
        for orig in (" dead,", " What", " We're", " getting", " Let's", " No,"):
            self.assertFalse(C.judge_edit(orig, "Dementors", names=names).ok, orig)

    def test_not_close(self):
        self.assertFalse(C.judge_edit(" Dumbledore", "Gandalf").ok)
        v = C.judge_edit(" Dumbledore", "Gandalf", names=HP.names)
        self.assertEqual((v.ok, v.reason), (False, "original is a known name"))
        self.assertFalse(C.judge_edit(" road?", "own").ok)

    def test_sound_alike_via_metaphone(self):
        self.assertEqual(C.metaphone("Knight"), C.metaphone("night"))
        # Spelled too differently for Levenshtein (> 0.5), same Metaphone key.
        for orig, rep in (("Phoebe", "Feeby"), ("Knight", "Nite")):
            v = C.judge_edit(" " + orig, rep)
            self.assertTrue(v.ok and v.reason.startswith("same sound"), v)

    def test_casing_and_punctuation_only(self):
        for rep in ("Patrona.", "patrona", "PATRONA!", " Patrona "):
            v = C.judge_edit(" Patrona!", rep)
            self.assertEqual((v.ok, v.reason), (False, "casing/punctuation only"), rep)

    def test_shape_rules(self):
        self.assertEqual(C.judge_edit(" a", "in little whinging now").reason, "4 words")
        self.assertEqual(C.judge_edit(" a", "in\nLittle").reason, "newline")
        self.assertEqual(C.judge_edit(" a", "").reason, "empty")
        self.assertEqual(C.judge_edit(" a", 7).reason, "not a string")
        self.assertEqual(C.judge_edit(" Patrona", "Patronum {x}").reason, "odd characters")

    def test_metaphone_basics(self):
        self.assertEqual(C.metaphone("Thompson"), "0MPSN")
        self.assertEqual(C.metaphone("Dumbledore"), "TMBLTR")
        self.assertEqual(C.metaphone("Austin"), "ASTN")
        self.assertEqual(C.metaphone(""), "")


class Timing(unittest.TestCase):
    def test_single_word_keeps_span_and_punctuation(self):
        ws = W("No, no, Patrona! now", probs={2: 0.6})
        out = C.apply_edits(ws, {2: "Patronum"})
        self.assertEqual(out[2].text, " Patronum!")
        self.assertEqual((out[2].start, out[2].end, out[2].prob), (ws[2].start, ws[2].end, 1.0))
        self.assertEqual(out[3], ws[3])

    def test_multi_word_split_by_characters(self):
        ws = [Word(1.0, 1.1, " Dementors,"), Word(1.2, 1.3, " a"), Word(2.0, 3.0, " whinging.")]
        out = C.apply_edits(ws, {2: "Little Whinging"})
        self.assertEqual([w.text for w in out[2:]], [" Little", " Whinging."])
        self.assertAlmostEqual(out[2].start, 2.0)
        self.assertAlmostEqual(out[2].end, 2.0 + 6 / 14)
        self.assertAlmostEqual(out[3].start, out[2].end)
        self.assertAlmostEqual(out[3].end, 3.0)

    def test_sentence_start_capitalised_mid_sentence_not(self):
        ws = W("Hello. austin you? Dumbledore, Austin?")
        out = C.apply_edits(ws, {1: "asked", 4: "asked you"})
        self.assertEqual(out[1].text, " Asked")
        self.assertEqual([w.text for w in out[4:]], [" asked", " you?"])


class Driver(unittest.TestCase):
    def setUp(self):
        # Two windows' worth: flags far apart.
        self.ws = W("No, no, Patrona! Mrs. Figg.", probs={2: 0.5})
        for n in range(40):
            self.ws += W(f"filler line {n} goes here.", start=10 + n * 3)
        self.ws += W("Dumbledore, Austin? you know him.", start=200, probs={1: 0.22})
        self.flags = sorted(C.flag_words(self.ws, 0.6, HP.names))

    def run_correct(self, replies, **kw):
        calls = []

        def fake(req, timeout):
            calls.append(json.loads(req.data))
            r = replies.pop(0)
            if isinstance(r, BaseException):
                raise r
            return r

        with mock.patch("urllib.request.urlopen", side_effect=fake):
            out, stats = C.correct(self.ws, HP, url="http://o:11434", model="m", **kw)
        return out, stats, calls

    def test_happy_path(self):
        out, stats, calls = self.run_correct(
            [
                reply([{"i": 1, "text": "Patronum"}]),
                reply([{"i": 1, "text": "Gandalf"}]),
            ]
        )
        self.assertEqual(len(calls), 2)
        self.assertNotIn("keep_alive", calls[0])
        self.assertEqual(calls[1]["keep_alive"], 0)
        self.assertEqual(out[2].text, " Patronum!")
        self.assertIn(" Austin?", [w.text for w in out])
        self.assertEqual((stats["accepted"], stats["rejected"], stats["errors"]), (1, 1, 0))
        self.assertEqual(stats["reasons"], {"not close": 1})

    def test_out_of_range_and_duplicate_index(self):
        _, stats, _ = self.run_correct(
            [
                reply([{"i": 9, "text": "x"}, {"i": 0, "text": "y"}, {"i": True, "text": "z"}]),
                reply([{"i": 1, "text": "Austen"}, {"i": 1, "text": "Austria"}]),
            ]
        )
        self.assertEqual(stats["accepted"], 1)
        self.assertEqual(stats["reasons"], {"index not flagged": 3, "duplicate index": 1})

    def test_garbage_fails_open(self):
        for bad in ("not json", '{"edits": [{"i": 1, "text": "Patr', '{"nope": 1}', "[]"):
            out, stats, _ = self.run_correct([reply(bad), reply(bad)])
            self.assertEqual(out, self.ws, bad)
            self.assertEqual(stats["errors"], 2)
        # A reply without message / not JSON at all.
        out, stats, _ = self.run_correct([io.BytesIO(b"{}"), io.BytesIO(b"<html>")])
        self.assertEqual((out, stats["errors"]), (self.ws, 2))

    def test_unreachable_stops_and_keeps_text(self):
        # Force 4 windows; after 2 connection errors the rest aren't sent.
        ws = self.ws
        for n in range(80):
            ws = ws + W(f"more filler {n} here.", start=300 + n * 3)
        ws += W("a Grangr moment.", start=600)
        ws += W("filler.", start=700) * 1
        self.ws = ws
        err = urllib.error.URLError(ConnectionRefusedError(111, "refused"))
        out, stats, calls = self.run_correct([err] * 10)
        self.assertEqual(out, self.ws)
        self.assertEqual(stats["stopped"], "unreachable")
        # 2 failed chat requests, then 1 best-effort unload (also fails, swallowed).
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[-1], {"model": "m", "messages": [], "keep_alive": 0})

    def test_timeout_fails_open(self):
        out, stats, _ = self.run_correct([TimeoutError("slow"), reply([])])
        self.assertEqual(out, self.ws)
        self.assertEqual(stats["errors"], 1)

    def test_cancel_between_requests(self):
        cancel = threading.Event()
        replies = [reply([{"i": 1, "text": "Patronum"}])]

        def fake(req, timeout):
            cancel.set()
            return replies.pop(0) if replies else io.BytesIO(b"{}")

        with mock.patch("urllib.request.urlopen", side_effect=fake) as m:
            out, stats = C.correct(self.ws, HP, url="http://o", model="m", cancel=cancel)
        self.assertEqual(stats["stopped"], "cancelled")
        self.assertEqual(m.call_count, 2)  # one window + the unload
        self.assertEqual(out[2].text, " Patronum!")

    def test_budget(self):
        ticks = iter([0.0, 0.0, 1000.0, 1000.0, 1000.0])
        out, stats, calls = self.run_correct(
            [reply([]), reply([])], budget_s=900, clock=lambda: next(ticks)
        )
        self.assertEqual(stats["stopped"], "budget")
        self.assertEqual(len(calls), 2)  # one window + the unload
        self.assertEqual(calls[-1]["keep_alive"], 0)

    def test_edit_cap(self):
        ws = W("so")  # every Zork is mid-sentence, so all 10 are flagged
        for n in range(10):
            ws += W(f"Zork{n} said", start=1 + n * 0.8)
        with mock.patch.object(C, "MAX_EDITS_PER_WINDOW", 3):
            with mock.patch(
                "urllib.request.urlopen",
                return_value=reply([{"i": k, "text": f"Zorg{k - 1}"} for k in range(1, 11)]),
            ):
                out, stats = C.correct(ws, None, url="http://o", model="m")
        self.assertEqual(stats["accepted"], 3)
        self.assertEqual(stats["reasons"]["window edit cap"], 7)


class PathParsing(unittest.TestCase):
    def test_movie(self):
        c = parse_path("/media/movies/Fargo (1996) [imdbid-tt0116282]/Fargo (1996) - 1080p.mkv")
        self.assertEqual((c.title, c.year, c.is_episode), ("Fargo", 1996, False))

    def test_movie_year_from_dir_and_scene_name(self):
        self.assertEqual(parse_path("/m/Snatch (2000)/snatch.mkv").year, 2000)
        c = parse_path("/m/Snatch.2000.1080p.BluRay.x264.mkv")
        self.assertEqual((c.title, c.year), ("Snatch", 2000))

    def test_episode(self):
        c = parse_path("/tv/Breaking Bad (2008)/Season 2/Breaking Bad - S02E03 - Bit by a.mkv")
        self.assertEqual(
            (c.title, c.year, c.season, c.episode, c.episode_title),
            ("Breaking Bad", 2008, 2, 3, "Bit by a"),
        )

    def test_scene_episode(self):
        c = parse_path("/tv/The Office (US)/Season 01/The.Office.US.S01E02.720p.x264.mkv")
        self.assertEqual((c.title, c.season, c.episode), ("The Office (US)", 1, 2))

    def test_never_raises(self):
        self.assertIsInstance(parse_path(None), MediaContext)  # type: ignore[arg-type]


class Jellyfin(unittest.TestCase):
    VIDEO = "/media/movies/HP5 (2007)/HP5 (2007).mkv"

    def setUp(self):
        ctxmod._cache.clear()
        ctxmod._series_cache.clear()

    def body(self, obj):
        return io.BytesIO(json.dumps(obj).encode())

    def test_movie_people_by_path(self):
        items = {
            "Items": [
                {"Name": "HP5", "Path": "/other/HP5.mkv", "ProductionYear": 2007},
                {
                    "Name": "HP5",
                    "Path": self.VIDEO,
                    "ProductionYear": 2007,
                    "Overview": "Back at Hogwarts, Harry faces Umbridge.",
                    "People": [
                        {"Name": "Daniel Radcliffe", "Role": "Harry Potter", "Type": "Actor"},
                        {"Name": "X", "Role": "Death Eater #2", "Type": "Actor"},
                        {"Name": "Y", "Role": "Himself", "Type": "Actor"},
                        {"Name": "D", "Role": "", "Type": "Director"},
                        {"Name": "Z", "Role": "Dudley Dursley (voice)", "Type": "Actor"},
                    ],
                },
            ]
        }
        with mock.patch("urllib.request.urlopen", return_value=self.body(items)) as m:
            c = media_context(self.VIDEO, jellyfin_url="http://jf:8096/", jellyfin_api_key="k")
        req = m.call_args.args[0]
        self.assertIn("searchTerm=HP5", req.full_url)
        self.assertIn("years=2007", req.full_url)
        self.assertEqual(req.get_header("Authorization"), 'MediaBrowser Token="k"')
        self.assertEqual(c.source, "jellyfin")
        self.assertEqual(c.names, ("Harry Potter", "Dudley Dursley"))
        self.assertEqual(c.terms, ("Hogwarts", "Harry", "Umbridge"))
        # Cached per job/path: no second request.
        with mock.patch("urllib.request.urlopen") as m2:
            media_context(self.VIDEO, jellyfin_url="http://jf:8096/", jellyfin_api_key="k")
        m2.assert_not_called()

    def test_episode(self):
        video = "/media/tv/Show (2010)/Season 1/Show - S01E02 - Pilot.mkv"
        series = {"Items": [{"Id": "abc", "Name": "Show", "Path": "/media/tv/Show (2010)"}]}
        series["Items"][0]["People"] = [{"Role": "Walter White", "Type": "Actor"}]
        eps = {
            "Items": [
                {"IndexNumber": 1, "People": [{"Role": "Wrong", "Type": "GuestStar"}]},
                {"IndexNumber": 2, "People": [{"Role": "Tuco", "Type": "GuestStar"}]},
            ]
        }
        with mock.patch(
            "urllib.request.urlopen", side_effect=[self.body(series), self.body(eps)]
        ) as m:
            c = media_context(video, jellyfin_url="http://jf", jellyfin_api_key="k")
        self.assertIn("/Shows/abc/Episodes?season=1", m.call_args.args[0].full_url)
        self.assertEqual(c.names, ("Tuco", "Walter White"))

    def test_no_key_no_request_and_errors_never_raise(self):
        with mock.patch("urllib.request.urlopen") as m:
            c = media_context(self.VIDEO)
        m.assert_not_called()
        self.assertEqual((c.title, c.year, c.source), ("HP5", 2007, "path"))
        ctxmod._cache.clear()
        for exc in (urllib.error.URLError("down"), ValueError("bad json"), KeyError("x")):
            with mock.patch("urllib.request.urlopen", side_effect=exc):
                c = media_context(self.VIDEO, jellyfin_url="http://jf", jellyfin_api_key="k")
            self.assertEqual((c.title, c.source), ("HP5", "path"))
            ctxmod._cache.clear()


if __name__ == "__main__":
    unittest.main()

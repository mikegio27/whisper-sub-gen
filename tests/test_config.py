import unittest
from datetime import time
from unittest import mock

from pydantic import ValidationError

from app.config import Settings, in_work_window, settings


class InWorkWindowTest(unittest.TestCase):
    def check(self, window: str, now: time) -> bool:
        with mock.patch.object(settings, "work_window", window):
            return in_work_window(now)

    def test_no_window_always_allowed(self):
        self.assertTrue(self.check("", time(3, 0)))
        self.assertTrue(self.check("", time(15, 0)))

    def test_same_day_window(self):
        w = "09:00-17:00"
        self.assertFalse(self.check(w, time(8, 59)))
        self.assertTrue(self.check(w, time(9, 0)))  # start inclusive
        self.assertTrue(self.check(w, time(12, 30)))
        self.assertFalse(self.check(w, time(17, 0)))  # end exclusive
        self.assertFalse(self.check(w, time(23, 0)))

    def test_window_across_midnight(self):
        w = "22:00-06:00"
        self.assertFalse(self.check(w, time(21, 59)))
        self.assertTrue(self.check(w, time(22, 0)))
        self.assertTrue(self.check(w, time(23, 59)))
        self.assertTrue(self.check(w, time(0, 0)))
        self.assertTrue(self.check(w, time(5, 59)))
        self.assertFalse(self.check(w, time(6, 0)))
        self.assertFalse(self.check(w, time(12, 0)))

    def test_single_digit_hours(self):
        self.assertTrue(self.check("1:00-5:30", time(5, 29)))
        self.assertFalse(self.check("1:00-5:30", time(5, 30)))


class SettingsTest(unittest.TestCase):
    def test_bad_work_window_rejected(self):
        with self.assertRaises(ValidationError):
            Settings(work_window="10pm-6am")

    def test_bad_modes_rejected(self):
        with self.assertRaises(ValidationError):
            Settings(run_mode="sometimes")
        with self.assertRaises(ValidationError):
            Settings(shutdown_mode="later")

    def test_modes_normalised(self):
        s = Settings(run_mode=" Manual ", shutdown_mode="FINISH")
        self.assertEqual(s.run_mode, "manual")
        self.assertEqual(s.shutdown_mode, "finish")

    def test_list_parsing(self):
        s = Settings(media_dirs=" /a, ,/b ", video_extensions=".MKV, mp4,,")
        self.assertEqual(s.media_dir_list, ["/a", "/b"])
        self.assertEqual(s.extension_set, {"mkv", "mp4"})

    def test_compute_type_auto(self):
        self.assertEqual(Settings(whisper_device="cpu").compute_type, "int8")
        self.assertEqual(Settings(whisper_device="cuda").compute_type, "float16")
        s = Settings(whisper_device="cuda", whisper_compute_type="int8_float16")
        self.assertEqual(s.compute_type, "int8_float16")


if __name__ == "__main__":
    unittest.main()

"""Memory hygiene between jobs: cgroup limit parsing and the clean-restart guard."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import memory


class LimitTest(unittest.TestCase):
    def read_limit(self, v2: str | None, v1: str | None = None) -> int:
        with tempfile.TemporaryDirectory() as d:
            p2, p1 = Path(d) / "v2", Path(d) / "v1"
            if v2 is not None:
                p2.write_text(v2)
            if v1 is not None:
                p1.write_text(v1)
            with (
                mock.patch.object(memory, "_CGROUP_V2", p2),
                mock.patch.object(memory, "_CGROUP_V1", p1),
            ):
                return memory.limit_bytes()

    def test_cgroup_v2(self):
        self.assertEqual(self.read_limit("12884901888\n"), 12 * 2**30)
        self.assertEqual(self.read_limit("max\n"), 0)

    def test_cgroup_v1_and_none(self):
        self.assertEqual(self.read_limit(None, "4294967296"), 4 * 2**30)
        self.assertEqual(self.read_limit(None, str(2**63 - 4096)), 0)  # v1 "unlimited"
        self.assertEqual(self.read_limit(None, None), 0)

    def test_over_budget(self):
        with (
            mock.patch.object(memory, "rss_bytes", return_value=9 * 2**30),
            mock.patch.object(memory, "limit_bytes", return_value=12 * 2**30),
        ):
            self.assertTrue(memory.over_budget(0.7)[0])
            self.assertFalse(memory.over_budget(0.8)[0])
            self.assertFalse(memory.over_budget(0)[0])  # disabled
        with (
            mock.patch.object(memory, "rss_bytes", return_value=9 * 2**30),
            mock.patch.object(memory, "limit_bytes", return_value=0),
        ):
            self.assertFalse(memory.over_budget(0.7)[0])  # no known limit: never

    def test_release_and_rss_are_safe_to_call(self):
        memory.release()
        self.assertGreater(memory.rss_bytes(), 0)

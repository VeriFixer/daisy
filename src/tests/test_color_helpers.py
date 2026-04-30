"""Tests for color output helpers in single_file_run.py."""

import unittest

import single_file_run as sfr


class TestColorHelpers(unittest.TestCase):
    """Verify ANSI wrapping and --no-color suppression."""

    def setUp(self):
        self._orig = sfr._USE_COLOR

    def tearDown(self):
        sfr._USE_COLOR = self._orig

    # --- color enabled ---

    def test_color_wraps_text(self):
        sfr._USE_COLOR = True
        result = sfr.color("hello", "31")
        self.assertEqual(result, "\033[31mhello\033[0m")

    def test_header_bold_cyan(self):
        sfr._USE_COLOR = True
        result = sfr.header("Title")
        self.assertIn("\033[1;36m", result)
        self.assertIn("Title", result)

    def test_success_green(self):
        sfr._USE_COLOR = True
        result = sfr.success("ok")
        self.assertEqual(result, "\033[32mok\033[0m")

    def test_error_red(self):
        sfr._USE_COLOR = True
        result = sfr.error("fail")
        self.assertEqual(result, "\033[31mfail\033[0m")

    def test_warning_yellow(self):
        sfr._USE_COLOR = True
        result = sfr.warning("warn")
        self.assertEqual(result, "\033[33mwarn\033[0m")

    def test_section_header_format(self):
        sfr._USE_COLOR = True
        result = sfr.section_header("Verification")
        self.assertIn("── Verification ──", result)
        self.assertIn("\033[1;36m", result)

    # --- color disabled (--no-color) ---

    def test_color_disabled_returns_plain(self):
        sfr._USE_COLOR = False
        self.assertEqual(sfr.color("hello", "31"), "hello")

    def test_header_no_color(self):
        sfr._USE_COLOR = False
        self.assertEqual(sfr.header("Title"), "Title")

    def test_success_no_color(self):
        sfr._USE_COLOR = False
        self.assertEqual(sfr.success("ok"), "ok")

    def test_error_no_color(self):
        sfr._USE_COLOR = False
        self.assertEqual(sfr.error("fail"), "fail")

    def test_warning_no_color(self):
        sfr._USE_COLOR = False
        self.assertEqual(sfr.warning("warn"), "warn")

    def test_section_header_no_color(self):
        sfr._USE_COLOR = False
        result = sfr.section_header("Verification")
        self.assertEqual(result, "── Verification ──")
        self.assertNotIn("\033[", result)

    # --- edge cases ---

    def test_empty_string(self):
        sfr._USE_COLOR = True
        self.assertEqual(sfr.color("", "31"), "\033[31m\033[0m")

    def test_empty_string_no_color(self):
        sfr._USE_COLOR = False
        self.assertEqual(sfr.color("", "31"), "")


if __name__ == "__main__":
    unittest.main()

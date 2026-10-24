"""0.2.3 RC stress test: #11 (A01b/A05) an install from an older feed shows the real reason with both sequence numbers, the folder that holds the
old state and one next step."""

import unittest

from strata_control import feed as feedmod, updater
from tests.terminal import test_updater


class ReplayedFeedWords(test_updater.UpdaterCase):
    def test_the_refusal_gives_both_numbers_the_folder_and_one_next_step(self):
        self.publish(("0.1.46", test_updater.fx.GOOD_ENGINE), sequence=108)
        updater.check(self.ctx())                                      # accepts sequence 108 and remembers it
        self.publish(("0.1.46", test_updater.fx.GOOD_ENGINE), sequence=107)
        with self.assertRaises(feedmod.FeedDowngrade) as caught:
            updater.check(self.ctx())
        text = str(caught.exception)
        self.assertIn("older than the one already accepted", text)
        self.assertIn("its sequence is 107", text)
        self.assertIn("the accepted one is 108", text)
        self.assertIn(str(self.components / ".updates"), text)
        self.assertIn("state.json", text)
        self.assertEqual(text.count("Use the newest release folder"), 1)


if __name__ == "__main__":
    unittest.main()

import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from make_features import _make_template

class TestFeatures(unittest.TestCase):
    def test_block_id_regex(self):
        # Case 1: Standard block ID
        line = "Receiving block blk_1608999687919862906 src: /10.250.19.102:54106"
        template = _make_template(line)
        self.assertIn("blk_<ID>", template)
        self.assertNotIn("blk_-<NUM>", template)

    def test_negative_block_id(self):
        # Case 2: Negative block ID (The bug)
        line = "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106"
        template = _make_template(line)
        # Should be unified
        self.assertIn("blk_<ID>", template)
        # Should NOT be split into blk_-<NUM>
        self.assertNotIn("blk_-<NUM>", template)

if __name__ == "__main__":
    unittest.main()

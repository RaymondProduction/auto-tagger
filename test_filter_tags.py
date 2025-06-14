import unittest
from auto_tagger import filter_tags_by_category, MAX_TAG_EDIT_DISTANCE
from tag_categories import CATEGORY_KEYWORDS

class TestFilterTagsByCategory(unittest.TestCase):

    def test_bg_category(self):
        tags = ["scenery", "blue sky", "forest", "pizza"]
        expected_filtered = {"scenery", "forest"}
        expected_unknown = {"pizza", "blue sky"}

        #Imitate argparse argument
        class Args:
            filter_category = "bg"
        global args
        args = Args()

        filtered, unknown = filter_tags_by_category(tags)
        self.assertEqual(set(filtered), expected_filtered)
        self.assertEqual(set(unknown), expected_unknown)

    def test_pose_category_with_typos(self):
        tags = ["sittting", "stradding", "pizza"]
        class Args:
            filter_category = "pose"
        global args
        args = Args()

        filtered, unknown = filter_tags_by_category(tags)
        self.assertIn("sittting", filtered)
        self.assertIn("stradding", filtered)
        self.assertIn("pizza", unknown)

    def test_nsfw_category(self):
        tags = ["nude", "explicit", "classroom", "banana"]
        class Args:
            filter_category = "nsfw"
        global args
        args = Args()

        filtered, unknown = filter_tags_by_category(tags)
        self.assertEqual(set(filtered), {"nude", "explicit"})
        self.assertEqual(set(unknown), {"classroom", "banana"})

if __name__ == "__main__":
    unittest.main()

import unittest

from crop_fit import preview_size, roi_to_original


class CropFitTest(unittest.TestCase):
    def test_large_image_fits_screen(self):
        pw, ph = preview_size(3000, 1800, 1649, 928)
        self.assertLessEqual(pw, int(1649 * .82))
        self.assertLessEqual(ph, int(928 * .65))
        self.assertAlmostEqual(pw / ph, 3000 / 1800, delta=.01)

    def test_small_image_not_upscaled(self):
        self.assertEqual(preview_size(320, 200, 1649, 928), (320, 200))

    def test_coordinates_map_to_full_resolution(self):
        self.assertEqual(roi_to_original((100, 80, 200, 160), (600, 400), (3000, 2000)), (500, 400, 1000, 800))

    def test_cancel(self):
        self.assertEqual(roi_to_original((0, 0, 0, 0), (600, 400), (3000, 2000)), (0, 0, 0, 0))


if __name__ == '__main__':
    unittest.main()

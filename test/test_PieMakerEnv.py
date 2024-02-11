from unittest import TestCase

from src.PieMakerEnv import PieGameEnv


class TestPieGameEnv(TestCase):
    def test_init_success(self):
        source_plates = ["01010000",
                         "00200200",
                         "00030003",
                         "00004000"]
        pge = PieGameEnv(s=4, t=3, c=4, n=8, p=0.249, source_plates_strs=source_plates)
        self.assertIsInstance(pge, PieGameEnv, msg="env initialized successfully")

    def test_init_raises_on_invalid_pfr(self):
        source_plates = ["01010000",
                         "00200200",
                         "00030003",
                         "00004000"]
        with self.assertRaises(AssertionError):
            pge = PieGameEnv(s=4, t=3, c=4, n=8, p=0.31, source_plates_strs=source_plates)

    def test_init_raises_on_invalid_c_in_source(self):
        source_plates = ["01010000",
                         "00200200",
                         "00030003",
                         "00004000"]
        with self.assertRaises(AssertionError):
            pge = PieGameEnv(s=4, t=3, c=4, n=8, p=0.249, source_plates_strs=source_plates)


class TestPieGameEnv(TestCase):
    def test__is_valid_transfer(self):
        self.fail()

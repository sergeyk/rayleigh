from context import *
from shutil import copy


class TestPalette(unittest.TestCase):
    def test_create_palette(self):
        dirname = rayleigh.util.makedirs(os.path.join(temp_dirname, 'palette'))
        for num_hues in [11, 10, 9, 8]:
            for val in [2, 3]:
                sat_range = light_range = val
                palette = rayleigh.Palette(num_hues, sat_range, light_range)
                palette.output(dirname)
                copy(
                    os.path.join(dirname, 'palette.png'),
                    os.path.join(dirname, 'palette_{}_{}.png'.format(num_hues, val)))


if __name__ == '__main__':
    unittest.main()

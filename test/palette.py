from context import *
from shutil import copy


class TestPalette(unittest.TestCase):
    def test_create_palette(self):
        dirname = skutil.makedirs(os.path.join(temp_dirname, 'palette'))
        for num_hues in [8, 12]:
            for val in [1, 2]:
                sat_range = light_range = val
                palette = rayleigh.Palette(num_hues, sat_range, light_range)
                height = 1 + 1 + sat_range + light_range
                assert(len(palette.hex_list) == height * num_hues)
                palette.output(dirname)
                copy(
                    os.path.join(dirname, 'palette.png'),
                    os.path.join(dirname, 'palette_{}_{}.png'.format(num_hues, val)))


if __name__ == '__main__':
    unittest.main()

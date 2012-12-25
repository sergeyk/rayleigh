from context import *
from shutil import copy


class TestPalette(unittest.TestCase):
    def test_create_palette(self):
        dirname = skutil.makedirs(os.path.join(temp_dirname, 'palette'))
        for num_hues in [12, 8]:
            for val in [1, 2]:
                sat_range = light_range = val
                palette = rayleigh.Palette(num_hues, sat_range, light_range)
                height = 1 + sat_range + light_range
                palette.output(dirname)
                copy(
                    os.path.join(dirname, 'palette.png'),
                    os.path.join(dirname, 'palette_{}_{}.png'.format(num_hues, val)))
                assert(len(palette.hex_list) == height * (num_hues + 1))


if __name__ == '__main__':
    unittest.main()

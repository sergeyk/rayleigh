from context import *


class TestPalette(unittest.TestCase):
    def test_create_palette(self):
        for num_hues in [7, 13]:
            for val in [0, 1, 2]:
                sat_range = light_range = val
                palette = rayleigh.Palette(num_hues, sat_range, light_range)
                height = 1 + 1 + sat_range + light_range
                assert(len(palette.hex_list) == height * num_hues)

        dirname = skutil.makedirs(os.path.join(temp_dirname, 'palette'))
        palette.output(dirname)
        assert(os.path.exists(os.path.join(dirname, 'palette.png')))

if __name__ == '__main__':
    unittest.main()

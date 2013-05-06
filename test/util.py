from context import *


class TestUtil(unittest.TestCase):

    def test_histogram(self):
        dirname = rayleigh.util.makedirs(os.path.join(temp_dirname, 'util'))

        num_hues = 7
        sat_range = light_range = 2
        palette = rayleigh.Palette(num_hues, sat_range, light_range)

        fname = save_synthetic_image(palette.hex_list[0], dirname, size=100)
        img = rayleigh.Image(fname)
        
        sigma = 20
        hist_fname = fname + '_hist_sigma_{}.png'.format(sigma)
        quant_fname = fname + '_quant.png'
        hist = img.histogram_colors(palette, sigma, hist_fname)
        img.quantize_to_palette(palette, quant_fname)

        fname_big = save_synthetic_image(palette.hex_list[0], dirname, size=333)
        img_big = rayleigh.Image(fname_big)

        hist_fname_big = fname + '_big_hist_sigma_{}.png'.format(sigma)
        quant_fname_big = fname + '_big_quant.png'
        hist_big = img_big.histogram_colors(palette, sigma, hist_fname_big)
        img_big.quantize_to_palette(palette, quant_fname_big)

        assert_almost_equal(hist, hist_big)

if __name__ == '__main__':
    unittest.main()

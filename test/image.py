"""
Test the image collection methods.
"""
from context import *


class TestImage(unittest.TestCase):

    def test_histogram(self):
        # Create a palette and save it to file
        num_hues = 7
        sat_range = light_range = 2
        palette = rayleigh.Palette(num_hues, sat_range, light_range)
        dirname = skutil.makedirs(os.path.join(temp_dirname, 'image'))
        palette.output(dirname)
        palette_filename = os.path.join(dirname, 'palette.png')
        assert(os.path.exists(palette_filename))

        # Load the palette as an Image and
        # - plot its histogram of its own palette
        # - output quantized image
        img = rayleigh.Image(palette_filename)
        
        sigma = 20
        fname = palette_filename + '_hist_sigma_{}.png'.format(sigma)
        img.histogram_colors(palette, sigma, fname)
        assert(os.path.exists(fname))

        fname = palette_filename + '_quant.png'
        img.quantize_to_palette(palette, fname)
        assert(os.path.exists(fname))

if __name__ == '__main__':
    unittest.main()

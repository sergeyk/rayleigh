"""
Test the image collection methods.
"""
from context import *


class TestPaletteQuery(unittest.TestCase):

    def test_histogram(self):
        dirname = rayleigh.util.makedirs(os.path.join(temp_dirname, 'palette_query'))
        palette = rayleigh.Palette()
        palette.output(dirname)

        colors = [palette.hex_list[3], palette.hex_list[20]]
        values = np.ones(len(colors))
        pq = rayleigh.PaletteQuery(colors, values)
        
        sigma = 20
        fname = os.path.join(dirname, 'hist_sigma_{}.png'.format(sigma))
        pq.histogram_colors(palette, sigma, fname)
        fname = os.path.join(dirname, 'quantized.png')
        pq.quantize_to_palette(palette, fname)

if __name__ == '__main__':
    unittest.main()

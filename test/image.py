"""
Test the image collection methods.
"""
from context import *
import shutil
from glob import glob


class TestImage(unittest.TestCase):

    def test_histogram(self):
        # Create a palette and save it to file
        num_hues = 7
        sat_range = light_range = 2
        palette = rayleigh.Palette(num_hues, sat_range, light_range)
        dirname = rayleigh.util.makedirs(os.path.join(temp_dirname, 'image'))
        palette.output(dirname)
        palette_filename = os.path.join(dirname, 'palette.png')
        assert(os.path.exists(palette_filename))

        img = rayleigh.Image(palette_filename)

        # Output unsmoothed histogram
        img.histogram_colors(palette, palette_filename + '_hist.png')

        # Test smoothed histogram
        sigma = 20
        fname = palette_filename + '_hist_direct_sigma_{}.png'.format(sigma)
        img.histogram_colors_smoothed(palette, sigma, fname, direct=True)
        fname = palette_filename + '_hist_sigma_{}.png'.format(sigma)
        img.histogram_colors_smoothed(palette, sigma, fname, direct=False)

        fname = palette_filename + '_quant.png'
        img.output_quantized_to_palette(palette, fname)
        assert(os.path.exists(fname))

    def test_flickr(self):
        # TODO: output a single HTML file instead of using the file browser to
        # look at all these files
        dirname = rayleigh.util.makedirs(os.path.join(temp_dirname, 'image_flickr'))
        palette = rayleigh.Palette(num_hues=10, light_range=3, sat_range=2)
        palette.output(dirname)

        image_filenames = glob(os.path.join(support_dirname, 'images', '*'))
        sigmas = [10]
        for ind, img_fname in enumerate(image_filenames):
            for sigma in sigmas:
                output_fname = os.path.join(dirname, str(ind))
                shutil.copy(img_fname, output_fname + '.jpg')
                img = rayleigh.Image(img_fname)

                img.histogram_colors(palette, output_fname + '_hist.png')
                
                fname = output_fname + '_hist_direct_sigma_{}.png'.format(sigma)
                img.histogram_colors_smoothed(palette, sigma, fname, direct=True)

                fname = output_fname + '_hist_sigma_{}.png'.format(sigma)
                img.histogram_colors_smoothed(palette, sigma, fname, direct=False)
                
                img.output_quantized_to_palette(
                    palette, output_fname + '_quantized.png')

                img.output_color_palette_image(
                    palette, output_fname + '_my_palette.png')

if __name__ == '__main__':
    unittest.main()

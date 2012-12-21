"""
Test the image collection methods.
"""
from context import *


class TestCollection(unittest.TestCase):
    def test_synthetic(self):
        dirname = skutil.makedirs(os.path.join(temp_dirname, 'synthetic_colors'))
        palette = rayleigh.Palette()
        filenames = [save_synthetic_image(color, dirname) for color in palette.hex_list]

        # save palette histograms and quantized versions
        sigma = 20
        for filename in filenames:
            img = rayleigh.Image(filename)
        
            # this takes a while, so only do for some
            if np.random.rand() > 0.7:
                fname = filename + '_hist_sigma_{}.png'.format(sigma)
                img.histogram_colors(palette, sigma, fname)

            q_filename = filename + '_quant.png'
            img.quantize_to_palette(palette, q_filename)

    def test_flickr(self):
        """
        Load subset of MIRFLICKR 25K [dataset](http://press.liacs.nl/mirflickr/).

        > find /Volumes/WD\ Data/mirflickr -name "*.jpg" | head -n 100 > mirflickr_100.txt
        """
        #image_list_filename = os.path.join(support_dirname, 'mirflickr_10000.txt')
        #with open(image_list_filename) as f:
        #    image_filenames = [x.strip() for x in f.readlines()]

        #hex_palette = rayleigh.create_palette()
        
        #ic = rayleigh.ImageCollection(hex_palette)
        #ic.add_images(image_filenames)

        #results = ic.search_by_image(image_filenames[0])
        #ic.save('%s.pickle'%image_list_filename)

if __name__ == '__main__':
    unittest.main()

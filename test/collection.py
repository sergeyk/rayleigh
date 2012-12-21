"""
Test the image collection methods.
"""
from context import *

from sklearn.utils import shuffle


class TestCollection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dirname = skutil.makedirs(os.path.join(temp_dirname, 'synthetic_colors'))
        cls.palette = rayleigh.Palette()
        cls.filenames = [save_synthetic_image(color, cls.dirname) for color in cls.palette.hex_list]


    def test_synthetic_creation(self):
        # save palette histograms and quantized versions
        sigma = 20
        n_samples = len(self.filenames) / 5
        s_filenames = shuffle(self.filenames, random_state=0, n_samples=n_samples)
        for filename in s_filenames:
            img = rayleigh.Image(filename)

            fname = filename + '_hist_sigma_{}.png'.format(sigma)
            img.histogram_colors(self.palette, sigma, fname)

            q_filename = filename + '_quant.png'
            img.quantize_to_palette(self.palette, q_filename)


    def test_synthetic_search(self):
        # set up jinja template
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(support_dirname))
        template = env.get_template('matches.html')

        # create a collection and output nearest matches to every color
        ic = rayleigh.ImageCollection(self.palette)
        ic.add_images(self.filenames)

        # search several query images and output to html summary
        matches_filename = os.path.join(self.dirname, 'matches.html')
        data = (ic.search_by_image(fname) for fname in self.filenames)
        # data is a list of (query_img, results) tuples
        with open(matches_filename, 'w') as f:
            f.write(template.render(data=data))


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

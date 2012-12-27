"""
Test the image collection methods.
"""
from context import *

from sklearn.utils import shuffle
from rayleigh import *


class TestSyntheticCollection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dirname = skutil.makedirs(os.path.join(temp_dirname, 'synthetic_colors'))
        cls.palette = rayleigh.Palette()
        cls.filenames = [save_synthetic_image(color, cls.dirname) for color in cls.palette.hex_list]

    def test_synthetic_creation(self):
        # save palette histograms and quantized versions
        sigma = 10
        n_samples = len(self.filenames) / 3
        s_filenames = shuffle(self.filenames, random_state=0, n_samples=n_samples)
        for filename in s_filenames:
            img = rayleigh.Image(filename)

            fname = filename + '_hist_sigma_{}.png'.format(sigma)
            img.histogram_colors_smoothed(self.palette, sigma, fname, direct=False)

            q_filename = filename + '_quant.png'
            img.output_quantized_to_palette(self.palette, q_filename)

    def test_synthetic_search(self):
        # set up jinja template
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(support_dirname))
        template = env.get_template('matches.html')

        # create a collection and output nearest matches to every color
        ic = rayleigh.ImageCollection(self.palette)
        ic.add_images(self.filenames)
        sic = rayleigh.SearchableImageCollectionExact(ic, 'euclidean', 0)

        # search several query images and output to html summary
        matches_filename = os.path.join(self.dirname, 'matches.html')
        data = (sic.search_by_image(fname) for fname in self.filenames)
        # data is a list of (query_img, results) tuples
        with open(matches_filename, 'w') as f:
            f.write(template.render(data=data))


class TestFlickrCollection(unittest.TestCase):
    def test_flickr(self):
        """
        Load subset of MIRFLICKR 25K [dataset](http://press.liacs.nl/mirflickr/).
        > find /Volumes/WD\ Data/mirflickr -name "*.jpg" | head -n 100 > mirflickr_100.txt
        """
        # Parametrization of our test.
        image_list_name = 'mirflickr_100'
        image_list_name = 'mirflickr_1K'
        image_list_name = 'mirflickr_25K'
        dirname = skutil.makedirs(os.path.join(temp_dirname, image_list_name))
        num_queries = 50
        palette = rayleigh.Palette(num_hues=11, sat_range=3, light_range=2)
        palette.output(dirname=dirname)

        # Set up jinja template.
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(support_dirname))
        template = env.get_template('matches.html')
        
        # Construct the list of images in the dataset.
        image_list_filename = os.path.join(
            support_dirname, image_list_name + '.txt')
        with open(image_list_filename) as f:
            image_filenames = [x.strip() for x in f.readlines()]

        # Load the image collection.
        ic_filename = os.path.join(
            temp_dirname, '{}.pickle'.format(image_list_name))

        if os.path.exists(ic_filename):
            print("Loading ImageCollection from cache.")
            ic = rayleigh.ImageCollection.load(ic_filename)
        else:
            ic = rayleigh.ImageCollection(palette)
            ic.add_images(image_filenames)
            ic.save(ic_filename)

        # Make several searchable collections.
        def create_or_load_sic(algorithm, distance_metric, sigma, num_dimensions):
            if algorithm == 'exact':
                sic_class = SearchableImageCollectionExact
            elif algorithm == 'flann':
                sic_class = SearchableImageCollectionFLANN
            elif algorithm == 'ckdtree':
                sic_class = SearchableImageCollectionCKDTree
            else:
                raise Exception("Unknown algorithm.")

            filename = os.path.join(dirname, '{}_{}_{}_{}_{}.pickle'.format(
                image_list_name, algorithm, distance_metric, sigma, num_dimensions))

            if os.path.exists(filename):
                sic = sic_class.load(filename)
            else:
                sic = sic_class(ic, distance_metric, sigma, num_dimensions)
                sic.save(filename)

            return sic

        # search several query images and output to html summary
        np.random.seed(0)
        image_inds = np.random.permutation(range(len(image_filenames)))
        image_inds = image_inds[:num_queries]

        # there are ~200 dimensions in our palette.
        modes = [
            ('exact', 'euclidean', 10, 50), ('exact', 'manhattan', 10, 50),
            ('exact', 'euclidean', 10, 0),  ('exact', 'manhattan', 10, 0),
            ('exact', 'chi_square', 10, 0), ('exact', 'chi_square', 20, 0),
            ('exact', 'chi_square', 30, 0), ('exact', 'euclidean', 20, 0),

            ('flann', 'euclidean', 10, 50), ('flann', 'manhattan', 10, 50),
            ('flann', 'euclidean', 10, 0),  ('flann', 'manhattan', 10, 0),
            ('flann', 'chi_square', 10, 0),
            ('flann', 'euclidean', 20, 0),  ('flann', 'manhattan', 20, 0),
            ('flann', 'chi_square', 20, 0),

            ('ckdtree', 'euclidean', 10, 30), ('ckdtree', 'manhattan', 10, 30)
        ]

        time_elapsed = {}
        for mode in modes:
            sic = create_or_load_sic(*mode)
            tt.tic(mode)
            data = [sic.search_by_image_in_dataset(ind) for ind in image_inds]
            time_elapsed[mode] = tt.qtoc(mode)
            print("Time elapsed for %s: %.3f s" % (mode, time_elapsed[mode]))

            filename = os.path.join(
                dirname, 'matches_{}_{}_{}_{}.html'.format(*mode))
            with open(filename, 'w') as f:
                f.write(template.render(
                    num_queries=num_queries, time_elapsed=time_elapsed[mode],
                    data=data))
    
if __name__ == '__main__':
    unittest.main()

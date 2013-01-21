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
        Load subset of the Flickr interestingness dataset that can be compiled
        with the rayleigh.assemble_flickr_dataset module.
        """
        # Parametrization of our test.
        image_list_name = 'flickr_10K'
        num_images = int(1e5)

        dirname = skutil.makedirs(os.path.join(temp_dirname, image_list_name))
        num_queries = 50
        palette = rayleigh.Palette(num_hues=10, light_range=3, sat_range=2)
        palette.output(dirname=dirname)

        # Set up jinja template.
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(support_dirname))
        template = env.get_template('matches.html')
        
        # Load the image collection.
        data_filename = os.path.join(repo_dirname, 'data', 'flickr_1M.json.gz')
        ids, urls = rayleigh.ids_and_urls_from_dataset(data_filename, num_images)
        # import cPickle
        # with open('temp.pickle') as f:
        #     ids, urls = zip(*cPickle.load(f))

        ic_filename = os.path.join(
            temp_dirname, '{}.pickle'.format(image_list_name))

        if os.path.exists(ic_filename):
            print("Loading ImageCollection from cache.")
            ic = rayleigh.ImageCollection.load(ic_filename)
        else:
            ic = rayleigh.ImageCollection(palette)
            ic.add_images(urls, ids)
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
        image_inds = np.random.permutation(range(len(urls)))
        image_inds = image_inds[:num_queries]

        # there are 88 dimensions in our palette.
        modes = [
            ('exact', 'euclidean', 8, 22),
            ('exact', 'euclidean', 8, 0),
            ('exact', 'euclidean', 16, 0),

            ('exact', 'manhattan', 8, 22),
            ('exact', 'manhattan', 8, 0),
            ('exact', 'manhattan', 16, 0),

            ('exact', 'chi_square', 8, 0),
            ('exact', 'chi_square', 16, 0),

            #('flann', 'euclidean', 8, 22),
            #('flann', 'euclidean', 8, 0),

            #('flann', 'manhattan', 8, 22),
            #('flann', 'manhattan', 8, 0),

            #('flann', 'chi_square', 8, 0),
            
            #('ckdtree', 'euclidean', 8, 22),
            #('ckdtree', 'manhattan', 8, 22)
        ]

        time_elapsed = {}
        for mode in modes:
            sic = create_or_load_sic(*mode)
            tt.tic(mode)
            data = [sic.search_by_image_in_dataset(ids[ind]) for ind in image_inds]
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

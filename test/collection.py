"""
Test the image collection methods.
"""
from context import *

from sklearn.utils import shuffle


class TestSyntheticCollection(unittest.TestCase):
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
        sic = rayleigh.SearchableImageCollectionExact(ic, 'euclidean')

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
            
        # set up jinja template
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(support_dirname))
        template = env.get_template('matches.html')

        dirname = skutil.makedirs(os.path.join(temp_dirname, 'mirflickr'))
        image_list_filename = os.path.join(support_dirname, 'mirflickr_25K.txt')
        with open(image_list_filename) as f:
            image_filenames = [x.strip() for x in f.readlines()]

        palette = rayleigh.Palette(num_hues=8, sat_range=2, light_range=2)
        palette.output(dirname=dirname)

        def load_ic():
            ic_filename = os.path.join(
                dirname, '{}.pickle'.format(image_list_filename))

            if os.path.exists(ic_filename):
                print("Loading ImageCollection from cache.")
                ic = rayleigh.ImageCollection.load(ic_filename)
            else:
                ic = rayleigh.ImageCollection(palette)
                ic.add_images(image_filenames)
                ic.save(ic_filename)
            return ic

        ic = load_ic()
        sics = {
            'euclidean_exact': rayleigh.SearchableImageCollectionExact(ic, 'euclidean'),
            'euclidean_flann': rayleigh.SearchableImageCollectionFLANN(ic, 'euclidean'),
            'euclidean_ckd': rayleigh.SearchableImageCollectionCKDTree(ic, 'euclidean'),
            'manhattan_exact': rayleigh.SearchableImageCollectionExact(ic, 'manhattan'),
            'manhattan_flann': rayleigh.SearchableImageCollectionFLANN(ic, 'manhattan'),
            'manhattan_ckd': rayleigh.SearchableImageCollectionCKDTree(ic, 'manhattan')
        }

        # search several query images and output to html summary
        np.random.seed(0)
        image_inds = np.random.permutation(range(len(image_filenames)))[:200]

        time_elapsed = {}
        for mode, sic in sics.iteritems():
            tt.tic(mode)
            #data = [sic.search_by_image(image_filenames[ind]) for ind in image_inds]
            data = [sic.search_by_image_in_dataset(ind) for ind in image_inds]
            time_elapsed[mode] = tt.qtoc(mode)
            # data is a list of (query_img, results) tuples
            filename = os.path.join(dirname, 'matches_{}.html'.format(mode))
            with open(filename, 'w') as f:
                f.write(template.render(
                    time_elapsed=time_elapsed[mode], data=data))
            print("Time elapsed for %s: %.3f s" % (mode, time_elapsed[mode]))
        
        for mode in ['euclidean_flann', 'euclidean_ckd']:
            speedup = time_elapsed['euclidean_exact'] / time_elapsed[mode]
            print("{}: speedup of {:.3f} over {}".format(mode, speedup, 'manhattan_exact'))

        for mode in ['manhattan_flann', 'manhattan_ckd']:
            speedup = time_elapsed[mode] / time_elapsed['manhattan_exact']
            print("{}: speedup of {:.3f} over {}".format(mode, speedup, 'manhattan_exact'))

if __name__ == '__main__':
    unittest.main()

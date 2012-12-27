import cPickle
import numpy as np
import IPython.parallel as parallel
import rayleigh

from skpyutils import TicToc
tt = TicToc()


class ImageCollection(object):
    """
    Collection of images with their color palette histograms.
    """

    @staticmethod
    def load(filename):
        """
        Load ImageCollection from filename and prepare the data structures.
        """
        return cPickle.load(open(filename))

    def save(self, filename):
        """
        Save self to filename.
        """
        cPickle.dump(self, open(filename, 'w'), 2)

    def __init__(self, palette):
        """
        Initalize an empty ImageCollection with a color palette,
        and set up needed data structures.

        Args:
          - palette (Palette)
        """
        self.palette = palette
        self.images = []
        self.hists = np.zeros((0, len(self.palette.hex_list)))


    def add_images(self, image_urls, image_ids=None):
        """
        Add all images in the list of URLs.

        If ipcluster is running, load images in parallel.

        Args:
            - image_urls (list)

            - image_ids (list) [None]:
                If given, images are stored with the given ids.
                If None, the index of the image in the dataset is its id.
        """
        # need to construct the arguments due to ipython's pickling nonsense
        jobs = [(url, self.palette) for url in image_urls]

        print("Loading images...")
        tt = TicToc()
        try:
            rc = parallel.Client()
            lview = rc.load_balanced_view()
            results = lview.map(process_image, jobs)
            results.wait_interactive()
        except:
            print("WARNING: launch an ipython cluster to parallelize loading.")
            results = map(process_image, jobs)
        print("Finished in %.3f s" % tt.qtoc())

        images, hists = zip(*results)
        self.images += images
        self.hists = np.vstack((self.hists, np.array(hists)))
        assert(len(self.images) == self.hists.shape[0])


# For parallel execution, function must be in module scope
def process_image(args):
    image_url, palette = args
    img = rayleigh.Image(image_url)
    hist = img.histogram_colors(palette)
    img.discard_data()
    return img, hist

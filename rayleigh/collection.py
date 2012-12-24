import cPickle
import numpy as np
import IPython.parallel as parallel
from itertools import izip_longest
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
          - palette (rayleigh.Palette)
        """
        self.palette = palette
        self.images = []
        self.hists = np.zeros((0, len(self.palette.hex_list)))


    def add_images(self, image_filenames):
        """
        Add all images in the image_filenames list.

        If ipcluster is running, load images in parallel.
        """
        iterable = izip_longest(image_filenames, [], fillvalue=self.palette)
        iterated = [x for x in iterable]  # need to do this because of ipython
        print("Loading images...")

        try:
            rc = parallel.Client()
            lview = rc.load_balanced_view()
            results = lview.map(process_image, iterated)
            results.wait_interactive()
        except:
            print("WARNING: launch an ipython cluster to parallelize loading.")
            tt = TicToc()
            results = map(process_image, iterated)
            print("Finished in %.3f s" % tt.qtoc())

        images, hists = zip(*results)

        self.images += images
        self.hists = np.vstack((self.hists, np.array(hists)))
        assert(len(self.images) == self.hists.shape[0])


# For parallel execution, must be in module scope
def process_image(args):
    image_filename, palette = args
    img = rayleigh.Image(image_filename)
    hist = img.histogram_colors(palette)
    img.discard_data()
    return img, hist

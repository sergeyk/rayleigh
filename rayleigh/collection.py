import cgi
import cPickle
import numpy as np
import pyflann
import IPython.parallel as parallel
from itertools import izip_longest
from skpyutils import TicToc

import rayleigh

tt = TicToc()


class ImageCollection(object):
    """
    ImageCollection: collection of images searchable by color.

    TODOS:
    - try rtree instead of flann
    """

    @staticmethod
    def load(filename):
        """
        Load ImageCollection from filename.
        """
        ic = cPickle.load(open(filename))
        ic.build_index()
        return ic


    def save(self, filename):
        """
        Save self to filename.
        Exclude the built FLANN index because of some bug there. We'll rebuild.
        """
        flann = self.flann
        self.flann = None
        cPickle.dump(self, open(filename, 'w'), 2)
        self.flann = flann


    def __init__(self, palette, distance_type='cs'):
        """
        Initalize an empty ImageCollection with a color palette, and set up the
        data structures.

        Args:
          - palette (rayleigh.Palette)
        """
        self.palette = palette
        self.images = []
        self.hists = np.zeros((0, len(self.palette.hex_list)))
        self.flann = None
        self.distance_type = distance_type

    def set_distance_type(self, distance_type='cs'):
        """
        Set the distance type used in the FLANN index.
        """
        if not self.distance_type == distance_type:
            self.distance_type = distance_type
            self.build_index(distance_type)


    def add_images(self, image_filenames):
        """
        Add all images in the image_filenames list.

        >>> python test/collection.py
        Loading images:
        10000/10000 tasks finished after 1841 s
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

        self.build_index(self.distance_type)


    def build_index(self, distance_type='cs'):
        """
        Build the FLANN index, with the default distance type of Chi-squared.
        """
        print("ImageCollection: building index")
        tt.tic('build_index')
        pyflann.set_distance_type(distance_type)
        self.flann = pyflann.FLANN()
        # self.params = self.flann.build_index(
        #     self.hists,
        #     algorithm='kdtree', trees=1)
        self.params = self.flann.build_index(
            self.hists,
            algorithm='kdtree')
        tt.toc('build_index')
        print(self.params)


    def search_by_image(self, image_filename, num=20, mode='euclid_flann'):
        """
        Search images in database by color similarity to image.

        Returns:
            - query_img (dict): info about the query image

            - results (list): list of dicts that were returned
        """
        img = rayleigh.Image(image_filename)
        color_hist = img.histogram_colors(self.palette)
        if mode == 'euclid_flann':
            results = self.search_by_color_hist_flann(color_hist, num, 'euclid')
        elif mode == 'chi2_flann':
            results = self.search_by_color_hist_flann(color_hist, num, 'chi2')
        elif mode == 'euclid_exact':
            results = self.search_by_color_hist_exact(color_hist, num, 'euclid')
        elif mode == 'chi2_exact':
            results = self.search_by_color_hist_exact(color_hist, num, 'chi2')
        else:
            raise Exception("Unsupported mode")
        return img.as_dict(), results


    def search_by_color_hist_exact(self, color_hist, num=20, mode='chi2'):
        if mode == 'euclid':
            from sklearn.metrics import euclidean_distances
            dists = euclidean_distances(
                self.hists, color_hist, squared=True).flatten()
        elif mode == 'chi2':
            from sklearn.metrics.pairwise import chi2_kernel
            dists = -chi2_kernel(self.hists, color_hist).flatten()
        else:
            raise Exception("Unsupported mode.")

        ind = np.argsort(dists)
        results = []
        for i in ind[:num]:
            dist = dists[i]
            img = self.images[i]
            results.append({
                'width': img.orig_width, 'height': img.orig_height,
                'filename': cgi.escape(img.filename), 'distance': dist})
        return results


    def search_by_color_hist_flann(self, color_hist, num=20, mode='euclid'):
        """
        Search images in database by color similarity to a color histogram.
        """
        assert(self.flann is not None)


        result, dists = self.flann.nn_index(
            color_hist, num, checks=self.params['checks'])
        images = np.array(self.images)[result].squeeze().tolist()
        results = []
        for img, dist in zip(images, dists.squeeze()):
            results.append({
                'width': img.orig_width, 'height': img.orig_height,
                'filename': cgi.escape(img.filename), 'distance': dist})
        return results


# For parallel execution, must be in module scope
def process_image(args):
    image_filename, palette = args
    img = rayleigh.Image(image_filename)
    hist = img.histogram_colors(palette)
    img.discard_data()
    return img, hist

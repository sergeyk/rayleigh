"""
The back-end of the multicolor image search.
"""

import os
import cgi
import cPickle
import numpy as np
import simplejson as json
from skimage import img_as_float
from skimage.io import imread, imsave, imshow
from skimage.color import hsv2rgb, gray2rgb, rgb2lab, lab2rgb
from sklearn.metrics import euclidean_distances
from scipy.misc import imresize
import pyflann

from IPython import embed


class ImageCollection(object):
    """
    ImageCollection is the collection of images searchable by color.

    TODOS:
    - try rtree instead of flann
    """

    @staticmethod
    def load(filename):
        """
        Load ImageCollection from filename.
        """
        ic = cPickle.load(open(filename))
        ic.flann = None
        ic.build_index()
        return ic


    def save(self, filename):
        """
        Save self to filename.
        """
        self.flann = None
        cPickle.dump(self, open(filename,'w'), 2)


    def __init__(self, hex_palette):
        """
        Initalize an empty ImageCollection with a color palette, and set up the
        data structures.

        Args:
          - hex_palette (list): [<hex color>, <hex color>, ...].
        """
        self.hex_palette = hex_palette
        self.lab_palette = lab_palette(hex_palette)
        self.images = []
        self.hists = np.zeros((0,len(self.hex_palette)))


    def add_images(self, image_filenames):
        """
        Add all images in the image_filenames list.

        >>> python test/collection.py
        Loading images:
        10000/10000 tasks finished after 1841 s
        """
        from IPython.parallel import Client
        rc = Client()
        lview = rc.load_balanced_view()

        from itertools import izip_longest
        iterable = izip_longest(image_filenames, [], fillvalue=self.lab_palette)
        iterated = [x for x in iterable] #  need to do this because of ipython
        print("Loading images:")
        results = lview.map(process_image, iterated)
        results.wait_interactive()
        images, hists = zip(*results)

        self.images += images
        self.hists = np.vstack((self.hists, np.array(hists)))
        assert(len(self.images) == self.hists.shape[0])

        self.build_index()


    def build_index(self):
        self.flann = pyflann.FLANN()
        hists = self.hists / np.atleast_2d(self.hists.sum(axis=1)).T
        self.params = self.flann.build_index(hists,
            algorithm='kdtree',
            trees=15)
        print(self.params)


    def search_by_image(self, image_filename, num=20):
        """
        Search images in database by color similarity to image.
        """
        img = Image(image_filename)
        color_hist = img.histogram_colors(self.lab_palette)
        color_hist /= color_hist.sum()
        return self.search_by_color_hist(color_hist, num)


    def search_by_color_hist(self, color_hist, num=20):
        """
        Search images in database by color similarity to a color histogram.
        """
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
    image_filename, lab_palette = args
    img = Image(image_filename)
    hist = img.histogram_colors(lab_palette)
    img.discard_data()
    return img, hist

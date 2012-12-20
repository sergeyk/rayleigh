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


def plot_histogram(palette, palette_histogram):
    """
    Return histogram plot of palette colors.
    """
    # TODO
    pass


def smoothed_histogram(palette, color_array, sigma=15):
    """"
    Assign colors in the image to nearby colors in the palette, weighted by
    distance in Lab color space.

    Args:
        - lab_palette ((K,3) ndarray):
            K is the number of colors, columns are L, a, b values.

        - color_array ((N, 3) ndarray):
            N is the number of data points, columns are L, a, b values.

        - sigma (float): (0,1] value to control the steepness of exponential
            falloff. To see the effect:

            >>> from pylab import *
            >>> ds = linspace(0,5000) # squared dsitance
            >>> sigma=10; plot(ds, exp(-ds/(2*sigma**2)), label='\sigma=%.1f'%sigma)
            >>> sigma=20; plot(ds, exp(-ds/(2*sigma**2)), label='\sigma=%.1f'%sigma)
            >>> sigma=40; plot(ds, exp(-ds/(2*sigma**2)), label='\sigma=%.1f'%sigma)
            >>> ylim([0,1]); legend()

            sigma=20 seems reasonable, hitting 0 around squared distance of 4000

    Returns:
        - (K,) ndarray: the normalized soft histogram of colors in the image.
    """
    # This is the fastest way to do this that I've been able to find
    # >>> %%timeit -n 100 from sklearn.metrics import euclidean_distances
    # >>> euclidean_distances(palette, self.lab_array, squared=True)
    # 100 loops, best of 3: 2.33 ms per loop
    dist = euclidean_distances(palette, color_array, squared=True).T
    n = 2.*sigma**2
    weights = np.exp(-dist/n)
    sums = np.maximum(weights.sum(1), 1e-6)
    hist = (weights / sums[:, np.newaxis]).sum(0)
    return hist


class Image(object):
    """
    Encapsulation of methods to extract information (e.g. color) from images.
    """

    MAX_DIMENSION = 300  # pixels

    def __init__(self, image_filename, max_dimension=MAX_DIMENSION):
        """
        Read the image at the URL, and make sure it's in RGB format.
        """
        self.filename = os.path.abspath(image_filename)
        img = img_as_float(imread(image_filename))

        # grayscale
        if img.ndim == 2:
            img = np.dstack((img,img,img))

        # with alpha
        # TODO: be smart here, but for now simply remove alpha channel
        if img.ndim == 4:
            img = img[:, :, :3]
        h, w, d = tuple(img.shape)
        assert(d == 3)
        self.orig_height, self.orig_width, self.orig_depth = h, w, d

        # downsample image if needed
        img_r = img
        resize_factor = 1
        if max(w, h) > max_dimension:
            resize_factor = float(max_dimension) / max(w, h)
            img_r = imresize(img, resize_factor, 'nearest')
            h, w, d = tuple(img_r.shape)
        self.h, self.w, self.d = h, w, d
        self.resize_factor = resize_factor

        # convert to Lab color space and reshape
        self.lab_array = rgb2lab(img_r).reshape((h*w, d))


    def histogram_colors(self, palette, sigma=15):
        """
        Return a smoothed histogram See smoothed_histogram().
        """
        assert(self.lab_array is not None)
        return smoothed_histogram(palette, self.lab_array, sigma)


    def quantize_to_palette(self, palette, filename):
        """
        Save to filename a version of the image with colors quantized
        to the nearest color in the given palette.

        Args:
            - palette (Kx3 ndarray): K Lab colors.

            - filename (string): where image will be written

        Returns:
            None
        """
        assert(self.lab_array is not None)

        dist = euclidean_distances(palette, self.lab_array, squared=True).T
        min_ind = np.argmin(dist, axis=1)
        lab_array = palette[min_ind, :]
        img = lab2rgb(lab_array.reshape((self.h, self.w, self.d)))
        imsave(filename, img)


    def discard_data(self):
        """
        Drop the pixel data.
        """
        self.lab_array = None


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

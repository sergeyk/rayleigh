"""
The back-end of the multicolor image search.
"""

import os
import cgi
import cPickle
import numpy as np
from skimage import img_as_float
from skimage.io import imread, imsave, imshow
from skimage.color import rgb2lab, lab2rgb
from sklearn.metrics import euclidean_distances
from scipy.misc import imresize
import pyflann

from IPython import embed


def create_palette():
    """
    Create a color palette.
    """
    from skimage.color import hsv2rgb, gray2rgb
    from skimage.io import imshow
    num_hues = 13
    hues = np.tile(np.linspace(0, 1, num_hues), (7, 1))
    sats = np.hstack((np.linspace(0.1, 0.6, 3), np.ones(4)))
    sats = np.tile(np.atleast_2d(sats).T, (1, num_hues))
    lights = np.hstack((np.ones(4), np.linspace(.8, 0.3, 3)))
    lights = np.tile(np.atleast_2d(lights).T, (1, num_hues))
    colors = hsv2rgb(np.dstack((hues , sats, lights)))
    gray = np.tile(np.atleast_3d(np.linspace(0, 1, num_hues)), (1, 1, 3))
    palette = np.vstack((colors, gray))
    imsave('gen_palette.png', palette)
    palette_json = [rgb2hex(row) for row in palette.reshape((w*h), d)]
    #output_palette_html(palette_json, num_hues)
    return palette_json


def output_palette_html(palette_json, num_hues):
    palette_arr = np.array(palette_json).reshape(len(palette_json)/num_hues, num_hues)
    html = ""
    for row in palette_arr:
        for color in row:
            html += '<a id="{0}"><span style="background-color: {0}" /></a>\n'.format(color)
        html += "<br />\n"
    return html


def rgb2hex(rgb):
    """
    Convert a tuple or list or array of 3 [0,1] RGB colors to a hex string.
    """
    return '#%02x%02x%02x' % tuple([np.round(val * 255) for val in rgb[:3]])


def hex2rgb(hexcolor_str):
    """
    Convert string containing an HTML-encoded color to an RGB tuple.

    >>> hex2rgb('#ffffff')
    (255, 255, 255)
    >>> hex2rgb('33cc00')
    (51, 204, 0)
    """
    hexcolor = int(hexcolor_str.strip('#'), 16)
    r = (hexcolor >> 16) & 0xff
    g = (hexcolor >> 8) & 0xff
    b = hexcolor & 0xff
    return (r, g, b)


def lab_palette(hex_palette):
    """
    Convert a list of hex color name strings to a (K,3) array of Lab colors.
    """
    colors = [hex2rgb(hexcolor_str) for hexcolor_str in hex_palette]
    return rgb2lab(np.array(colors, ndmin=3) / 255.).squeeze()


def plot_histogram(palette, palette_histogram):
    """
    Return histogram plot of palette colors.
    """
    pass


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
            img = np.tile(img, (1, 1, 3))

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
        Assign colors in the image to nearby colors in the palette, weighted by
        distance in Lab color space.

        Args:
            - lab_palette ({'grayscale': (K,3) ndarray}):
                K is the number of colors, columns are Lab.
            - sigma (float): (0,1] value to control the steepness of exponential
                falloff. To see the effect:

                from pylab import *
                ds = linspace(0,5000) # squared dsitance
                sigma=10; plot(ds, exp(-ds/(2*sigma**2)), label='\sigma=%.1f'%sigma)
                sigma=20; plot(ds, exp(-ds/(2*sigma**2)), label='\sigma=%.1f'%sigma)
                sigma=40; plot(ds, exp(-ds/(2*sigma**2)), label='\sigma=%.1f'%sigma)
                ylim([0,1]); legend()

                sigma=20 seems reasonable, hitting 0 around squared distance of 4000
                
                # TODO: try sigma=40 if performance is lacking

        Returns:
          - (K,) ndarray: the normalized soft histogram of colors in the image.
        """
        # This is the fastest way to do this that I've been able to find
        # >>> %%timeit -n 100 from sklearn.metrics import euclidean_distances
        # >>> euclidean_distances(palette, self.lab_array, squared=True)
        # 100 loops, best of 3: 2.33 ms per loop
        dist = euclidean_distances(palette, self.lab_array, squared=True).T
        n = 2.*sigma**2
        weights = np.exp(-dist/n)
        sums = np.maximum(weights.sum(1), 1e-6)
        hist = (weights / sums[:,np.newaxis]).sum(0)
        return hist


    def quantize_to_palette(self, palette):
        """
        Return image with all colors converted to the nearest palette color.
        """
        dist = euclidean_distances(palette, self.lab_array, squared=True).T
        min_ind = np.argmin(dist, axis=1)
        lab_array = palette[min_ind, :]
        img = lab2rgb(lab_array.reshape((self.h, self.w, self.d)))
        imsave(self.filename+'_quantized.png', img)


    def discard_array(self):
        """
        Drop the pixel data.
        """
        self.lab_array = None


class ImageCollection(object):
    """
    ImageCollection is the collection of images searchable by color.

    TODOS:
    - try rtree
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
        """
        hists = np.zeros((len(image_filenames), len(self.hex_palette)))
        for i, image_filename in enumerate(image_filenames):
            img = Image(image_filename)
            self.images.append(img)
            hists[i,:] = img.histogram_colors(self.lab_palette)
            img.discard_array()

        self.hists = np.vstack((self.hists, hists))
        assert(len(self.images) == self.hists.shape[0])

        self.build_index()


    def build_index(self):
        self.flann = pyflann.FLANN()
        self.params = self.flann.build_index(self.hists, log_level='info')
        print(self.params)


    def search_by_image(self, image_filename, num=10):
        """
        Search images in database by color similarity to image.
        """
        img = Image(image_filename)
        color_hist = img.histogram_colors(self.lab_palette)
        return self.search_by_color_hist(color_hist, num)


    def search_by_color_hist(self, color_hist, num=10):
        """
        Search images in database by color similarity to a color histogram.
        """
        result, dists = self.flann.nn_index(
          color_hist, num, checks=self.params['checks'])
        images = np.array(self.images)[result].squeeze().tolist()
        results = []
        for img, dist in zip(images, dists.squeeze()):
            results.append({
                'width': img.width, 'height': img.height,
                'filename': cgi.escape(img.filename), 'distance': dist})
        return results

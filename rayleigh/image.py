import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.color import rgb2lab, lab2rgb
from sklearn.metrics import euclidean_distances

import util


class Image(object):
    """
    Methods to extract color information from an image.

    Note that we are not resizing images because the only acceptable
    image resizing method that I'm able to find converts images to 8 bits.
    """

    MAX_DIMENSION = 200

    def __init__(self, image_filename):
        """
        Read the image at the URL in RGB format, downsample if needed,
        and convert to Lab colorspace. Store original dimensions, resize_factor,
        and the filename of the image.
        """
        self.filename = os.path.abspath(image_filename)
        img = imread(image_filename, plugin='matplotlib')

        # grayscale
        if img.ndim == 2:
            img = np.dstack((img, img, img))

        # with alpha
        # TODO: be smart here, but for now simply remove alpha channel
        if img.ndim == 4:
            img = img[:, :, :3]
        h, w, d = tuple(img.shape)
        assert(d == 3)
        self.orig_height, self.orig_width, self.orig_depth = h, w, d

        # downsample for speed
        # >>> def d(dim, max_dim): return arange(0,dim,dim/max_dim+1).shape
        # >>> plot(range(1200), [d(x, 200) for x in range(1200)])
        stride = min(h, w) / self.MAX_DIMENSION + 1
        img = img[::stride, ::stride, :]

        h, w, d = tuple(img.shape)
        self.h, self.w, self.d = tuple(img.shape)

        # convert to Lab color space and reshape
        self.lab_array = rgb2lab(img).reshape((h * w, d))

    def as_dict(self):
        """
        Return relevant info about self in a dict.
        """
        return {'filename': self.filename,
                'width': self.orig_width, 'height': self.orig_height}

    def discard_data(self):
        """
        Drop the pixel data, which is useful to do after the histogram is
        computed, to avoid storing it in an ImageCollection.
        """
        self.lab_array = None


    def histogram_colors(self, palette, sigma=15, plot_filename=None):
        """
        Return a palette histogram of colors in the image, smoothed with
        a Gaussian. See rayleigh.util.smoothed_histogram() for details.

        Args:
            - palette (rayleigh.Palette): of K colors

            - sigma (float): parameter of the smoothing Gaussian

            - plot_filename (string): [optional] if given, save histogram

        Returns:
            - hist (Kx, ndarray)
        """
        hist = util.smoothed_histogram(palette, self.lab_array, sigma)
        if plot_filename is not None:
            fig = plt.figure(figsize=(10, 6), dpi=300)
            ax = fig.add_subplot(111)
            ax.bar(range(len(hist)), hist,
                   color=palette.hex_list, edgecolor='black')
            ax.set_ylim((0, 1))
            ax.xaxis.set_ticks([])
            ax.set_xlim((0, len(palette.hex_list)))
            fig.savefig(plot_filename, dpi=300, facecolor='none')
        return hist


    def quantize_to_palette(self, palette, filename):
        """
        Save to filename a version of the image with colors quantized
        to the nearest color in the given palette.

        Args:
            - palette (rayleigh.Palette)

            - filename (string): where image will be written

        Returns:
            - None
        """
        assert(self.lab_array is not None)
        dist = euclidean_distances(
            palette.lab_array, self.lab_array, squared=True).T
        min_ind = np.argmin(dist, axis=1)
        quantized_lab_array = palette.lab_array[min_ind, :]
        img = lab2rgb(quantized_lab_array.reshape((self.h, self.w, self.d)))
        imsave(filename, img)

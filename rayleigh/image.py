import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2lab, lab2rgb
from sklearn.metrics import euclidean_distances
import util


class Image(object):
    """
    Methods to extract color information from an image, histogram the colors
    with a given palette, and to output a palette image.
    """

    MAX_DIMENSION = 240 + 1

    def __init__(self, url):
        """
        Read the image at the URL in RGB format, downsample if needed,
        and convert to Lab colorspace.
        Store original dimensions, resize_factor, and the filename of the image.

        Args:
            - url (string):
                a URL or file path to the image to load.
        """
        self.url = url
        img = imread(url)

        # grayscale
        if img.ndim == 2:
            img = np.tile(img[:, :, np.newaxis], (1, 1, 3))

        # TODO: Should be smart here in the future.
        # For now simply remove alpha channel.
        # with alpha
        if img.ndim == 4:
            img = img[:, :, :3]
        
        h, w, d = tuple(img.shape)
        self.orig_h, self.orig_w, self.orig_d = tuple(img.shape)

        # Downsample for speed.
        #
        # >>> def d(dim, max_dim): return arange(0, dim, dim / max_dim + 1).shape
        # >>> plot(range(1200), [d(x, 200) for x in range(1200)])
        #
        # NOTE: I'd like to resize properly, but cannot seem to find a good
        # method for this in Python. scipy.misc.imresize uses PIL, which needs
        # 8bit data. Anyway, this is faster and probably about as good.
        h_stride = h / self.MAX_DIMENSION + 1
        w_stride = w / self.MAX_DIMENSION + 1
        img = img[::h_stride, ::w_stride, :]

        h, w, d = img.shape
        self.h, self.w, self.d = img.shape

        # convert to Lab color space and reshape
        self.lab_array = rgb2lab(img).reshape((h * w, d))

    def as_dict(self):
        """
        Return relevant info about self in a dict.
        """
        return {'url': self.url,
                'width': self.orig_w, 'height': self.orig_h}

    def discard_data(self):
        """
        Drop the pixel data (useful when storing Image in an ImageCollection.
        """
        self.lab_array = None

    def histogram_colors(self, palette, plot_filename=None):
        """
        Return a palette histogram of colors in the image.

        Args:
            - palette (rayleigh.Palette): Containing K colors.

            - plot_filename (string) [optional]:
                If given, save histogram to this filename.

        Returns:
            - color_hist (K, ndarray)
        """
        color_hist = util.histogram_colors(palette, self.lab_array)
        if plot_filename is not None:
            util.plot_histogram(color_hist, palette, plot_filename)
        return color_hist

    def histogram_colors_smoothed(self, palette, sigma=10,
                                  plot_filename=None, direct=True):
        """
        Return a palette histogram of colors in the image, smoothed with
        a Gaussian.

        Args:
            - palette (rayleigh.Palette):
                Consisting of K colors.

            - sigma (float):
                Variance of the smoothing Gaussian.

            - direct (bool) [default=True]:
                If True, constructs a smoothed histogram directly from pixels.
                If False, constructs a nearest-color histogram and then
                smoothes it.

        Returns:
            - color_hist (K, ndarray)
        """
        if direct:
            color_hist = util.histogram_colors_smoothed(
                palette, self.lab_array, sigma)
        else:
            color_hist = util.histogram_colors(palette, self.lab_array)
            color_hist = util.smooth_histogram(palette, color_hist, sigma)
        if plot_filename is not None:
            util.plot_histogram(color_hist, palette, plot_filename)
        return color_hist

    def output_quantized_to_palette(self, palette, filename):
        """
        Save to filename a version of the image with all colors quantized
        to the nearest color in the given palette.

        Args:
            - palette (rayleigh.Palette)

            - filename (string): where image will be written

        Returns:
            - None
        """
        dist = euclidean_distances(
            palette.lab_array, self.lab_array, squared=True).T
        min_ind = np.argmin(dist, axis=1)
        quantized_lab_array = palette.lab_array[min_ind, :]
        img = lab2rgb(quantized_lab_array.reshape((self.h, self.w, self.d)))
        imsave(filename, img)

    def output_color_palette_image(self, palette, filename, percentile=90):
        """
        Output the main colors of the image to a "palette image."

        Args:
            - palette (rayleigh.Palette)

            - filename (string): where image will be written

            - percentile (int) [90]:
                Output only colors above this percentile of prevalence
                in the image.

        Returns:
            - None
        """
        color_hist = util.histogram_colors(palette, self.lab_array)
        util.color_hist_to_palette_image(
            color_hist, palette, percentile, filename)

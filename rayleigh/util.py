import os
import numpy as np
import tempfile
import cStringIO as StringIO
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from skimage.io import imsave


def rgb2hex(rgb_number):
    """
    Args:
        - rgb_number (sequence of float)

    Returns:
        - hex_number (string)
    """
    return '#%02x%02x%02x' % tuple([np.round(val * 255) for val in rgb_number])


def hex2rgb(hexcolor_str):
    """
    Args:
        - hexcolor_str (string): e.g. '#ffffff' or '33cc00'

    Returns:
        - rgb_color (sequence of floats): e.g. (0.2, 0.3, 0)
    """
    color = hexcolor_str.strip('#')
    rgb = lambda x: round(int(x, 16) / 255., 5)
    return (rgb(color[:2]), rgb(color[2:4]), rgb(color[4:6]))


def color_hist_to_palette_image(color_hist, palette, percentile=90,
                                width=200, height=50, filename=None):
    """
    Output the main colors in the histogram to a "palette image."

    Parameters
    ----------
    color_hist : (K,) ndarray
    palette : rayleigh.Palette
    percentile : int, optional:
        Output only colors above this percentile of prevalence in the image.
    filename : string, optional:
        If given, save the resulting image to file.

    Returns
    -------
    rgb_image : ndarray
    """
    ind = np.argsort(-color_hist)
    ind = ind[color_hist[ind] > np.percentile(color_hist, percentile)]
    hex_list = np.take(palette.hex_list, ind)
    values = color_hist[ind]
    rgb_image = palette_query_to_rgb_image(dict(zip(hex_list, values)))
    if filename:
        imsave(filename, rgb_image)
    return rgb_image


def palette_query_to_rgb_image(palette_query, width=200, height=50):
    """
    Convert a list of hex colors and their values to an RGB image of given
    width and height.

    Args:
        - palette_query (dict):
            a dictionary of hex colors to unnormalized values,
            e.g. {'#ffffff': 20, '#33cc00': 0.4}.
    """
    hex_list, values = zip(*palette_query.items())
    values = np.array(values)
    values /= values.sum()
    nums = np.array(values * width, dtype=int)
    rgb_arrays = (np.tile(np.array(hex2rgb(x)), (num, 1))
                  for x, num in zip(hex_list, nums))
    rgb_array = np.vstack(rgb_arrays)
    rgb_image = rgb_array[np.newaxis, :, :]
    rgb_image = np.tile(rgb_image, (height, 1, 1))
    return rgb_image


def plot_histogram(color_hist, palette, plot_filename=None):
    """
    Return Figure containing the color palette histogram.

    Args:
        - color_hist (K, ndarray)

        - palette (Palette)

        - plot_filename (string) [default=None]:
                Save histogram to this file, if given.

    Returns:
        - fig (Figure)
    """
    fig = plt.figure(figsize=(5, 3), dpi=150)
    ax = fig.add_subplot(111)
    ax.bar(
        range(len(color_hist)), color_hist,
        color=palette.hex_list, edgecolor='black')
    ax.set_ylim((0, 0.3))
    ax.xaxis.set_ticks([])
    ax.set_xlim((0, len(palette.hex_list)))
    if plot_filename:
        fig.savefig(plot_filename, dpi=150, facecolor='none')
    return fig


def output_plot_for_flask(color_hist, palette):
    """
    Return an object suitable to be sent as an image by Flask,
    containing the color palette histogram.

    Args:
        - color_hist (K, ndarray)

        - palette (Palette)

    Returns:
        - png_output (StringIO)
    """
    fig = plot_histogram(color_hist, palette)
    strIO = StringIO.StringIO()
    plt.savefig(strIO, dpi=fig.dpi)
    strIO.seek(0)
    return strIO


def output_histogram_base64(color_hist, palette):
    """
    Return base64-encoded image containing the color palette histogram.

    Args:
        - color_hist (K, ndarray)

        - palette (Palette)

    Returns:
        - data_uri (base64 encoded string)
    """
    _, tfname = tempfile.mkstemp('.png')
    plot_histogram(color_hist, palette, tfname)
    data_uri = open(tfname, 'rb').read().encode('base64').replace('\n', '')
    os.remove(tfname)
    return data_uri


def histogram_colors_strict(lab_array, palette, plot_filename=None):
    """
    Return a palette histogram of colors in the image.

    Parameters
    ----------
    lab_array : (N,3) ndarray
        The L*a*b color of each of N pixels.
    palette : rayleigh.Palette
        Containing K colors.
    plot_filename : string, optional
        If given, save histogram to this filename.

    Returns
    -------
    color_hist : (K,) ndarray
    """
    # This is the fastest way that I've found.
    # >>> %%timeit -n 200 from sklearn.metrics import euclidean_distances
    # >>> euclidean_distances(palette, lab_array, squared=True)
    dist = euclidean_distances(palette.lab_array, lab_array, squared=True).T
    min_ind = np.argmin(dist, axis=1)
    num_colors = palette.lab_array.shape[0]
    num_pixels = lab_array.shape[0]
    color_hist = 1. * np.bincount(min_ind, minlength=num_colors) / num_pixels
    if plot_filename is not None:
        plot_histogram(color_hist, palette, plot_filename)
    return color_hist


def histogram_colors_smoothed(lab_array, palette, sigma=10,
                              plot_filename=None, direct=True):
    """
    Returns a palette histogram of colors in the image, smoothed with
    a Gaussian. Can smooth directly per-pixel, or after computing a strict
    histogram.

    Parameters
    ----------
    lab_array : (N,3) ndarray
        The L*a*b color of each of N pixels.
    palette : rayleigh.Palette
        Containing K colors.
    sigma : float
        Variance of the smoothing Gaussian.
    direct : bool, optional
        If True, constructs a smoothed histogram directly from pixels.
        If False, constructs a nearest-color histogram and then smoothes it.

    Returns
    -------
    color_hist : (K,) ndarray
    """
    if direct:
        color_hist_smooth = histogram_colors_with_smoothing(
            lab_array, palette, sigma)
    else:
        color_hist_strict = histogram_colors_strict(lab_array, palette)
        color_hist_smooth = smooth_histogram(color_hist_strict, palette, sigma)
    if plot_filename is not None:
        plot_histogram(color_hist_smooth, palette, plot_filename)
    return color_hist_smooth


def smooth_histogram(color_hist, palette, sigma=10):
    """
    Smooth the given palette histogram with a Gaussian of variance sigma.

    Parameters
    ----------
    color_hist : (K,) ndarray
    palette : rayleigh.Palette
        containing K colors.

    Returns
    -------
    color_hist_smooth : (K,) ndarray
    """
    n = 2. * sigma ** 2
    weights = np.exp(-palette.distances / n)
    norm_weights = weights / weights.sum(1)[:, np.newaxis]
    color_hist_smooth = (norm_weights * color_hist).sum(1)
    color_hist_smooth[color_hist_smooth < 1e-5] = 0
    return color_hist_smooth


def histogram_colors_with_smoothing(lab_array, palette, sigma=10):
    """
    Assign colors in the image to nearby colors in the palette, weighted by
    distance in Lab color space.

    Parameters
    ----------
    lab_array (N,3) ndarray:
        N is the number of data points, columns are L, a, b values.
    palette : rayleigh.Palette
        containing K colors.
    sigma : float
        (0,1] value to control the steepness of exponential falloff.
        To see the effect:

    >>> from pylab import *
    >>> ds = linspace(0,5000) # squared distance
    >>> sigma=10; plot(ds, exp(-ds/(2*sigma**2)), label='\sigma=%.1f'%sigma)
    >>> sigma=20; plot(ds, exp(-ds/(2*sigma**2)), label='\sigma=%.1f'%sigma)
    >>> sigma=40; plot(ds, exp(-ds/(2*sigma**2)), label='\sigma=%.1f'%sigma)
    >>> ylim([0,1]); legend()

        sigma=20 seems reasonable: hits 0 around squared distance of 4000.

    Returns:
    color_hist : (K,) ndarray
        the normalized, smooth histogram of colors.
    """
    dist = euclidean_distances(palette.lab_array, lab_array, squared=True).T
    n = 2. * sigma ** 2
    weights = np.exp(-dist / n)
    
    # normalize by sum: if a color is equally well represented by several colors
    # it should not contribute much to the overall histogram
    normalizing = weights.sum(1)
    normalizing[normalizing == 0] = 1e16
    normalized_weights = weights / normalizing[:, np.newaxis]

    color_hist = normalized_weights.sum(0)
    color_hist /= lab_array.shape[0]
    color_hist[color_hist < 1e-5] = 0
    return color_hist

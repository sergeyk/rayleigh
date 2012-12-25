import os
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from sklearn.metrics import euclidean_distances
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import StringIO


def rgb2hex(rgb):
    """
    Convert a sequence of three [0,1] RGB colors to a hex string.
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


def plot_histogram(hist, palette):
    """
    TODO
    """
    fig = plt.figure(figsize=(5, 3), dpi=300)
    ax = fig.add_subplot(111)
    ax.bar(range(len(hist)), hist,
           color=palette.hex_list, edgecolor='black')
    ax.set_ylim((0, 1))
    ax.xaxis.set_ticks([])
    ax.set_xlim((0, len(palette.hex_list)))
    return fig


def plot_histogram_html(hist, palette, as_html=False):
    """
    TODO
    """
    _, tfname = tempfile.mkstemp('.png')
    fig = plot_histogram(hist, palette)
    fig.savefig(tfname, dpi=150, facecolor='none')
    data_uri = open(tfname, 'rb').read().encode('base64').replace('\n', '')
    os.remove(tfname)
    if as_html:
        return '<img width="300px" src="data:image/png;base64,{0}">'.format(data_uri)
    return data_uri


def plot_histogram_flask(hist, palette):
    """
    TODO
    """
    fig = plot_histogram(hist, palette)
    canvas = FigureCanvas(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    return png_output


def smoothed_histogram(palette, color_array, sigma=15):
    """
    Assign colors in the image to nearby colors in the palette, weighted by
    distance in Lab color space.

    Args:
        - palette (rayleigh.Palette): of K colors

        - color_array ((N, 3) ndarray):
            N is the number of data points, columns are L, a, b values.

        - sigma (float): (0,1] value to control the steepness of exponential
            falloff. To see the effect:

    >>> from pylab import *
    >>> ds = linspace(0,5000) # squared distance
    >>> sigma=10; plot(ds, exp(-ds/(2*sigma**2)), label='\sigma=%.1f'%sigma)
    >>> sigma=20; plot(ds, exp(-ds/(2*sigma**2)), label='\sigma=%.1f'%sigma)
    >>> sigma=40; plot(ds, exp(-ds/(2*sigma**2)), label='\sigma=%.1f'%sigma)
    >>> ylim([0,1]); legend()

            sigma=20 seems reasonable: hits 0 around squared distance of 4000.

    Returns:
        - (K,) ndarray: the normalized soft histogram of colors in the image.
    """
    # This is the fastest way that I've found.
    # >>> %%timeit -n 100 from sklearn.metrics import euclidean_distances
    # >>> euclidean_distances(palette, self.lab_array, squared=True)
    # 100 loops, best of 3: 2.33 ms per loop
    dist = euclidean_distances(palette.lab_array, color_array, squared=True).T
    n = 2. * sigma ** 2
    weights = np.exp(-dist / n)
    # normalize by sum: if a color is equally well represented by several colors
    # it should not contribute much to the overall histogram
    normalizing = weights.sum(1)

    # TODO: optimize for speed here
    normalizing[normalizing == 0] = 1e-12
    normalized_weights = weights / normalizing[:, np.newaxis]
    hist = normalized_weights.sum(0) / color_array.shape[0]
    hist[hist < 1e-4] = 0
    return hist

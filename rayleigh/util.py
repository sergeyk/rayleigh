import numpy as np
from sklearn.metrics import euclidean_distances


def smoothed_histogram(palette, color_array, sigma=15):
    """"
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
    weights = dist
    weights = np.exp(-dist / n)
    weights = np.maximum(weights, 1e-6)
    # normalize by sum: if a color is equally well represented by several colors
    # it should not contribute much to the overall histogram
    normalized_weights = weights / weights.sum(1)[:, np.newaxis]
    hist = normalized_weights.sum(0) / color_array.shape[0]
    return hist

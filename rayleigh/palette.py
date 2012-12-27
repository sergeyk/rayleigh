"""
Provide methods to work with color conversion and the Palette class.
"""

import os
import numpy as np
from skimage.color import hsv2rgb, rgb2lab
from skimage.io import imsave
from sklearn.metrics import euclidean_distances

from util import rgb2hex


class Palette(object):
    """
    Encapsulate the list of hex colors and array of Lab values representations
    of a palette (codebook) of colors.

    Provide a parametrized method to generate a palette that covers the range
    of colors.
    """

    def __init__(self, num_hues=8, sat_range=2, light_range=2):
        """
        Create a color palette (codebook) in the form of a 2D grid of colors.

        Args:
            - num_hues (int): number of colors with full
                lightness and saturation, in the middle row.

            - sat_range (int): number of rows above middle row that show
                the same hues with decreasing saturation.

            - light_range (int) number of rows below middle row that show
                the same hues with decreasing lightness.

        Further, the bottom row has num_hues gradations from black to white.

        Returns:
            (Palette)
        """
        height = 1 + sat_range + (2 * light_range - 2)
        # generate num_hues+1 hues, but don't take the last one:
        # hues are on a circle, and we would be oversampling the origin
        hues = np.tile(np.linspace(0, 1, num_hues + 1)[:-1], (height, 1))
        if num_hues == 8:
            hues = np.tile(np.array(
                [0.,  0.10,  0.15,  0.28, 0.51, 0.58, 0.77,  0.85]), (height, 1))
        elif num_hues == 11:
            hues = np.tile(np.array(
                [0.0, 0.0833, 0.166, 0.25,
                 0.333, 0.5, 0.56333,
                 0.666, 0.73, 0.803,
                 0.916]), (height, 1))
        
        sats = np.hstack((
            np.linspace(0, 1, sat_range + 2)[1:-1],
            1,
            [1] * (light_range - 1),
            [.3] * (light_range - 1),
        ))
        lights = np.hstack((
            [1] * sat_range,
            1,
            np.linspace(1, 0, light_range + 2)[1:-2],
            np.linspace(1, 0, light_range + 2)[1:-2],
        ))

        sats = np.tile(np.atleast_2d(sats).T, (1, num_hues))
        lights = np.tile(np.atleast_2d(lights).T, (1, num_hues))
        colors = hsv2rgb(np.dstack((hues, sats, lights)))
        grays = np.tile(
            np.linspace(1, 0, height)[:, np.newaxis, np.newaxis], (1, 1, 3))

        self.rgb_image = np.hstack((colors, grays))

        # Make a nice histogram ordering of the hues and grays
        h, w, d = colors.shape
        color_array = colors.T.reshape((d, w * h)).T
        h, w, d = grays.shape
        gray_array = grays.T.reshape((d, w * h)).T
        
        self.rgb_array = np.vstack((color_array, gray_array))
        self.lab_array = rgb2lab(self.rgb_array[None, :, :]).squeeze()
        self.hex_list = [rgb2hex(row) for row in self.rgb_array]
        #assert(np.all(self.rgb_array == self.rgb_array[None, :, :].squeeze()))

        self.distances = euclidean_distances(self.lab_array, squared=True)

    def output(self, dirname):
        """
        Output an image of the palette, josn list of the hex
        colors, and an HTML color picker for it.

        Args:
            - dirname (string): directory for the files to be output

        Returns:
            - None
        """
        def get_palette_html():
            """
            Return HTML for a color picker using the given palette.
            """
            html = """
            <style>
                span {
                    width: 20px;
                    height: 20px;
                    margin: 2px;
                    padding: 0px;
                    display: inline-block;
                }
            </style>
            """
            for row in self.rgb_image:
                for rgb_color in row:
                    s = '<a id="{0}"><span style="background-color: {0}" /></a>\n'
                    html += s.format(rgb2hex(rgb_color))
                html += "<br />\n"
            return html

        imsave(os.path.join(dirname, 'palette.png'), self.rgb_image)
        with open(os.path.join(dirname, 'palette.html'), 'w') as f:
            f.write(get_palette_html())

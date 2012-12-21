"""
Provide methods to work with color conversion and the Palette class.
"""

import os
import simplejson as json
import numpy as np
from skimage.color import hsv2rgb, rgb2lab
from skimage.io import imsave


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


class Palette(object):
    """
    Encapsulate the list of hex colors and array of Lab values representations
    of a palette (codebook) of colors.

    Provide a parametrized method to generate a palette that covers the range
    of colors.
    """

    
    def __init__(self, num_hues=13, sat_range=2, light_range=2):
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
        height = 1 + sat_range + light_range
        # generate num_hues+1 hues, but don't take the last one:
        # hues are on a circle, and we would be oversampling the origin
        hues = np.tile(np.linspace(0, 1, num_hues+1)[:-1], (height, 1))

        sats = np.hstack(
            (np.linspace(0.1, 0.6, sat_range), np.ones(1 + light_range)))
        sats = np.tile(np.atleast_2d(sats).T, (1, num_hues))

        lights = np.hstack(
            (np.ones(1 + sat_range), np.linspace(.8, 0.3, light_range)))
        lights = np.tile(np.atleast_2d(lights).T, (1, num_hues))

        colors = hsv2rgb(np.dstack((hues, sats, lights)))
        grays = np.tile(np.atleast_3d(np.linspace(0, 1, num_hues)), (1, 1, 3))

        self.rgb_image = np.vstack((colors, grays))

        # Make a nice histogram ordering of the hues and grays: order is dif
        h, w, d = colors.shape
        color_array = colors.T.reshape((d, w * h)).T
        h, w, d = grays.shape
        gray_array = grays.T.reshape((d, w * h)).T
        
        self.rgb_array = np.vstack((color_array, gray_array))
        self.lab_array = rgb2lab(self.rgb_array[None, :, :]).squeeze()
        self.hex_list = [rgb2hex(row) for row in self.rgb_array]
        #assert(np.all(self.rgb_array == self.rgb_array[None, :, :].squeeze()))


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
        with open(os.path.join(dirname, 'palette.json'), 'w') as f:
            json.dump(self.hex_list, f)
        with open(os.path.join(dirname, 'palette.html'), 'w') as f:
            f.write(get_palette_html())

"""
The back-end of the multicolor image search.
"""

import os
import cgi
import simplejson as json
import cPickle
import numpy as np
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.color import rgb2lab, lab2rgb
from sklearn.utils import shuffle
from sklearn.metrics import euclidean_distances
from scipy.misc import imresize
from IPython import embed
import pyflann

def hex2rgb(hexcolor_str):
  """
  Convert string containing an HTML-encoded color  to an RGB tuple.

  >>> hex2rgb('#ffffff') 
  (255, 255, 255)
  >>> hex2rgb('33cc00')
  (51, 204, 0)
  """
  hexcolor = int(hexcolor_str.strip('#'), 16)
  r = ( hexcolor >> 16 ) & 0xff
  g = ( hexcolor >> 8 ) & 0xff
  b = hexcolor & 0xff
  return (r, g, b)

def lab_palette(hex_palette):
  """
  Convert a list of hex color name strings to a (K,3) array of Lab colors.
  """
  colors = [hex2rgb(hexcolor_str) for hexcolor_str in hex_palette]
  return rgb2lab(np.array(colors, ndmin=3)/255.).squeeze()

def plot_histogram(palette, palette_histogram):
  """
  Return histogram plot of palette colors.
  """
  pass

class Image(object):
  """
  Encapsulation of methods to extract information (e.g. color) from images.
  """

  MAX_DIMENSION = 300 # pixels

  def __init__(self, image_filename, max_dimension=MAX_DIMENSION):
    """
    Read the image at the URL, and make sure it's in RGB format.
    """
    img = img_as_float(imread(image_filename))

    # grayscale
    if img.ndim == 2:
      img = np.tile(img,(1,1,3))
    
    # with alpha
    # TODO: be smart here, but for now simply remove alpha channel
    if img.ndim == 4:
      img = img[:,:,:3]
    h, w, d = tuple(img.shape)
    assert(d == 3)

    # downsample image if needed
    f = None
    img_r = img
    resize_factor = 1
    if max(w,h) > max_dimension:
      resize_factor = float(max_dimension) / max(w,h)
      img_r = imresize(img, resize_factor, 'nearest')
      h, w, d = tuple(img_r.shape)

    # convert to Lab color space and reshape
    self.lab_array = rgb2lab(img_r).reshape((h*w, d))
    self.height, self.width, self.depth = tuple(img.shape)
    self.resize_factor = resize_factor
    self.filename = os.path.abspath(image_filename)

  def histogram_colors(self, palette, sigma=25):
    """
    Assign colors in the image to nearby colors in the palette, weighted by
    distance in Lab color space.

    Args:
      - palette ((K,3) ndarray): K is the number of colors, columns are Lab.
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

  def discard_array(self):
    """
    Drop the pixel data.
    """
    self.lab_array = None

class ImageCollection(object):
  """
  ImageCollection is the collection of images searchable by color.

  TODOS:
  - add flann
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

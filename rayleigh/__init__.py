"""
Rayleigh is an open-source system for quickly searching medium-sized image
collections by multiple colors given as a palette or derived from a query image.
"""


from palette import Palette

from image import Image, PaletteQuery

from collection import ImageCollection

from searchable_collection import \
  SearchableImageCollectionExact, \
  SearchableImageCollectionFLANN, \
  SearchableImageCollectionCKDTree

import util

"""
Test the image collection methods.
"""

from context import *

class TestCollection(unittest.TestCase):

  def test_flickr(self):
    """
    Load subset of Flickr 25K dataset.

    Download the MIRFLICKR datset from http://press.liacs.nl/mirflickr/.

    Make several lists of images, to test scaling.

    > find /Volumes/WD\ Data/mirflickr -name "*.jpg" | head -n 10 > images_10.txt
    > find /Volumes/WD\ Data/mirflickr -name "*.jpg" | head -n 100 > images_100.txt
    > find /Volumes/WD\ Data/mirflickr -name "*.jpg" | head -n 1000 > images_1000.txt
    > find /Volumes/WD\ Data/mirflickr -name "*.jpg" | head -n 10000 > images_10000.txt
    """
    image_list_filename = os.path.join(support_dirname, 'images_100.txt')
    with open(image_list_filename) as f:
      image_filenames = [x.strip() for x in f.readlines()]
    palette_filename = os.path.join(support_dirname, 'palette.json')
    hex_palette = json.load(open(palette_filename))
    
    ic = rayleigh.ImageCollection(hex_palette)
    ic.add_images(image_filenames)

    results = ic.search_by_image(image_filenames[0])
    ic.save('mir100.pickle')

if __name__ == '__main__':
  unittest.main()

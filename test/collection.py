"""
Test the image collection methods.
"""

from context import *

class TestCollection(unittest.TestCase):

  def test_synthetic(self):
    
    def save_synthetic_image(color, dirname):
      """
      Save a solid color image of the given hex color to the given directory.
      """
      filename = os.path.join(dirname, color+'.png')
      cmd = "convert -size 100x100 'xc:{color}' '{filename}'"
      os.system(cmd.format(**locals()))
      return filename
    
    dirname = skutil.makedirs(os.path.join(temp_dirname, 'synthetic_colors'))
    hex_palette = rayleigh.create_palette()
    filenames = [save_synthetic_image(color, dirname) for color in hex_palette]

    # save quantized versions
    for filename in filenames:
      img = rayleigh.Image(filename)
      q_filename = filename + '_q.png'
      img.quantize_to_palette(lab_palette, q_filename)


  def test_flickr(self):
    """
    Load subset of MIRFLICKR 25K [dataset](http://press.liacs.nl/mirflickr/).

    > find /Volumes/WD\ Data/mirflickr -name "*.jpg" | head -n 100 > mirflickr_100.txt
    """
    image_list_filename = os.path.join(support_dirname, 'images_10000.txt')
    with open(image_list_filename) as f:
      image_filenames = [x.strip() for x in f.readlines()]

    hex_palette = rayleigh.create_palette()
    
    ic = rayleigh.ImageCollection(hex_palette)
    ic.add_images(image_filenames)

    results = ic.search_by_image(image_filenames[0])
    ic.save('%s.pickle'%image_list_filename)

if __name__ == '__main__':
  unittest.main()

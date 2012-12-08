"""
Test the image collection methods.
"""

from context import *
import matplotlib.pyplot as plt

class TestImage(unittest.TestCase):
  def test_image(self):
    """
    Test color functionality of the Image class.
    """
    def plot_palette_histogram(image_filename):
      img = rayleigh.Image(os.path.join(support_dirname, image_filename))
      palette_filename = os.path.join(support_dirname, 'palette.json')
      hex_palette = json.load(open(palette_filename))
      lab_palette = rayleigh.lab_palette(hex_palette)
      lab_hist = img.histogram_colors(lab_palette)
      plt.clf()
      plt.bar(range(len(lab_hist)), lab_hist, color=hex_palette, edgecolor='black')
      plt.savefig(image_filename+'_hist.png')
    
    image_filenames = ['lightgreen.png', 'palette.png', 'cool.jpg', 'landscape.jpg', 'luna.jpg', 'lion.jpg']
    for image_filename in image_filenames:
      plot_palette_histogram(image_filename)

if __name__ == '__main__':
  unittest.main()

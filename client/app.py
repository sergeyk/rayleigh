import numpy as np
import simplejson as json
from bson import json_util
import cStringIO as StringIO
from skimage.io import imsave
from flask import Flask, render_template, request, make_response, send_file
import sys
import os

repo_dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_dirname)
import rayleigh
import rayleigh.util as util

app = Flask(__name__)
app.debug = False  # TODO: make sure this is False in production


def make_json_response(body, status_code=200):
    resp = make_response(json.dumps(body, default=json_util.default))
    resp.status_code = status_code
    resp.mimetype = 'application/json'
    return resp


def bad_id_response():
    return make_json_response({'message': 'invalid id'}, 400)


"""
Load the Searchable Image Collections that can be used to search.
"""
sics = {
    'chi_square_exact_8': rayleigh.SearchableImageCollectionExact.load(os.path.join(
        repo_dirname, 'data/mirflickr_25K_exact_chi_square_8_0.pickle')),

    'chi_square_exact_16': rayleigh.SearchableImageCollectionExact.load(os.path.join(
        repo_dirname, 'data/mirflickr_25K_exact_chi_square_16_0.pickle')),
    
    'chi_square_flann_8': rayleigh.SearchableImageCollectionFLANN.load(os.path.join(
        repo_dirname, 'data/mirflickr_25K_flann_chi_square_8_0.pickle'))
}

default_sic_type = "chi_square_exact_8"

"""
Set the smoothing parameter applied to the color palette queries.
"""
sigma = 16


@app.route('/sic_types')
def sic_types():
    return make_json_response(sics.keys())


@app.route('/')
def index():
    return render_template('index.html')


def parse_request():
    """
    Parse the GET request string for the SIC type and query palette information.

    Returns
    -------
    pq : rayleigh.PaletteQuery
    """
    colors = request.args.get('colors', '').split(',')
    values = request.args.get('values', None)
    if values is None:
        values = np.ones(len(colors), 'float') / len(colors)
    else:
        values = np.array(values, 'float') / sum(values)
    pq = rayleigh.PaletteQuery(dict(zip(colors, values)))

    return pq


@app.route('/<sic_type>/search')
def search_with_palette(sic_type):
    """
    Get all images sorted by distance to the color histogram given.
    Also provide the palette image and the histogram of the query.
    """
    sic = sics[sic_type]
    palette_query = parse_request()
    color_hist = palette_query.histogram_colors_smoothed(
        sic.ic.palette, sigma=sigma, direct=False)
    b64_hist = util.output_histogram_base64(color_hist, sic.ic.palette)
    results = sic.search_by_color_hist(color_hist, 80)
    return make_json_response({'results': results, 'pq_hist': b64_hist})


# TODO: add sigma parameter
@app.route('/image_histogram/<image_id>.png')
def get_image_histogram(image_id):
    sic = sics[default_sic_type]
    hist = sic.ic.hists[int(image_id), :]
    strIO = rayleigh.util.output_plot_for_flask(hist, sic.ic.palette)
    return send_file(strIO, mimetype='image/png')


@app.route('/palette_image/<image_id>.png')
def get_palette_image(image_id):
    sic = sics[default_sic_type]
    hist = sic.ic.hists[int(image_id), :]
    img = rayleigh.util.color_hist_to_palette_image(hist, sic.ic.palette)
    strIO = StringIO.StringIO()
    imsave(strIO, img, plugin='pil', format_str='png')
    strIO.seek(0)
    return send_file(strIO, mimetype='image/png')


@app.route('/similar_to/<sic_type>/<image_id>')
def get_similar_images(sic_type, image_id):
    sic = sics[sic_type]
    hist = sic.ic.hists[int(image_id), :]
    data = sic.search_by_color_hist(hist, 80)
    return make_json_response(data)


if __name__ == '__main__':
    app.run()

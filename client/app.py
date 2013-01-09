import numpy as np
import simplejson as json
from bson import json_util
from flask import Flask, render_template, request, make_response
import sys
import os


repo_dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_dirname)
import rayleigh
import rayleigh.util as util


def make_json_response(body, status_code=200):
    resp = make_response(json.dumps(body, default=json_util.default))
    resp.status_code = status_code
    resp.mimetype = 'application/json'
    return resp


def bad_id_response():
    return make_json_response({'message': 'invalid id'}, 400)


app = Flask(__name__)
app.debug = False  # TODO: make sure this is False in production


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

default_sic_type = "chi_square_exact"

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

    Returns:
        - pq (PaletteQuery)
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


@app.route('/image_histogram/<sic_type>/<int:ind>.png')
def get_image_histogram(sic_type, ind):
    sic = sics[sic_type]
    hist = sic.ic.hists[int(ind), :]
    png_output = rayleigh.util.output_histogram_for_flask(hist, sic.ic.palette)
    response = make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response


@app.route('/similar_to/<sic_type>/<int:ind>')
def get_similar_images(sic_type, ind):
    sic = sics[sic_type]
    hist = sic.ic.hists[int(ind), :]
    data = sic.search_by_color_hist(hist, 80)
    return make_json_response(data)


if __name__ == '__main__':
    app.run()

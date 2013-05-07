import matplotlib
matplotlib.use('Agg')

import numpy as np
import simplejson as json
from bson import json_util
import cStringIO as StringIO
from skimage.io import imsave
from flask import Flask, render_template, request, make_response, send_file, redirect, url_for, Markup
import sys
import os
from urllib2 import unquote

repo_dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_dirname)
import rayleigh
import rayleigh.util as util

app = Flask(__name__)
app.debug = False  # TODO: make sure this is False in production
#app.debug = True


def make_json_response(body, status_code=200):
    resp = make_response(json.dumps(body, default=json_util.default))
    resp.status_code = status_code
    resp.mimetype = 'application/json'
    return resp


"""
Load the Searchable Image Collections that can be used to search.
"""
fname_dict = {
    'data/flickr_100K_exact_chi_square_8_0.pickle': (
        'Chi-square, sigma=8, Exact', rayleigh.SearchableImageCollectionExact),
    'data/flickr_100K_exact_chi_square_16_0.pickle': (
        'Chi-square, sigma=16, Exact', rayleigh.SearchableImageCollectionExact),
    'data/flickr_100K_flann_chi_square_16_0.pickle': (
        'Chi-square, sigma=16, FLANN', rayleigh.SearchableImageCollectionFLANN),
    'data/flickr_100K_exact_manhattan_8_0.pickle': (
        'Manhattan, sigma=8, Exact', rayleigh.SearchableImageCollectionExact),
    'data/flickr_100K_exact_manhattan_16_0.pickle': (
        'Manhattan, sigma=16, Exact', rayleigh.SearchableImageCollectionExact),
    'data/flickr_100K_flann_manhattan_16_0.pickle': (
        'Manhattan, sigma=16, FLANN', rayleigh.SearchableImageCollectionFLANN),
    'data/flickr_100K_exact_euclidean_8_0.pickle': (
        'Euclidean, sigma=8, Exact', rayleigh.SearchableImageCollectionExact),
    'data/flickr_100K_exact_euclidean_16_0.pickle': (
        'Euclidean, sigma=16, Exact', rayleigh.SearchableImageCollectionExact),
    'data/flickr_100K_flann_euclidean_16_0.pickle': (
        'Euclidean, sigma=16, FLANN', rayleigh.SearchableImageCollectionFLANN),
}

sics = {}
for fname in fname_dict.keys():
    full_fname = os.path.join(repo_dirname, fname)
    if os.path.exists(full_fname):
        name, cls = fname_dict[fname]
        sics[name] = cls.load(os.path.join(full_fname))
default_sic_type = sics.keys()[0]

"""
Set the default smoothing parameter applied to the color palette queries.
"""
sigmas = [8, 16, 20]
default_sigma = 16


@app.route('/')
def index():
    return redirect(url_for(
        'search_by_palette', sic_type=default_sic_type, sigma=default_sigma))


def parse_colors_and_values():
    """
    Parse the GET request string for the palette query information.
    The query string looks like "?colors=#ffffff,#000000&values=0.5,0.5

    Returns
    -------
    colors : dict of hex color strings to their nromalized value, or None
    """
    colors = request.args.get('colors', '')
    print(request.args)

    if len(colors) < 1:
        return None
    colors = unquote(colors).split(',')

    values = request.args.get('values', '')
    if len(values) < 1:
        values = np.ones(len(colors), 'float') / len(colors)
    else:
        values = np.array(unquote(values).split(','), 'float') / sum(values)
    
    assert(len(values) == len(colors))
    return dict(zip(colors, values.tolist()))


@app.route('/search_by_palette')
def search_by_palette_default():
    return redirect(url_for(
        'search_by_palette', sic_type=default_sic_type, sigma=default_sigma))


@app.route('/search_by_palette/<sic_type>/<int:sigma>')
def search_by_palette(sic_type, sigma):
    colors = parse_colors_and_values()
    return render_template(
        'search_by_palette.html',
        sic_types=sorted(sics.keys()), sic_type=sic_type,
        sigmas=sigmas, sigma=sigma,
        colors=Markup(json.dumps(colors)))


@app.route('/search_by_palette_json/<sic_type>/<int:sigma>')
def search_by_palette_json(sic_type, sigma):
    sic = sics[sic_type]
    colors = parse_colors_and_values()
    if colors is None:
        return make_json_response({'message': 'no request data'}, 400)
    pq = rayleigh.PaletteQuery(colors)
    color_hist = util.histogram_colors_smoothed(
        pq.lab_array, sic.ic.palette, sigma=sigma, direct=False)
    b64_hist = util.output_histogram_base64(color_hist, sic.ic.palette)
    results, time_elapsed = sic.search_by_color_hist(color_hist, 80)
    return make_json_response({
        'results': results, 'time_elapsed': time_elapsed, 'pq_hist': b64_hist})


@app.route('/search_by_image/<sic_type>/<image_id>')
def search_by_image(sic_type, image_id):
    # TODO: don't rely on the two methods below in the template, but render
    # images directly here.
    image = sics[sic_type].ic.get_image(image_id, no_hist=True)
    return render_template(
        'search_by_image.html',
        sic_types=sorted(sics.keys()), sic_type=sic_type,
        image_url=image['url'], image_id=image_id)


@app.route('/search_by_image_json/<sic_type>/<image_id>')
def search_by_image_json(sic_type, image_id):
    sic = sics[sic_type]
    query_data, results, time_elapsed = sic.search_by_image_in_dataset(image_id, 80)
    return make_json_response({
        'results': results, 'time_elapsed': time_elapsed})


@app.route('/image_histogram/<sic_type>/<int:sigma>/<image_id>.png')
def get_image_histogram(sic_type, sigma, image_id):
    """
    Return png of the image histogram.

    Parameters
    ----------
    sic_type: string

    sigma: int
        If given as 0, return histogram as smoothed by the SIC.
        If given as 1, return unsmoothed histogram.
        If otherwise given, get the unsmoothed histogram and smooth manually.

    image_id: string

    Returns
    -------
    strIO: binary of a png file.
    """
    sic = sics[sic_type]
    if sigma == 0:
        hist = sic.get_image_hist(image_id)
    else:
        hist = sic.ic.get_image(image_id)['hist']
        if sigma != 1:
            hist = util.smooth_histogram(hist, sic.ic.palette, sigma)
    strIO = rayleigh.util.output_plot_for_flask(hist, sic.ic.palette)
    return send_file(strIO, mimetype='image/png')


@app.route('/palette_image/<sic_type>/<image_id>.png')
def get_palette_image(sic_type, image_id):
    sic = sics[sic_type]
    hist = sic.ic.get_image(image_id)['hist']
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

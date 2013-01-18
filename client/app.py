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
app.debug = True


def make_json_response(body, status_code=200):
    resp = make_response(json.dumps(body, default=json_util.default))
    resp.status_code = status_code
    resp.mimetype = 'application/json'
    return resp


"""
Load the Searchable Image Collections that can be used to search.
"""
sics = {
    'chi_square_exact_8': rayleigh.SearchableImageCollectionExact.load(os.path.join(
        repo_dirname, 'data/mirflickr_25K_exact_chi_square_8_0.pickle')),

    'chi_square_exact_16': rayleigh.SearchableImageCollectionExact.load(os.path.join(
        repo_dirname, 'data/mirflickr_25K_exact_chi_square_16_0.pickle')),
    
    # 'chi_square_flann_8': rayleigh.SearchableImageCollectionFLANN.load(os.path.join(
    #     repo_dirname, 'data/mirflickr_25K_flann_chi_square_8_0.pickle'))
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
    return redirect(url_for('search_with_palette'))


def parse_sic_type():
    """
    Parse the GET request string for the SIC type.

    Returns
    -------
    sic_type : string
    """
    sic_type = request.args.get('sic_type', '')
    if len(sic_type) < 1:
        return default_sic_type
    return sic_type


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


@app.route('/search_with_palette')
def search_with_palette():
    sic_type = parse_sic_type()
    colors = parse_colors_and_values()

    def checked(st):
        return 'checked' if st == sic_type else ''
    
    return render_template(
        'index.html',
        sic_types=sics.keys(),
        sic_type=sic_type, checked=checked,
        colors=Markup(json.dumps(colors)))


@app.route('/search_with_palette_json')
def search_with_palette_json():
    sic = sics[parse_sic_type()]
    colors = parse_colors_and_values()
    if colors is None:
        return make_json_response({'message': 'no request data'}, 400)

    print(colors)
    pq = rayleigh.PaletteQuery(colors)
    color_hist = pq.histogram_colors_smoothed(
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

import numpy as np
import simplejson as json
from bson import json_util
from flask import Flask, render_template, request, make_response
import sys
import os


repo_dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_dirname)
import rayleigh


def make_json_response(body, status_code=200):
    resp = make_response(json.dumps(body, default=json_util.default))
    resp.status_code = status_code
    resp.mimetype = 'application/json'
    return resp


def bad_id_response():
    return make_json_response({'message': 'invalid id'}, 400)


app = Flask(__name__)
app.debug = True  # TODO: comment out in production


sic = rayleigh.SearchableImageCollectionExact.load(
    '../data/mirflickr_1K_exact_euclidean_0.pickle')


@app.route('/')
def index():
    return render_template('index.html')


def parse_colors_and_values(request):
    print("Request args: {}".format(request.args))
    colors = request.args.get('colors', '').split(',')
    values = request.args.get('values', None)
    if values is None:
        values = np.ones(len(colors), 'float') / len(colors)
    else:
        values = np.array(values, 'float') / sum(values)
    palette_query = rayleigh.PaletteQuery(colors, values)
    color_hist = palette_query.histogram_colors(sic.ic.palette, sigma=15)
    return color_hist


# TODO: be able to take URLs like #colors=71b99e,5486bd;weights=50,50;
@app.route('/images')
def get_images():
    """
    Get all images sorted by distance to the color histogram given.
    """
    color_hist = parse_colors_and_values(request)
    data = sic.search_by_color_hist(color_hist, 80)
    print("Sending: ", data)
    return make_json_response(data)


@app.route('/image_histogram/<ind>.png')
def get_image_histogram(ind):
    hist = sic.ic.hists[int(ind), :]
    png_output = rayleigh.util.plot_histogram_flask(hist, sic.ic.palette)
    response = make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response


@app.route('/query_histogram')
def get_palette_query_histogram():
    hist = parse_colors_and_values(request)
    data = rayleigh.util.plot_histogram_html(hist, sic.ic.palette)
    return data


@app.route('/similar_to/<ind>')
def get_similar_images(ind):
    hist = sic.ic.hists[int(ind), :]
    data = sic.search_by_color_hist(hist, 80)
    return make_json_response(data)


@app.route('/stats/', methods=['GET'])
def get_statistics():
    # TODO
    data = None
    return make_json_response(data)


if __name__ == '__main__':
    app.run()

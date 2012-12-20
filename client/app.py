import numpy as np
import pymongo
import json
from bson import json_util
from bson.objectid import ObjectId
from flask import Flask, render_template, request, make_response

# make sure rayleigh is on the path
import sys
import os
repo_dirname = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.insert(0, repo_dirname)

from IPython import embed

import rayleigh

### Util methods

def make_json_response(body, status_code=200):
  resp = make_response(json.dumps(body, default=json_util.default))
  resp.status_code = status_code
  resp.mimetype = 'application/json'
  return resp

def bad_id_response():
  return make_json_response({'message': 'invalid id'}, 400)

### Route methods

app = Flask(__name__)
app.debug = True # TODO: comment out in production

ic = rayleigh.ImageCollection.load('../data/mir10000.pickle')

@app.route('/')
def index():
  """
  Our single-page Backbone.js app.
  """
  return render_template('index.html')

# @app.route('/images/', methods=['GET'])
# def get_all_images():
#   """
#   Get all images.
#   """
#   print("HERE@")
#   data = None
#   return make_json_response(data)

# TODO: add ability to search by image

# TODO: be able to take URLs like #colors=71b99e,5486bd;weights=50,50;
@app.route('/images')
def get_images():
  """
  Get all images sorted by distance to the color histogram given.
  """
  print("Request args: %s"%request.args)
  colors = request.args.get('colors', '').split(',')
  values = request.args.get('values', None)
  if values is None:
    values = np.ones(len(colors), 'float') / len(colors)
  else:
    values = np.array(values, 'float') / sum(values)

  lab_colors = rayleigh.lab_palette(colors)
  color_hist = rayleigh.smoothed_histogram(ic.lab_palette, lab_colors)
  data = ic.search_by_color_hist(color_hist)

  #data = ic.search_by_image('../test/support/luna.jpg')

  #data = {'colors': colors, 'values': values}
  print("Sending: ", data)
  return make_json_response(data)

@app.route('/stats/', methods=['GET'])
def get_statistics():
  """
  Get all images.
  """
  data = None
  return make_json_response(data)

if __name__ == '__main__':
  app.run()
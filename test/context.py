import sys
import os
repo_dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_dirname)

support_dirname = os.path.join(repo_dirname, 'test', 'support')
temp_dirname = os.path.join(repo_dirname, 'test', '_temp')

import simplejson as json
import unittest
from numpy.testing import *
import numpy as np
from IPython import embed

from skpyutils import skutil

import rayleigh
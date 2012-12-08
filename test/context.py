import sys
import os
repo_dirname = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.insert(0, repo_dirname)

support_dirname = os.path.join(repo_dirname, 'test', 'support')

import simplejson as json
import unittest
from numpy.testing import *
import numpy as np
from IPython import embed

import rayleigh
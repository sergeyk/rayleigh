"""
ImageCollection stores color information about images and exposes a method to
add images to it, with support for parallel processing.
The datastore is MongoDB, so a server must be running (launch with the settings
in mongo.conf).
"""
from __future__ import print_function
import sys
import cPickle
from warnings import warn
import numpy as np
import IPython.parallel as parallel
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from bson import Binary
import rayleigh
from skpyutils import TicToc
tt = TicToc()


def get_mongodb_collection():
    """
    Establish connection to MongoDB and return the relevant collection.

    Returns
    -------
    collection : pymongo.Collection
        Pymongo Collection of images and their histograms.
    """
    try:
        connection = MongoClient('localhost', 27666)
    except ConnectionFailure:
        raise Exception("Cannot instantiate ImageCollection without \
                         a MongoDB server running on port 27666")
    return connection.image_collection.images


# For parallel execution, function must be in module scope
collection = get_mongodb_collection()


def process_image(args):
    """
    Returns
    -------
    success : boolean
    """
    image_url, image_id, palette = args
    try:
        # Check if the image with this id already exists in the database.
        if collection.find({'id': image_id}).count() > 0:
            return True
        img = rayleigh.Image(image_url, image_id)
        hist = rayleigh.util.histogram_colors_strict(img.lab_array, palette)
        bson_hist = Binary(cPickle.dumps(hist, protocol=2))
        img_data = dict(img.as_dict().items() + {'hist': bson_hist}.items())
        collection.insert(img_data)
        return True
    except Exception as e:
        print("process_image encountered error: {}".format(e), file=sys.stderr)
        return False


class ImageCollection(object):
    """
    Initalize an empty ImageCollection with a color palette that will be
    used to extract color information from images.

    Parameters
    ----------
    palette : Palette
        Palette object representing the accepted colors.
    """

    def __init__(self, palette):
        self.palette = palette
        self.images = []
        self.hists = np.zeros((0, len(self.palette.hex_list)))

    @staticmethod
    def load(filename):
        """
        Load ImageCollection from filename.
        """
        return cPickle.load(open(filename))

    def save(self, filename):
        """
        Save self to filename.
        """
        cPickle.dump(self, open(filename, 'w'), 2)

    def get_hists(self):
        """
        Return histograms of all images as a single numpy array.

        Returns
        -------
        hists : (N,K) ndarray
            where N is the number of images in the database and K is the number
            of colors in the palette.
        """
        # TODO: scale this to larger datasets by using PyTables
        # http://www.pytables.org/moin/HowToUse
        cursor = collection.find()
        return np.array([cPickle.loads(image['hist']) for image in cursor])

    def get_image(self, image_id, no_hist=False):
        """
        Return information about the image at id, or None if it doesn't exist.

        Parameters
        ----------
        image_id : string
        no_hist : boolean
            If True, does not return the histogram, only the image metadata.

        Returns
        -------
        image : dict, or None
            information in database for this image id.
        """
        if no_hist:
            results = collection.find({'id': image_id}, fields={'hist': False})
        else:
            results = collection.find({'id': image_id})
        if results.count() == 1:
            r = results[0]
            if 'hist' in r:
                r['hist'] = cPickle.loads(r['hist'])
            return r
        elif results.count() == 0:
            return None
        else:
            raise("Should never be more than one result for the same id.")

    def get_id_ind_map(self):
        """
        Return dict of id to index and index to id.
        """
        ids = [d['id'] for d in collection.find()]
        ids_to_ind = zip(ids, range(len(ids)))
        ind_to_ids = zip(range(len(ids)), ids)
        return dict(ids_to_ind + ind_to_ids)

    def add_images(self, image_urls, image_ids=None):
        """
        Add all images in a list of URLs.
        If ipcluster is running, load images in parallel.

        Parameters
        ----------
        image_urls : list
        image_ids : list, optional
            If given, images are stored with the given ids.
            If None, the index of the image in the dataset is its id.
        """
        collection.ensure_index('id')

        # Construct the arguments list due to IPython.parallel's pickling
        if image_ids is None:
            jobs = [(url, None, self.palette) for url in image_urls]
        else:
            jobs = [(url, _id, self.palette) for url, _id in zip(image_urls, image_ids)]

        print("Loading images...")
        tt = TicToc()
        parallelized = False
        try:
            rc = parallel.Client()
            lview = rc.load_balanced_view()
            parallelized = True
        except:
            warn(Warning("Launch an IPython cluster to parallelize \
                           ImageCollection loading."))
            
        if parallelized:
            results = lview.map(process_image, jobs)
            results.wait_interactive()
        else:
            results = map(process_image, jobs)

        collection.ensure_index('id')
        print("Finished inserting {} images in {:.3f} s".format(
            len(image_urls), tt.qtoc()))

import cgi
import abc
import cPickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import \
    manhattan_distances, euclidean_distances, additive_chi2_kernel
import pyflann
from scipy.spatial import cKDTree

import rayleigh

from skpyutils import TicToc
tt = TicToc()


class SearchableImageCollection(object):
    """
    Methods to search an ImageCollection with brute force, exhaustive search.
    """

    @staticmethod
    def load(filename):
        """
        Load ImageCollection from filename and prepare the data structures.
        """
        return cPickle.load(open(filename))

    def save(self, filename):
        """
        Save self to filename.
        """
        cPickle.dump(self, open(filename, 'w'), 2)

    def __init__(self, image_collection, distance_metric, num_dimensions):
        """
        Initialize with a rayleigh.ImageCollection, a distance_metric, and the
        number of dimensions to reduce the histograms to.

        Args:
            - image_collection (rayleigh.ImageCollection)

            - distance_metric (string): must be in self.DISTANCE_METRICS

            - num_dimensions (int): number of dimensions to reduce
                the histograms to, using PCA. If 0, do not reduce dimensions.
        """
        self.ic = image_collection

        if distance_metric not in self.DISTANCE_METRICS:
            raise Exception("Unsupported distance metric.")
        self.distance_metric = distance_metric

        self.num_dimensions = num_dimensions
        self.hists_reduced = self.ic.hists
        if self.num_dimensions > 0:
            self.reduce_dimensionality()

    def reduce_dimensionality(self):
        """
        Compute and store PCA dimensionality-reduced histograms.
        """
        tt.tic('reduce_dimensionality')
        self.pca = PCA(n_components=self.num_dimensions, whiten=True)
        self.pca.fit(self.ic.hists)
        self.hists_reduced = self.pca.transform(self.ic.hists)
        tt.toc('reduce_dimensionality')

    def search_by_image_in_dataset(self, img_ind, num=20):
        """
        Search images in database for similarity to the image at img_ind in the
        dataset.

        See search_by_color_hist().
        """
        query_img = self.ic.images[img_ind]
        color_hist = self.ic.hists[img_ind, :]
        return query_img.as_dict(), self.search_by_color_hist(color_hist)

    def search_by_image(self, image_filename, num=20):
        """
        Search images in database by color similarity to image.
        
        See search_by_color_hist().
        """
        query_img = rayleigh.Image(image_filename)
        color_hist = query_img.histogram_colors(self.ic.palette)
        return query_img.as_dict(), self.search_by_color_hist(color_hist)

    def search_by_color_hist(self, color_hist, num=20):
        """
        Search images in database by color similarity to the given histogram.

        Args:
            - color_hist (K, ndarray): histogram over the color palette.

            - num (int): number of nearest neighbors to return.

        Returns:
            - query_img (dict): info about the query image

            - results (list): list of dicts of nearest neighbors to query
        """
        if self.num_dimensions > 0:
            color_hist = self.pca.transform(color_hist)
        nn_ind, nn_dists = self.nn_ind(color_hist, num)
        results = []
        for ind, dist in zip(nn_ind, nn_dists):
            img = self.ic.images[ind]
            results.append({
                'width': img.orig_width, 'height': img.orig_height,
                'filename': cgi.escape(img.filename), 'distance': dist})
        return results

    @abc.abstractmethod
    def nn_ind(self, color_hist, num):
        """
        Return num closest nearest neighbors (potentially approximate) to the
        query color_hist, and the distances to them.

        Override this search method in extending classes.

        Args:
            - color_hist (K, ndarray): histogram over the color palette.

            - num (int): number of nearest neighbors to return.

        Returns:
            - nn_ind (num, ndarray): indices of the neighbors in the dataset.

            - nn_dists (num, ndarray): distances to these neighbors.
        """
        pass


class SearchableImageCollectionExact(SearchableImageCollection):
    """
    Search the image collection exhaustively (mainly through np.dot).
    """

    DISTANCE_METRICS = ['manhattan', 'euclidean']

    def nn_ind(self, color_hist, num):
        """
        Exact nearest neighbor seach through exhaustive comparison.
        """
        if self.distance_metric == 'manhattan':
            dists = manhattan_distances(color_hist, self.hists_reduced)
        elif self.distance_metric == 'euclidean':
            dists = euclidean_distances(color_hist, self.hists_reduced, squared=True)
        elif self.distance_metric == 'chi_square':
            dists = -additive_chi2_kernel(color_hist, self.hists_reduced)
        
        dists = dists.flatten()
        nn_ind = np.argsort(dists).flatten()[:num]
        nn_dists = dists[nn_ind]
        
        return nn_ind, nn_dists


class SearchableImageCollectionFLANN(SearchableImageCollection):
    """
    Search the image collection using the FLANN library for aNN indexing.
    
    The FLANN index is built with automatic tuning of the search algorithm,
    which can take a while (~90s on 25K images).
    """

    DISTANCE_METRICS = ['manhattan', 'euclidean', 'chi_square']

    @staticmethod
    def load(filename):
        # Saving the flann object results in memory errors, so we use its own
        # method to save its index in a separate file.
        sic = cPickle.load(open(filename))
        return sic.build_index(filename + '_flann_index')

    def save(self, filename):
        # See comment in load().
        flann = self.flann
        self.flann = None
        cPickle.dump(self, open(filename, 'w'), 2)
        flann.save_index(filename + '_flann_index')
        self.flann = flann

    def __init__(self, image_collection, distance_metric, dimensions):
        super(SearchableImageCollectionFLANN, self).__init__(
            image_collection, distance_metric, dimensions)
        self.build_index()

    def build_index(self, index_filename=None):
        tt.tic('build_index')
        pyflann.set_distance_type(self.distance_metric)
        self.flann = pyflann.FLANN()
        if index_filename:
            self.flann.load_index(index_filename, self.hists_reduced)
        else:
            self.params = self.flann.build_index(
                self.hists_reduced, algorithm='autotuned',
                sample_fraction=0.3, target_precision=.8,
                build_weight=0.01, memory_weight=0.)
        print(self.params)
        tt.toc('build_index')
        return self

    def nn_ind(self, color_hist, num):
        nn_ind, nn_dists = self.flann.nn_index(
            color_hist, num, checks=self.params['checks'])
        return nn_ind.flatten(), nn_dists.flatten()


class SearchableImageCollectionCKDTree(SearchableImageCollection):
    """
    Use the cKDTree data structure from scipy.spatial for the index.

    Parameters:
        - LEAF_SIZE (int): The number of points at which the algorithm switches
            over to brute-force.
        - EPS (non-negative float): Parameter for query(), such that the
            k-th returned value is guaranteed to be no further than (1 + eps)
            times the distance to the real k-th nearest neighbor.

    NOTE: These parameters have not been tuned.
    """

    DISTANCE_METRICS = ['manhattan', 'euclidean']
    Ps = {'manhattan': 1, 'euclidean': 2}
    LEAF_SIZE = 5
    EPSILON = 1

    @staticmethod
    def load(filename):
        return cPickle.load(open(filename)).build_index()

    def __init__(self, image_collection, distance_metric, dimensions):
        super(SearchableImageCollectionCKDTree, self).__init__(
            image_collection, distance_metric, dimensions)
        self.build_index()

    def build_index(self):
        tt.tic('build_index_ckdtree')
        self.ckdtree = cKDTree(self.hists_reduced, self.LEAF_SIZE)
        self.p = self.Ps[self.distance_metric]
        tt.toc('build_index_ckdtree')
        return self

    def nn_ind(self, color_hist, num):
        nn_dists, nn_ind = self.ckdtree.query(
            color_hist, num, eps=self.EPSILON, p=self.p)
        return nn_ind.flatten(), nn_dists.flatten()

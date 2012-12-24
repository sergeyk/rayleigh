import cgi
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


class SearchableImageCollectionExact(object):
    """
    Methods to search an ImageCollection.

    This is the parent class, implenting exact search.

    Parameters:
        - PCA_COMPS (int): number of dimensions that PCA
            will reduce all histograms to.
    """

    DISTANCE_METRICS = ['manhattan', 'euclidean', 'chi_square']
    PCA_COMPS = 30

    @staticmethod
    def load(filename):
        """
        Load ImageCollection from filename and prepare the data structures.
        """
        return cPickle.load(open(filename)).prepare()

    def save(self, filename):
        """
        Save self to filename.
        """
        cPickle.dump(self, open(filename, 'w'), 2)

    def __init__(self, image_collection, distance_metric):
        """
        Initialize with a rayleigh.ImageCollection and a distance_metric.
        """
        if distance_metric not in self.DISTANCE_METRICS:
            raise Exception("Unsupported distance metric.")
        self.distance_metric = distance_metric
        self.ic = image_collection
        self.prepare()

    def prepare(self):
        """
        Compute PCA dimensionality-reduced histograms.

        Returns:
            - self
        """
        print("SearchableImageCollection: preparing")
        tt.tic('prepare')

        # Reduce the number of dimensions of the data with PCA.
        self.pca = PCA(n_components=self.PCA_COMPS, whiten=True)
        self.pca.fit(self.ic.hists)

        # Store the reduced-dimension histograms and precompute
        # for fast euclidean distance.
        self.pca_hists = self.pca.transform(self.ic.hists)
        self.pca_hists_norm = (self.pca_hists ** 2).sum(1)

        # TODO: sparsify the PCA-reduced matrix
        
        #self.images = np.array(self.ic.images)
        tt.toc('prepare')
        return self

    def search_by_image_in_dataset(self, img_ind, num=20):
        color_hist = self.pca_hists[img_ind, :]
        nn_ind, nn_dists = self.nn_ind(color_hist, num)
        results = []
        for ind, dist in zip(nn_ind, nn_dists):
            img = self.ic.images[ind]
            results.append({
                'width': img.orig_width, 'height': img.orig_height,
                'filename': cgi.escape(img.filename), 'distance': dist})
        return self.ic.images[img_ind].as_dict(), results

    def search_by_image(self, image_filename, num=20):
        """
        Search images in database by color similarity to image, exhaustively.

        Returns:
            - query_img (dict): info about the query image

            - results (list): list of dicts of nearest neighbors to query
        """
        query_img = rayleigh.Image(image_filename)
        color_hist = query_img.histogram_colors(self.ic.palette)
        color_hist = self.pca.transform(color_hist)

        nn_ind, nn_dists = self.nn_ind(color_hist, num)

        results = []
        for ind, dist in zip(nn_ind, nn_dists):
            img = self.ic.images[ind]
            results.append({
                'width': img.orig_width, 'height': img.orig_height,
                'filename': cgi.escape(img.filename), 'distance': dist})
        return query_img.as_dict(), results

    def nn_ind(self, color_hist, num):
        """
        Exact nearest neighbor seach through exhaustive comparison.
        """
        # TODO: investigate further or get rid of this
        # # first, we rule out those histograms that have little overlap with us
        # num_overlaps = ((color_hist>0) & (self.hists>0)).sum(1)
        # ind = np.where(num_overlaps < np.mean(num_overlaps))[0]
        # hists = self.hists[ind]

        if self.distance_metric == 'manhattan':
            dists = manhattan_distances(color_hist, self.pca_hists)
        elif self.distance_metric == 'euclidean':
            dists = euclidean_distances(color_hist, self.pca_hists,
                                        self.pca_hists_norm, squared=True)
        elif self.distance_metric == 'chi_square':
            dists = -additive_chi2_kernel(color_hist, self.pca_hists)
        
        dists = dists.flatten()
        nn_ind = np.argsort(dists).flatten()[:num]
        nn_dists = dists[nn_ind]
        
        return nn_ind, nn_dists


class SearchableImageCollectionFLANN(SearchableImageCollectionExact):
    """
    Use the FLANN library for the index.
    """

    DISTANCE_METRICS = ['manhattan', 'euclidean', 'chi_square']

    def prepare(self):
        super(SearchableImageCollectionFLANN, self).prepare()
        
        tt.tic('build_index_flann')
        pyflann.set_distance_type(self.distance_metric)
        self.flann = pyflann.FLANN()
        self.params = self.flann.build_index(
            self.pca_hists, algorithm='autotuned',
            sample_fraction=0.3, target_precision=.8,
            build_weight=0.01, memory_weight=0.)
        print(self.params)
        tt.toc('build_index_flann')

        return self

    def nn_ind(self, color_hist, num):
        nn_ind, nn_dists = self.flann.nn_index(
            color_hist, num, checks=self.params['checks'])
        return nn_ind.flatten(), nn_dists.flatten()


class SearchableImageCollectionCKDTree(SearchableImageCollectionExact):
    """
    Use the cKDTree data structure from scipy.spatial for the index.

    Parameters:
        - LEAF_SIZE (int): The number of points at which the algorithm switches
            over to brute-force.
        - EPS (non-negative float): Parameter for query(), such that the
            k-th returned value is guaranteed to be no further than (1 + eps)
            times the distance to the real k-th nearest neighbor.
    """

    DISTANCE_METRICS = ['manhattan', 'euclidean']
    LEAF_SIZE = 5
    EPSILON = 1
    Ps = {'manhattan': 1, 'euclidean': 2}

    def prepare(self):
        super(SearchableImageCollectionCKDTree, self).prepare()

        tt.tic('build_index_ckdtree')
        self.ckdtree = cKDTree(self.pca_hists, self.LEAF_SIZE)
        self.p = self.Ps[self.distance_metric]
        tt.toc('build_index_ckdtree')

        return self

    def nn_ind(self, color_hist, num):
        nn_dists, nn_ind = self.ckdtree.query(
            color_hist, num, eps=self.EPSILON, p=self.p)
        return nn_ind.flatten(), nn_dists.flatten()

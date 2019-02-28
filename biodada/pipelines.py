import logging
import numpy

logger = logging.getLogger(__name__)


class PipelinesMixin(object):  # pytlint: disable=no-init
    """Scikit-learn pipelines for SequenceDataFrame objects."""

    def encoder(self, encoder='one-hot', dtype=None):
        """
        Return a transformer encoding sequence data into numeric.

        Parameters
        ----------
        encoder : 'one-hot', 'ordinal'
            sklearn encoder class: OneHotEncoder or OrdinalEncoder
        dtype : number type
            Default: numpy.float64 (one-hot), numpy.int8 (ordinal)

        Returns
        -------
        sklearn transformer

        """
        from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
        categories = [list(self.alphabet)] * (self.shape[1] - 1)
        if encoder == 'one-hot':
            return OneHotEncoder(
                categories=categories, dtype=dtype or numpy.float64)
        elif encoder == 'ordinal':
            return OrdinalEncoder(
                categories=categories, dtype=dtype or numpy.float64)

    def pca(self, n_components=3):
        """
        Return a transformer encoding sequence data into principal components.

        The pipeline steps are:
        - One-hot encoding of sequence data into a sparse matrix
        - Truncated SVD on the sparse data matrix.
          Return output data of dimensionality n_components + 3
        - PCA. Return output of dimensionality n_components.

        Attributes
        ----------
        n_components : int
            Number of components to keep.

        Returns
        -------
        sklearn transformer

        """
        from sklearn.pipeline import Pipeline
        from sklearn.decomposition import PCA as PCA
        from sklearn.decomposition import TruncatedSVD as tSVD
        return Pipeline([('encode', self.encoder()),
                         ('svd',
                          tSVD(
                              n_components=n_components + 3,
                              algorithm='arpack')),
                         ('pca', PCA(n_components=n_components))])

    def clustering(self, n_clusters, n_components=3):
        """Return a cluster estimator for sequence data.

        Agglomerative clustering of sequences in principal components space
        with Ward's method. Connectivity constraints from the k-neighbors
        graph.

        Parameters
        ----------
        n_custers : int
            The target number of clusters.
        n_components : int
            Number of principal components to keep in the dimensionality
            reduction step.

        Returns
        -------
        cluster pipeline object

        """
        from sklearn.pipeline import Pipeline
        from sklearn.neighbors import kneighbors_graph
        from sklearn.cluster import AgglomerativeClustering

        def connectivity(X):
            """Return k-neighbors graph."""
            return kneighbors_graph(X, n_neighbors=10, include_self=False)

        return Pipeline([('pca', self.pca(n_components=n_components)),
                         ('cluster',
                          AgglomerativeClustering(
                              n_clusters=n_clusters,
                              connectivity=connectivity,
                              linkage='ward'))])

    def classifier(self, n_neighbors=5):
        """Return a classifier for sequence data.

        sklearn KNeighborsClassifier (k-nearest neighbors vote)

        Parameters
        ----------
        n_neighbors : int, optional (default=5)
            Number of neighbors to use.

        Returns
        -------
        sklearn classifier
        """
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=n_neighbors)

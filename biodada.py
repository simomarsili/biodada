"""SequenceDataFrame class and related functions."""

import logging
import pkg_resources
from functools import wraps
import numpy
import pandas
from pandas import DataFrame
import gopen  # pylint: disable=import-error

project_name = 'biodada'
__version__ = pkg_resources.require(project_name)[0].version
__copyright__ = 'Copyright (C) 2019 Simone Marsili'
__license__ = 'BSD 3 clause'
__author__ = 'Simone Marsili <simo.marsili@gmail.com>'
__all__ = ['SequenceDataFrame', 'read_alignment', 'load', 'ALPHABETS']

logger = logging.getLogger(__name__)

ALPHABETS = {
    'protein': '-ACDEFGHIKLMNPQRSTVWY',
    'dna': '-ACGT',
    'rna': '-ACGU',
    'protein_u': '-ACDEFGHIKLMNPQRSTVWYBZX',
    'dna_u': '-ACGTRYMKWSBDHVN',
    'rna_u': '-ACGURYMKWSBDHVN',
}


def timeit(func):
    """Timeit decorator."""

    @wraps(func)
    def timed(*args, **kwargs):
        import time
        ts0 = time.time()
        result = func(*args, **kwargs)
        ts1 = time.time()
        logger.debug('%r: %2.4f secs', func, ts1 - ts0)
        return result

    return timed


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

    def classifier(self, n_neighbors=3):
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


class SequenceDataFrame(PipelinesMixin, DataFrame):  # pylint: disable=too-many-ancestors
    """
    In addition to the standard DataFrame constructor arguments,
    SequenceDataFrame also accepts the following keyword arguments:

    Parameters
    ----------
    alphabet : str
        Alphabet for the alignment. Default: None. See biodada.ALPHABETS.

    """

    _metadata = ['alphabet']

    def __init__(self, *args, **kwargs):

        alphabet = kwargs.pop('alphabet', None)

        logger.debug('init SequenceDataFrame, alphabet: %r', alphabet)

        if not hasattr(self, 'alphabet'):
            self.alphabet = None

        super().__init__(*args, **kwargs)

        if isinstance(self.columns, pandas.RangeIndex):
            lmax = max(len(x) for x in self[0])
            if lmax == 1:
                raise ValueError(
                    'The first data field must contain sequence identifiers')
            else:
                self.columns = ['id'] + list(range(self.shape[1] - 1))

        if alphabet:
            self.alphabet = alphabet

    @property
    def _constructor(self):
        return SequenceDataFrame

    @classmethod
    def from_sequence_records(cls, records, alphabet=None):
        """Return a SequenceDataFrame from records iterable."""
        return cls(([identifier] + list(sequence)
                    for identifier, sequence in records),
                   alphabet=alphabet)

    @property
    @timeit
    def data(self):
        """Sequences as array of one-letter codes."""
        return self.to_numpy(copy=False, dtype='U1')[:, 1:]

    def encoded(self, encoder='one-hot', dtype=None):
        """
        Encoded sequence data into numeric with different techniques.

        Parameters
        ----------
        encoder : 'one-hot', 'ordinal'
            encoder class:sklearn OneHotEncoder or OrdinalEncoder
        dtype : number type
            Default: numpy.float64 (one-hot), numpy.int8 (ordinal)

        Returns
        -------
        Encoded data : sparse matrix (one-hot) or numpy array (ordinal)
            Transformed array
        """
        encoder = self.encoder(encoder=encoder, dtype=dtype)
        return encoder.fit_transform(self.data)

    def principal_components(self, n_components=3, pca=None):
        """Return `n_components` principal components from PCA.

        See SequenceDataFrame.pca method for details.

        Attributes
        ----------
        n_components : int
            Number of components to keep.
        pca : a fitted PCA pipeline.
            If passed, just transform the data with `pca`

        Returns
        -------
        array-like, shape=(n_records, n_components)

        """
        from sklearn.exceptions import NotFittedError
        if not pca:
            pca = self.pca(n_components=n_components)
            pca.fit(self.data)
        try:
            return pca.transform(self.data)
        except NotFittedError:
            raise

    def clusters(self, n_clusters, n_components=3):
        """For a given number of clusters, return the cluster labels.

        See SequenceDataFrame.clustering for details.

        Parameters
        ----------
        n_custers : int
            The number of clusters.
        n_components : int
            Number of principal components to keep in the dimensionality
            reduction step.

        Returns
        -------
        cluster_labels : list

        """

        clustering = self.clustering(
            n_clusters=n_clusters, n_components=n_components)
        labels = clustering.fit_predict(self.data)
        return labels

    def classify(self, labeled_data, n_neighbors=3, transformer=None):
        """Classify records from labeled data."""
        classifier = self.classifier(n_neighbors=n_neighbors).fit(
            *labeled_data)
        if not transformer:
            X1 = self.data
        else:
            X1 = transformer.transform(self.data)
        return classifier.predict(X1)

    @timeit
    def save(self, target):
        """Save frame as bzipped json."""
        import json
        import codecs
        from bz2 import BZ2File
        dd = {}
        dd['columns'] = [-1] + list(self.columns)[1:]
        dd['records'] = list(self.records)
        dd['alphabet'] = self.alphabet
        handle = codecs.getwriter('utf8')(BZ2File(target, 'w'))
        json.dump(dd, fp=handle)

    @property
    def records(self):
        """Iterable of frame records."""
        return ((r[0], ''.join(r[1:]))
                for r in self.itertuples(index=False, name=None))


def parse(source, frmt, hmm=True):
    """Parse records from source."""
    import lilbio  # pylint: disable=import-error
    preprocess = lilbio.uppercase_only if hmm else None
    return lilbio.parse(source, frmt, func=preprocess)


def filter_redundant(records, threshold=0.9):
    """Return an iterable of non-redundant records."""
    import pcdhit  # pylint: disable=import-error
    return pcdhit.filter(records, threshold)


@timeit
def filter_gaps(frame, threshold=0.1):
    """Return a copy of frame after removing gappy records/positions."""
    import cleanset  # pylint: disable=import-error
    logger.debug('start filtering gaps')
    cleaner = cleanset.Cleaner(
        fna=threshold, condition=lambda x: x == '-' or x == 'X', axis=0.5)
    frame = cleaner.fit_transform(frame)
    logger.debug('stop filtering gaps')
    return frame


@timeit
def validate_alphabet(df):
    valid_records = []
    null = 0
    alphabet_set = set(df.alphabet)
    for index, row in enumerate(df.data):
        if set(row) <= alphabet_set:
            valid_records.append(index)
        else:
            null += 1
    logger.debug('null records: %r', null)
    # select valid records and reset row indexing
    df = df.iloc[valid_records]
    df.reset_index(drop=True, inplace=True)
    return df


def score_alphabet(alphabet, counts):
    """Score for alphabet given counts."""
    import math
    chars = set(alphabet) - set('*-')
    score = (sum([counts.get(a, 0) for a in chars]) / math.log(len(alphabet)))
    logger.debug('alphabet %r score %r', alphabet, score)
    return score


def guess_alphabet(records):
    """Guess alphabet from an iterable of records."""
    from collections import Counter
    data = numpy.array([list(record[1]) for record in records],
                       dtype='U1').flatten()
    counts = Counter(data)
    max_score = float('-inf')
    for key, alphabet in ALPHABETS.items():
        score = score_alphabet(alphabet, counts)
        if score > max_score:
            max_score = score
            guess = key
    logger.info('Alphabet guess: %r', guess)
    return ALPHABETS[guess]


@timeit
def read_alignment(source, fmt, hmm=True, c=0.9, g=0.1, alphabet=None):  # pylint: disable=too-many-arguments
    """Parse a pandas dataframe from an alignment file.

    Parameters
    ----------
    source : filepath or file-like
        The alignment file
    fmt : str
        Alignment format. Valid options are: 'fasta', 'stockholm'.
    hmm : boolean
        If True, return only uppercase symbols and {-', '*'} symbols.
    c : float
        Sequence identity threshold for redundancy filter. 0 < c < 1.
    g : float
        Gap fraction threshold for gap filter. 0 <= g <= 1.

    Returns
    -------
    dataframe
        A pandas dataframe.
    """
    import itertools
    # parse records
    records = parse(source, fmt, hmm=hmm)

    # filter redundant records via cdhit
    if c:
        records = filter_redundant(records, c)

    if not alphabet:
        records_head = itertools.islice(records, 50)
        alphabet = guess_alphabet(records_head)
        records = itertools.chain(records_head, records)

    # convert records to a dataframe
    df = SequenceDataFrame.from_sequence_records(records, alphabet=alphabet)

    # check alphabet consistency
    df = validate_alphabet(df)

    # reduce gappy records/positions
    if g:
        df = filter_gaps(df, g)

    return df


@timeit
def load(source):
    """Load a frame as bzipped json."""
    import json
    with gopen.readable(source) as fp:
        dd = json.load(fp)
    df = SequenceDataFrame(([identifier] + list(sequence)
                            for identifier, sequence in dd['records']),
                           columns=dd['columns'],
                           alphabet=dd['alphabet'])
    # sort rows/columns by index
    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, inplace=True)
    df.columns = ['id'] + list(df.columns)[1:]
    return df


def scatterplot(X,  # pylint: disable=too-many-arguments
                fig_size=(8, 6),
                n_points=False,
                size=10,
                color=None,
                ax=None):
    """
    Scatter plot of points in X.

    Return Axes object.
    """
    import matplotlib.pyplot as plt
    n, p = X.shape
    if p > 2:
        from mpl_toolkits.mplot3d import Axes3D
        if not ax:
            fig = plt.figure(1, figsize=fig_size)
            ax = Axes3D(fig, elev=-150, azim=110)
    if p == 2:
        fig, ax = plt.subplots(figsize=fig_size)
    elif p < 2:
        raise ValueError('X must be at least 2D')
    if n_points:
        idx = numpy.random.choice(range(n), size=n_points, replace=False)
        X = X[idx]
    XT = X.T
    kwargs = {}
    if color is not None:
        if n_points:
            kwargs['c'] = color[idx]
        else:
            kwargs['c'] = color
    kwargs['s'] = size
    if p == 2:
        ax.scatter(XT[0], XT[1], **kwargs)
    else:
        ax.scatter(XT[0], XT[1], XT[2], **kwargs)
    return ax

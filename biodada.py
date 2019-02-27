import logging
import pkg_resources
from functools import wraps
import numpy
import pandas
from pandas import DataFrame
import gopen

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


class PipelinesMixin:
    """Scikit-learn pipelines."""
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
            return OneHotEncoder(categories=categories,
                                 dtype=dtype or numpy.float64)
        elif encoder == 'ordinal':
            return OrdinalEncoder(categories=categories,
                                  dtype=dtype or numpy.float64)

    def pca(self, n_components=3):
        """Pipeline for principal component analysis.

        The transform steps are:
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
        pca pipeline object.

        """
        from sklearn.pipeline import Pipeline
        from sklearn.decomposition import PCA as PCA
        from sklearn.decomposition import TruncatedSVD as tSVD
        return Pipeline([
            ('encode', self.encoder()),
            ('svd', tSVD(n_components=n_components+3, algorithm='arpack')),
            ('pca', PCA(n_components=n_components))])

    def clustering(self, n_clusters, n_components=3):
        """Pipeline for cluster analysis.

        Cluster sequences in the space of principal components using
        agglomerative clustering with Ward's method. The clustering is
        performed with connectivity constraints from the k-neighbors grph.

        Parameters
        ----------
        n_custers : int
            The number of clusters.
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
            return kneighbors_graph(X, n_neighbors=10,
                                    include_self=False)
        return Pipeline([
            ('pca', self.pca(n_components=n_components)),
            ('cluster', AgglomerativeClustering(n_clusters=n_clusters,
                                                connectivity=connectivity,
                                                linkage='ward'))])

    def classifier(self, n_neighbors=3):
        from sklearn.pipeline import Pipeline
        from sklearn.neighbors import KNeighborsClassifier
        return Pipeline([
            ('classifier', KNeighborsClassifier(n_neighbors=3)),
        ])


class SequenceDataFrame(PipelinesMixin, DataFrame):

    _metadata = ['_alphabet']

    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, alphabet=None):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype,
                         copy=copy)
        self._alphabet = alphabet

    @property
    def _constructor(self):
        return SequenceDataFrame

    @classmethod
    def from_sequence_records(cls, records):
        df = cls(pandas.DataFrame.from_records(
            ((rec[0], *list(rec[1])) for rec in records)))
        # set dataframe column labels
        df.columns = ['id'] + list(range(df.shape[1]-1))
        return df

    @staticmethod
    def score_alphabet(alphabet, counts):
        import math
        chars = set(alphabet) - set('*-')
        score = (sum([counts.get(a, 0) for a in chars]) /
                 math.log(len(alphabet)))
        logger.debug('alphabet %r score %r', alphabet, score)
        return score

    @property
    @timeit
    def alphabet(self):
        if not self._alphabet:
            # guess
            from collections import Counter
            counts = Counter(self.iloc[:, 1:].head(50).values.flatten())
            max_score = float('-inf')
            for key, alphabet in ALPHABETS.items():
                score = self.score_alphabet(alphabet, counts)
                if score > max_score:
                    max_score = score
                    guess = key
            logger.info('Alphabet guess: %r', guess)
            self._alphabet = ALPHABETS[guess]
        return self._alphabet

    @alphabet.setter
    def alphabet(self, alphabet):
        self._alphabet = alphabet

    @property
    @timeit
    def data(self):
        return self.iloc[:, 1:]

    @timeit
    def as_array(self, copy=False):
        return self.data.to_numpy(dtype='U1', copy=copy)

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
        return encoder.transform(self.data)

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

        clustering = self.clustering(n_clusters=n_clusters,
                                     n_components=n_components)
        labels = clustering.fit_predict(self.data)
        return labels

    def classify(self, labeled_data, n_neighbors=3, transformer=None):
        classifier = self.classifier().fit(*labeled_data)
        if not transformer:
            X1 = self.data
        else:
            X1 = transformer.transform(self.data)
        return classifier.predict(X1)

    @timeit
    def save(self, target):
        import json
        import codecs
        from bz2 import BZ2File
        dd = {}
        dd['columns'] = [-1] + list(self.columns)[1:]
        dd['records'] = list(self.records)
        dd['alphabet'] = self.alphabet
        handle = codecs.getwriter('utf8')(BZ2File(target, 'w'))
        json.dump(dd, fp=handle)

    @timeit
    def to_fasta(self, fp):
        for header, seq in self.records:
            print('>%s\n%s' % (header, seq), file=fp)

    @property
    def records(self):
        return ((r[0], ''.join(r[1:])) for r in self.itertuples(index=False,
                                                                name=None))


def copy(df, *args, **kwargs):
    df1 = df.copy(*args, **kwargs)
    df1.alignment.alphabet = df.alignment.alphabet
    return df1


def parse(source, frmt, hmm=True):
    import lilbio
    preprocess = lilbio.uppercase_only if hmm else None
    return lilbio.parse(source, frmt, func=preprocess)


def filter_redundant(records, threshold=0.9):
    import pcdhit
    return pcdhit.filter(records, threshold)


@timeit
def filter_gaps(frame, threshold=0.1):
    import cleanset
    logger.debug('start filtering gaps')
    cleaner = cleanset.Cleaner(fna=threshold,
                               condition=lambda x: x == '-' or x == 'X',
                               axis=0.5)
    frame = cleaner.fit_transform(frame)
    logger.debug('stop filtering gaps')
    return frame


@timeit
def validate_alphabet(df):
    valid_records = []
    null = 0
    alphabet_set = set(df.alphabet)
    for index, row in enumerate(df.as_array()):
        if set(row) <= alphabet_set:
            valid_records.append(index)
        else:
            null += 1
    logger.debug('null records: %r', null)
    # select valid records and reset row indexing
    df = df.iloc[valid_records]
    df.reset_index(drop=True, inplace=True)
    return df


@timeit
def read_alignment(source, fmt, hmm=True, c=0.9, g=0.1, alphabet=None):
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
    # parse records
    records = parse(source, fmt, hmm=hmm)

    # filter redundant records via cdhit
    if c:
        records = filter_redundant(records, c)

    # convert records to a dataframe
    # df = SequenceDataFrame.from_sequence_records(records)
    df = SequenceDataFrame(([identifier] + list(sequence)
                            for identifier, sequence in records),
                           alphabet=alphabet)
    # set dataframe column labels
    df.columns = ['id'] + list(range(df.shape[1]-1))

    # reduce gappy records/positions
    if g:
        df = filter_gaps(df, g)

    # check alphabet consistency
    df = validate_alphabet(df)

    return df


@timeit
def load(source):
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


def scatterplot(X, fig_size=(8, 6), n_points=False, size=10, color=None,
                ax=None):
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

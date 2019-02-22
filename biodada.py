import logging
import pkg_resources
from functools import wraps
import numpy
import pandas

project_name = 'biodada'
__version__ = pkg_resources.require(project_name)[0].version
__copyright__ = 'Copyright (C) 2019 Simone Marsili'
__license__ = 'BSD 3 clause'
__author__ = 'Simone Marsili <simo.marsili@gmail.com>'
__all__ = ['dataframe', 'save', 'load']

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


@pandas.api.extensions.register_dataframe_accessor('alignment')
class Alignment():
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._alphabet = None
        self.encoder_pipe = None
        self.pca_pipe = None
        self.cluster_pipe = None

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
            counts = Counter(self._obj.iloc[:, 1:].head(50).values.flatten())
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
        return self._obj.iloc[:, 1:]

    @timeit
    def as_array(self, copy=False):
        return self.data.to_numpy(dtype='U1', copy=copy)

    @timeit
    def replace(self, encoding=None):
        if not encoding:
            encoding = {c: k for k, c
                        in enumerate(self.alignment.alphabet)}
        return self.replace(encoding)

    def encoder(self, encoder='one-hot', categories=None,
                dtype=None):
        """
        encoding: 'one-hot', 'ordinal'
        categories: None, auto or list of lists/arrays
        """
        from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
        n, p = self._obj.shape
        if not categories:
            categories = [list(self.alphabet)] * (p - 1)
        if encoder == 'one-hot':
            if dtype is None:
                dtype = numpy.float64
            enc = OneHotEncoder(categories=categories, dtype=dtype)
        elif encoder == 'ordinal':
            if dtype is None:
                dtype = numpy.int8
            enc = OrdinalEncoder(categories=categories, dtype=dtype)
        return enc

    def encoded(self, encoder='one-hot', categories=None,
                dtype=None):
        encoder = self.encoder(encoder=encoder, categories=categories,
                               dtype=dtype)
        self.encoder_pipe = encoder.fit(self.data)
        return encoder.transform(self.data)

    def pca(self, n_components=3):
        from sklearn.pipeline import Pipeline
        from sklearn.decomposition import PCA as PCA
        from sklearn.decomposition import TruncatedSVD as tSVD
        return Pipeline([
            ('encode', self.encoder()),
            ('svd', tSVD(n_components=n_components+3, algorithm='arpack')),
            ('pca', PCA(n_components=n_components))])

    def principal_components(self, n_components=3, pca=None):
        from sklearn.exceptions import NotFittedError
        if not pca:
            pca = self.pca(n_components=n_components)
            pca.fit(self.data)
        self.pca_pipe = pca
        try:
            return pca.transform(self.data)
        except NotFittedError:
            raise

    def clustering(self, n_clusters, n_components=3):
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

    def clusters(self, n_clusters, n_components=3):
        clustering = self.clustering(n_clusters=n_clusters,
                                     n_components=n_components)
        labels = clustering.fit_predict(self.data)
        self.cluster_pipe = clustering
        return labels

    def classifier(self, n_neighbors=3):
        from sklearn.pipeline import Pipeline
        from sklearn.neighbors import KNeighborsClassifier
        return Pipeline([
            ('classifier', KNeighborsClassifier(n_neighbors=3)),
        ])

    def classify(self, labeled_data, n_neighbors=3, transformer=None):
        classifier = self.classifier().fit(*labeled_data)
        self.classifier_pipe = classifier
        if not transformer:
            X1 = self.data
        else:
            X1 = transformer.transform(self.data)
        return classifier.predict(X1)


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
    cleaner = cleanset.Cleaner(fna=threshold,
                               condition=lambda x: x == '-' or x == 'X',
                               axis=0.5)
    return cleaner.fit_transform(frame)


@timeit
def validate_alphabet(df):
    valid_records = []
    null = 0
    alphabet = df.alignment.alphabet
    alphabet_set = set(alphabet)
    values = df.iloc[:, 1:].values
    for index, row in enumerate(values):
        if set(row) <= alphabet_set:
            valid_records.append(index)
        else:
            null += 1
    logger.debug('null records: %r', null)
    # select valid records and reset row indexing
    df = df.iloc[valid_records]
    df.alignment.alphabet = alphabet
    df.reset_index(drop=True, inplace=True)
    return df


@timeit
def frame_from_records(records):
    return pandas.DataFrame.from_records(
        ((rec[0], *list(rec[1])) for rec in records))


@timeit
def dataframe(source, fmt, hmm=True, c=0.9, g=0.1, alphabet=None):
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
    df = frame_from_records(records)

    # set dataframe column labels
    df.columns = ['id'] + list(range(df.shape[1]-1))

    # reduce gappy records/positions
    if g:
        df = filter_gaps(df, g)

    # check alphabet consistency
    if alphabet:
        df.alignment.alphabet = alphabet
    df = validate_alphabet(df)

    return df


@timeit
def save(df, target):
    df.columns = [-1] + list(df.columns)[1:]
    df.to_json(target, compression='bz2')


@timeit
def load(source):
    df = pandas.read_json(source, compression='bz2')
    # sort rows/columns by index
    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, inplace=True)
    df.columns = ['id'] + list(df.columns)[1:]
    return df


@timeit
def save_fasta(df, target):
    with open(target, 'w') as fp:
        for row in df.itertuples(index=False, name=None):
            title = row[0]
            seq = ''.join(row[1:])
            print('>%s\n%s' % (title, seq), file=fp)

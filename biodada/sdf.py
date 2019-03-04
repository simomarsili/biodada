"""SequenceDataFrame class module."""
import logging
import numpy
import pandas
from pandas import DataFrame
from biodada.utils import timeit
from biodada.pipelines import PipelinesMixin

logger = logging.getLogger(__name__)


class SequenceDataFrame(PipelinesMixin, DataFrame):
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

        self.alphabet = kwargs.pop('alphabet', None)

        logger.debug('init SequenceDataFrame, alphabet: %r', self.alphabet)

        super().__init__(*args, **kwargs)

        # set column labels
        if isinstance(self.columns, pandas.RangeIndex):
            lmax = max(len(x) for x in self[0])
            if lmax == 1:
                raise ValueError(
                    'The first data field must contain sequence identifiers')
            else:
                self.columns = ['id'] + list(range(self.shape[1] - 1))

    @property
    def _constructor(self):
        return SequenceDataFrame

    @classmethod
    def from_sequence_records(cls, records, alphabet=None):
        """
        Return a SequenceDataFrame from records iterable.

        If alphabet, filter out records with symbols not in alphabet.
        """
        from biodada.alphabets import check_alphabet, check_alphabet_records
        if alphabet:
            # check alphabet first
            alphabet = check_alphabet(alphabet)
            records = check_alphabet_records(records, alphabet)
        return cls(([identifier] + list(sequence)
                    for identifier, sequence in records),
                   alphabet=alphabet)

    @property
    @timeit
    def data(self):
        """Return an ndarray of one-letter codes."""
        return self.to_numpy(copy=False, dtype='U1')[:, 1:]

    @property
    def records(self):
        """Iterable of frame records."""
        return ((r[0], ''.join(r[1:]))
                for r in self.itertuples(index=False, name=None))

    def encoded(self, encoder='one-hot', dtype=None):
        """
        Return sequence data encoded into integer labels.

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
        """Return n_components principal components from PCA.

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
        n_clusters : int
            The number of clusters.
        n_components : int
            Number of principal components to keep in the dimensionality
            reduction pre-processing step.

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
        dd['index'] = list(self.index)
        dd['columns'] = [-1] + list(self.columns)[1:]
        dd['records'] = list(self.records)
        dd['alphabet'] = self.alphabet
        handle = codecs.getwriter('utf8')(BZ2File(target, 'w'))
        json.dump(dd, fp=handle)


def parse_records(source, frmt, uppercase=True):
    """Parse records from source."""
    import lilbio  # pylint: disable=import-error
    preprocess = lilbio.uppercase_only if uppercase else None
    return lilbio.parse(source, frmt, func=preprocess)


def non_redundant_records(records, threshold=0.9):
    """Return an iterable of non-redundant records."""
    import pcdhit  # pylint: disable=import-error
    return pcdhit.filter(records, threshold)


@timeit
def ungap_frame(frame, threshold=0.1):
    """Return a copy of frame after removing gappy records/positions."""
    import cleanset  # pylint: disable=import-error
    logger.debug('start filtering gaps')
    cleaner = cleanset.Cleaner(
        fna=threshold, condition=lambda x: x == '-' or x == 'X', axis=0.5)
    frame = cleaner.fit_transform(frame)
    logger.debug('stop filtering gaps')
    return frame


@timeit
def read_alignment(source, fmt, uppercase=True, c=0.9, g=0.1, alphabet=None):
    """Parse a pandas dataframe from an alignment file.

    Parameters
    ----------
    source : filepath or file-like
        The alignment file
    fmt : str
        Alignment format. Valid options are: 'fasta', 'stockholm'.
    uppercase : boolean
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
    from biodada.alphabets import check_alphabet, guess_alphabet
    # parse records
    records = parse_records(source, fmt, uppercase=uppercase)

    # filter redundant records via cdhit
    if c:
        records = non_redundant_records(records, c)

    if not alphabet:
        records_head = itertools.islice(records, 50)
        alphabet = guess_alphabet(records_head)
        records = itertools.chain(records_head, records)
    else:
        alphabet = check_alphabet(alphabet)

    # convert records to a dataframe
    df = SequenceDataFrame.from_sequence_records(records, alphabet=alphabet)

    # reduce gappy records/positions
    if g:
        df = ungap_frame(df, g)

    return df


@timeit
def load(source):
    """Load a frame as bzipped json."""
    import json
    import gopen
    with gopen.readable(source) as fp:
        dd = json.load(fp)
    index = dd['index']
    columns = dd['columns']
    columns.sort()
    columns = ['id'] + columns[1:]
    df = SequenceDataFrame(([identifier] + list(sequence)
                            for identifier, sequence in dd['records']),
                           index=index.sort(),
                           columns=columns,
                           alphabet=dd['alphabet'])
    # sort rows/columns by index and reset column labels
    return df

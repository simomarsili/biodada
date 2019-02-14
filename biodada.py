import logging
import pkg_resources
from functools import wraps
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


def parse(source, frmt, hmm=True):
    import lilbio
    preprocess = lilbio.funcs.only_hmm if hmm else None
    return lilbio.parse(source, frmt, func=preprocess)


def filter_redundant(records, threshold=0.9):
    import pcdhit
    return pcdhit.filter(records, threshold)


@timeit
def filter_gaps(frame, threshold=0.1):
    import cleanset
    cleaner = cleanset.Cleaner(f0=threshold, f1=threshold,
                               condition=lambda x: x == '-', axis=0.5)
    return cleaner.fit_transform(frame)


@timeit
def guess_alphabet(df):
    # guess
    from collections import Counter
    counts = Counter(df.iloc[:, 1:].head(50).values.flatten())
    max_score = float('-inf')
    for key, alphabet in ALPHABETS.items():
        score = score_alphabet(alphabet, counts)
        if score > max_score:
            max_score = score
            guess = key
    logger.info('Alphabet guess: %r', guess)
    return ALPHABETS[guess]


def score_alphabet(alphabet, counts):
    import math
    chars = set(alphabet) - set('*-')
    score = (sum([counts.get(a, 0) for a in chars]) /
             math.log(len(alphabet)))
    logger.debug('alphabet %r score %r', alphabet, score)
    return score


@timeit
def validate_alphabet(df):
    valid_records = []
    null = 0
    alphabet = df.alphabet
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
    df.alphabet = alphabet
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
        df.alphabet = alphabet
    else:
        df.alphabet = guess_alphabet(df)
    df = validate_alphabet(df)

    return df


@timeit
def array(source, fmt, hmm=True, c=0.9, g=0.1):
    df = dataframe(source, fmt, hmm=hmm, c=c, g=g)
    return df.iloc[:, 1:].values


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

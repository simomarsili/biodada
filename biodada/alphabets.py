"""Alphabet-related methods."""
import logging
import numpy

ALPHABETS = {
    'protein': '-ACDEFGHIKLMNPQRSTVWY',
    'dna': '-ACGT',
    'rna': '-ACGU',
    'protein_u': '-ACDEFGHIKLMNPQRSTVWYBZX',
    'dna_u': '-ACGTRYMKWSBDHVN',
    'rna_u': '-ACGURYMKWSBDHVN',
}

logger = logging.getLogger(__name__)


def check_alphabet(alphabet):
    # A string of ordered, unique symbols
    return ''.join(sorted(set(alphabet)))


def check_alphabet_records(records, alphabet):
    """Filter out records not consistent with alphabet."""
    alphabet_set = set(alphabet)
    return (r for r in records if set(r[1]) <= alphabet_set)


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
    from biodada import ALPHABETS
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

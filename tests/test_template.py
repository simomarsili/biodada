import os
import pytest
import biodada


ALIGNMENT_FILE = 'PF17907.full.gz'


def tests_dir():
    """Return None is no tests dir."""
    cwd = os.getcwd()
    basename = os.path.basename(cwd)
    if basename == 'tests':
        return cwd
    else:
        tests_dir = os.path.join(cwd, 'tests')
        if os.path.exists(tests_dir):
            return tests_dir


@pytest.fixture()
def alignment():
    return os.path.join(tests_dir(), ALIGNMENT_FILE)


@pytest.fixture()
def records():
    import lilbio
    path = os.path.join(tests_dir(), ALIGNMENT_FILE)
    preprocess = lilbio.uppercase_only
    records = lilbio.parse(path, 'stockholm', func=preprocess)
    return list(records)


@pytest.fixture()
def frame():
    path = os.path.join(tests_dir(), ALIGNMENT_FILE)
    return biodada.read_alignment(path, 'stockholm')


def test_read_alignment(alignment):
    df = biodada.read_alignment(alignment, 'stockholm')
    assert df.shape == (721, 33)


def test_from_records(records):
    alphabet = biodada.ALPHABETS['protein']
    df = biodada.SequenceDataFrame.from_sequence_records(records,
                                                         alphabet=alphabet)
    assert df.shape == (2366, 40)


def test_pca(frame):
    import numpy
    pca_transformer = frame.pca(3).fit(frame.data)
    variance = pca_transformer.named_steps['pca'].explained_variance_
    assert numpy.isclose(variance.sum(), 3.8116467693860563)

import pkg_resources
import logging
from biodada.sdf import SequenceDataFrame, read_alignment, load

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

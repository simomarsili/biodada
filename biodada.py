import logging
import pkg_resources
import pandas

project_name = 'biodada'
__version__ = pkg_resources.require(project_name)[0].version
__copyright__ = 'Copyright (C) 2019 Simone Marsili'
__license__ = 'BSD 3 clause'
__author__ = 'Simone Marsili <simo.marsili@gmail.com>'
__all__ = []


logging.basicConfig(
    # filename=<filename>,
    # filemode='a',
    format='%(module)-10s %(funcName)-20s: %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parse(source, frmt, hmm=True):
    import lilbio
    return lilbio.parse(source, frmt, func=lilbio.funcs.only_hmm)


def filter_redundant(records, thr=0.9):
    import pcdhit
    return pcdhit.filter(records, thr=thr)


def to_dataframe(source, frmt, hmm=True, redundant=0.9, gaps=0.1):
    import cleanset
    records = parse(source, frmt, hmm=hmm)
    records = filter_redundant(records, thr=redundant)
    df = pandas.DataFrame.from_records(
        ((rec[0], *list(rec[1])) for rec in records))
    n, p = df.shape
    df.columns = ['header'] + [str(x) for x in range(p-1)]
    if gaps:
        cleaner = cleanset.Cleaner(f0=gaps, f1=gaps,
                                   condition=lambda x: x == '-', axis=0.5)
        return cleaner.fit_transform(df)
    else:
        return df

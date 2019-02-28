import numpy
from functools import wraps
import logging

logger = logging.getLogger(__name__)


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

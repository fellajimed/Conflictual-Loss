import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from collections.abc import Iterable


def fig_subplots(number_plots, nrows=None, ncols=None,
                 fig=None, figsize=(16, 9), axsize=None):
    """
    A generalization of the function `figures_axes`
    allowing to adjust `ncols` and `nrows`

    return: figure, axes
    """
    if nrows is None and ncols is None:
        return figure_axes(number_plots, fig, figsize)

    if nrows is None:
        ncols = min(number_plots, ncols)

        if number_plots % ncols == 0:
            nrows = number_plots//ncols
            if (axsize is not None and isinstance(axsize, Iterable)
                    and len(axsize) == 2):
                figsize = (ncols*axsize[0], nrows*axsize[1])

            if fig is None:
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                         squeeze=False, figsize=figsize)
            else:
                axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False)
            return fig, axes.flatten()
        nrows = number_plots//ncols + 1
    else:
        nrows = min(number_plots, nrows)
        if number_plots % nrows == 0:
            ncols = number_plots//nrows
            if (axsize is not None and isinstance(axsize, Iterable)
                    and len(axsize) == 2):
                figsize = (ncols*axsize[0], nrows*axsize[1])

            if fig is None:
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                         squeeze=False, figsize=figsize)
            else:
                axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False)
            return fig, axes.flatten()
        ncols = number_plots//nrows + 1

    if (axsize is not None and isinstance(axsize, Iterable)
            and len(axsize) == 2):
        figsize = (ncols*axsize[0], nrows*axsize[1])

    last_row_col = number_plots % ncols
    double_row = 1 + ((last_row_col + ncols) % 2)
    shape = (nrows, ncols*double_row)

    if fig is None:
        fig = plt.figure(figsize=figsize)

    axes = [plt.subplot2grid(shape, (i, j*double_row), rowspan=1,
                             colspan=double_row, fig=fig)
            for (i, j) in product(range(nrows-1), range(ncols))]
    edge = (double_row * (ncols - last_row_col)) // 2
    for j in range(edge, ncols*double_row - edge, double_row):
        axes.append(plt.subplot2grid(shape, (nrows-1, j), rowspan=1,
                                     colspan=double_row, fig=fig))
    return fig, np.array(axes)


def figure_axes(number_plots, fig=None, figsize=(16, 9)):
    """
    intiliaze a figure with the associated axes on 2 rows

    -------------      -------------
    | X | X | X |      | X | X | X |
    -------------  OR  -------------
      | X | X |        | X | X | X |
    -------------      -------------

    return: figure, axes
    """
    if number_plots % 2 == 0:
        if fig is None:
            fig, axes = plt.subplots(nrows=2, ncols=number_plots//2,
                                     figsize=figsize)
        else:
            axes = fig.subplots(nrows=2, ncols=number_plots//2)
        axes = axes.flatten()
    else:
        if fig is None:
            fig = plt.figure(figsize=figsize)

        axes = []
        # row 1
        for i in range((number_plots//2)+1):
            axes.append(plt.subplot2grid((2, number_plots+1), (0, 2*i),
                                         rowspan=1, colspan=2, fig=fig))
        # row 2
        for i in range(number_plots//2):
            axes.append(plt.subplot2grid((2, number_plots+1), (1, 2*i+1),
                                         rowspan=1, colspan=2, fig=fig))

        axes = np.array(axes)

    return fig, axes

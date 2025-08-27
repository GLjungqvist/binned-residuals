from matplotlib import pyplot as plt

import numpy as np

import pandas as pd


def get_binned_averages(x_data, y_data, subset=None, n_bins=None):
    if subset is not None:
        subset = subset & ~(x_data.isnull())
    else:
        subset = ~(x_data.isnull())
    y_data = y_data[subset]
    x_data = x_data[subset]

    n_bins = int(np.floor(np.sqrt(len(y_data))) / 2) if n_bins is None else n_bins
    # find number of bins based on floor division of sqrt number of target observations

    left_boundaries = np.unique(np.percentile(x_data, [x * 100.0 / n_bins for x in range(n_bins)]))
    if (len(left_boundaries) == 1) and (len(np.unique(x_data)) > 1):
        # "Force" an extra bin. What probably happened is that there are loads of values equal to left_boundaries[0]
        if x_data.mean() > left_boundaries[0]:
            left_boundaries = np.append(left_boundaries, x_data.mean())

    list_of_data_frames = []
    for i in range(len(left_boundaries)):
        left_x_bound = left_boundaries[i]
        right_x_bound = left_boundaries[i + 1] if (i + 1) < len(left_boundaries) else np.inf
        x_values_in_bin = x_data[(x_data >= left_x_bound) & (x_data < right_x_bound)]
        y_values_in_bin = y_data[(x_data >= left_x_bound) & (x_data < right_x_bound)]
        x_bar = np.mean(x_values_in_bin)
        y_bar = np.mean(y_values_in_bin)
        n_in_bin = len(x_values_in_bin)
        y_stddev = np.std(y_values_in_bin)
        list_of_data_frames.append(pd.DataFrame(data={
            'x_bar': [x_bar],
            'y_bar': [y_bar],
            'n_in_bin': [n_in_bin],
            'y_se': [2 * y_stddev / np.sqrt(n_in_bin)]
        }))
    return pd.concat(list_of_data_frames)


def plot_binned_averages(residuals, x_values, subset=None, n_bins=None, grid=False, ci=True,
                         ylabel='Binned residuals', title=None, ax=None):
    data_for_plot = get_binned_averages(x_values, residuals, subset, n_bins)

    if ax is None:
        _f, ax = plt.subplots()

    if ci:
        ax.plot(data_for_plot['x_bar'], data_for_plot['y_se'].values, 'darkgray',
                data_for_plot['x_bar'], -data_for_plot['y_se'].values, 'darkgray',
                data_for_plot['x_bar'], data_for_plot['y_bar'], 'o')
    else:
        ax.plot(data_for_plot['x_bar'], data_for_plot['y_bar'], 'o')
    ax.set_xlabel(x_values.name)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.grid(grid)
    return ax


def plot_binned_residuals(residuals, dataframe, colnames, figsize, nrows=None, **kwargs):
    if len(colnames) == 1:
        return plot_binned_averages(residuals, dataframe[colnames[0]], **kwargs)
    if not nrows:
        nrows = int(np.floor(np.sqrt(len(colnames))))
    ncols = int(np.ceil(len(colnames) / nrows))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ax, cname in zip(axs.ravel(), colnames):
        try:
            plot_binned_averages(residuals, dataframe[cname], ax=ax, **kwargs)
        except:
            print('Error when trying to plot column ' + cname)

    return fig, axs

" Utilities for discrete variable creation"
from typing import List

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOURS = ["lightskyblue", "slategrey", "rosybrown"]
COLOURS_DICT = {0: "lightskyblue", 1: "slategrey", 2: "rosybrown"}


# pylint: disable=too-many-arguments
def plot_circles(
    data: np.array,
    labels: np.array,
    title: str,
    xlabel: str,
    ylabel: str,
    frame_on: bool = True,
) -> plt.subplots:
    """
    Plot concentric circles from sklearn.datasets.make_circles.
    """
    _, axis = plt.subplots()
    axis.spines[["right", "top"]].set_visible(False)
    for label in np.unique(labels):
        i = np.where(labels == label)
        axis.scatter(
            data[i, 0],
            data[i, 1],
            c=COLOURS_DICT[label],
            label=label,
            alpha=0.5,
        )
    plt.legend(frameon=frame_on)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    return axis


def plot_scatter(
    data: jnp.array,
    title: str,
    xlabel: str,
    ylabel: str,
    frame_on: bool = True,
) -> plt.subplot:
    """
    Create a matplotlib.pyplot scatterplot.
    """
    _, axis = plt.subplots()
    axis.spines[["right", "top"]].set_visible(False)
    for idx, color, label in zip(range(data.shape[0]), COLOURS, [1, 2, 3]):
        axis.scatter(
            data[idx, :, 0], data[idx, :, 1], alpha=0.5, c=color, label=label
        )
    plt.legend(frameon=frame_on)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return axis


def plot_metrics(
    df_metrics: pd.DataFrame,
    label: str,
    color: List[str],
    type_data: str,
    metrics: List[str],
    axis: plt.subplot,
) -> None:
    """Plot validation and training loss metrics for a given dataset.

    Args:
        df (pandas.DataFrame): The dataframe containing the loss metrics
        to plot.
        label (str): The label to use for the plot legend.
        color (List[str]): A list of two strings representing the colors
        to use for the validation and training loss lines, respectively.
        type_data (str): A string representing the type of data
         used to generate the plot.

    Returns:
        None
    """
    for metric in metrics:
        axis.plot(
            df_metrics[~df_metrics[metric].isnull()]["epoch"],
            df_metrics[~df_metrics[metric].isnull()][metric],
            label=f"{metric} {label}",
            c=color[0],
        )
        x_pos = df_metrics[~df_metrics[metric].isnull()]["epoch"]
        y_pos = df_metrics[~df_metrics[metric].isnull()][metric]
    axis.annotate(
        f"{type_data}",
        xy=(x_pos.iloc[-1], y_pos.iloc[-1]),
        xytext=(1.02 * x_pos.iloc[-1], y_pos.iloc[-1]),
        color=color[1],
    )

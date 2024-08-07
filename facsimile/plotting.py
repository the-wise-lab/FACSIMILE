from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import r2_score


def plot_predictions(
    true: ArrayLike,
    pred: ArrayLike,
    target_names: List[str] = None,
    scale: float = 1.0,
    fname: str = None,
    palette: List[str] = None,
    scatter_kws: Optional[Dict] = None,
    line_kws: Optional[Dict] = None,
    figure_kws: Optional[Dict] = None,
):
    """
    Plot predicted scores against true scores.

    Args:
        true (ArrayLike): True target variable scores.
        pred (ArrayLike): Predicted target variable scores.
        target_names (List[str], optional): List of target variable names.
            Defaults to `None`.
        scale (float, optional): Scale of the plot. Defaults to `1.0`.
        fname (str, optional): Filename to save the plot to. Defaults to
            `None`.
        palette (List[str], optional): List of colours to use for each target.
            Defaults to `None`.
        scatter_kws (Optional[Dict], optional): Keyword arguments to pass to
            the scatter plot. Defaults to `None`.
        line_kws (Optional[Dict], optional): Keyword arguments to pass to the
            regression line. Defaults to `None`.
        figure_kws (Optional[Dict], optional): Keyword arguments to pass to the
            figure. Defaults to `None`.
    """

    # Get number of targets
    n_targets = true.shape[1]

    # Make sure true and pred are numpy arrays
    true = np.array(true)
    pred = np.array(pred)

    # Check target variable names are correct length
    if target_names is not None:
        assert (
            len(target_names) == n_targets
        ), "Number of target variable names must equal number of targets"
    else:
        target_names = ["Variable {}".format(i + 1) for i in range(n_targets)]

    # Set default figure and line keyword arguments if not provided
    if figure_kws is None:
        figure_kws = dict(
            figsize=((n_targets * 3.333) * scale, 3.5 * scale),
            dpi=100,
            facecolor="white",
        )
    if line_kws is None:
        line_kws = {}
    if scatter_kws is None:
        scatter_kws = dict(
            alpha=0.2,
            s=0.5,
        )

    # Set up figure
    f, ax = plt.subplots(
        1,
        n_targets,
        **figure_kws,
    )

    # If a single axis is returned, convert to array
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    # If no palette is provided, use matplotlib default
    if palette is None:
        palette = [
            matplotlib.colors.to_hex(c)
            for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ]

    # If true is provided as a dataframe, convert to array
    if isinstance(true, pd.DataFrame):
        true = true.values

    # Plot each target variable
    for i in range(n_targets):
        ax[i].scatter(x=true[:, i], y=pred[:, i], color=palette[i], **scatter_kws)
        ax[i].set_title(
            target_names[i]
            + "\n$R^2$ = {0}".format(np.round(r2_score(true[:, i], pred[:, i]), 3)),
        )

        # Add regression line
        ax[i].plot(
            np.unique(true[:, i]),
            np.poly1d(np.polyfit(true[:, i], pred[:, i], 1))(np.unique(true[:, i])),
            color=palette[i],
            **line_kws,
        )

        ax[i].set_xlabel("True score")
        ax[i].set_ylabel("Predicted score")

        # Set axis limits based on values
        ax[i].set_xlim([np.min(true[:, i]), np.max(true[:, i])])
        ax[i].set_ylim([np.min(pred[:, i]), np.max(pred[:, i])])

        (diag_line,) = ax[i].plot(
            ax[i].get_xlim(), ax[i].get_ylim(), ls="--", color=palette[i]
        )

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname, dpi=300)

    plt.show()

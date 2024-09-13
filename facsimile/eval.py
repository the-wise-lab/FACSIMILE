from functools import partial
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import cm
from numpy.polynomial.polynomial import Polynomial
from numpy.typing import ArrayLike
from sklearn.metrics import r2_score
from tqdm import tqdm

from .model import FACSIMILE
from .utils import tqdm_joblib


def calculate_score(
    r2: Union[np.ndarray, list], n_included_items: int, n_features: int
) -> float:
    """
    Calculate the score accounting for the number of included items and minimum
    r2.

    Args:
        r2 (Union[np.ndarray, list]): Array or list of r2 values.
        n_included_items (int): Number of included items in the classifier.
        n_features (int): Number of features in the training data.

    Returns:
        float: Calculated score.
    """
    r2_array = np.array(r2)
    score = np.min(r2_array) * (1 - n_included_items / n_features)
    return score


def evaluate_facsimile(
    X_train: Union[pd.DataFrame, ArrayLike],
    y_train: Union[pd.DataFrame, ArrayLike],
    X_val: Union[pd.DataFrame, ArrayLike],
    y_val: Union[pd.DataFrame, ArrayLike],
    alphas: Tuple[float],
    fit_intercept: bool = True,
    additional_metrics: Optional[Dict[str, callable]] = None,
) -> Tuple[float, np.ndarray, int]:
    """
    Evaluate the item reduction model for a given set of alphas.

    The overall score is defined as the minimum R^2 value across the target
    variables, multiplied by 1 minus the number of included items divided by
    the total number of items. This ensures that it selects a model with a good
    fit, but also with a small number of items.

    Args:
        X_train (Union[pd.DataFrame, ArrayLike]): Item responses for
            training.
        y_train (Union[pd.DataFrame, ArrayLike]): Target scores for
            training.
        X_val (Union[pd.DataFrame, ArrayLike]): Item responses for
            validation.
        y_val (Union[pd.DataFrame, ArrayLike]): Target scores for
            validation.
        alphas (Tuple[float]): Alpha values for the targets.
        fit_intercept (bool, optional): Whether to fit an intercept.
            Defaults to `True`.
        additional_metrics (Optional[Dict[str, callable]], optional):
            Dictionary of additional metrics to calculate. These should be
            supplied as functions that take the true and predicted values as
            arguments and return a single value. Defaults to `None`.

    Returns:
        Tuple[float, np.ndarray, int]: Tuple containing the score, R2 and
            number of included items.
    """

    # Set up model
    clf = FACSIMILE(alphas=alphas, fit_intercept=fit_intercept)

    # Dictionary to store metrics
    metrics = {}

    # Fit and predict
    try:
        clf.fit(X_train, y_train)

        pred_val = clf.predict(X_val)

        # Get R2 for each variable
        r2 = r2_score(y_val, pred_val, multioutput="raw_values")

        # Add r2 to metrics
        metrics["r2"] = r2

        # Get other metrics
        if additional_metrics is not None:
            for metric_name, metric_func in additional_metrics.items():
                metric_value = metric_func(y_val, pred_val)
                metrics[metric_name] = metric_value

        # Store number included items
        n_items = clf.n_included_items

        # Get score accounting for n_included_items and minumum r2
        score = calculate_score(r2, clf.n_included_items, X_train.shape[1])

        # Add score to metrics
        metrics["score"] = score

    except Exception as e:
        print("WARNING: Fitting failed. Error:")
        print(e)
        n_items = np.nan
        metrics = {k: np.nan for k in ["score", "r2"] + list(metrics.keys())}

    return metrics, n_items


class FACSIMILEOptimiser:
    def __init__(
        self,
        n_iter: int = 100,
        fit_intercept: bool = True,
        n_jobs: int = 1,
        seed: int = 42,
        alpha_dist_scaling: float = 1,
        additional_metrics: Optional[Dict[str, callable]] = None,
    ) -> None:
        """
        Optimise the alpha values for each target.

        The procedure estimates a "score" for each set of alpha values which
        balances accuracy (R^2) with parsimony (number of items included). The
        score is defined as the minimum R^2 value across the target variables,
        multiplied by 1 minus the number of included items divided by the total
        number of items. This ensures that it selects a model with a good fit,
        but also with a small number of items.

        It also returns R^2 values for each target, the minimum R^2 value, the
        number of included items, and the alpha values for each target.

        Args:
            n_iter (int, optional): Number of iterations to run. Defaults to
                `100`.
            fit_intercept (bool, optional): Whether to fit an intercept.
                Defaults to `True`.
            n_jobs (int, optional): Number of jobs to run in parallel. Defaults
                to `1`.
            seed (int, optional): Random seed. Defaults to `42`.
            alpha_dist_scaling (float, optional): Scaling factor for the
                distribution of alpha (regularisation parameter) values. By
                default, alpha values are sampled from a beta distribution that
                is skewed towards zero. This parameter allows this distribution
                to be scaled, which may be more appropriate for certain
                datasets. Defaults to `1`.
            additional_metrics (Optional[Dict[str, callable]], optional):
                Dictionary of additional metrics to calculate, in addition to
                the penalised score and R^2. These should be supplied as
                functions that take the true and predicted values as arguments
                and return a single value. Defaults to `None`.
        """

        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.n_jobs = n_jobs
        self.seed = seed
        self.alpha_dist_scaling = alpha_dist_scaling

        # Check additional metrics are callable
        if additional_metrics is not None:
            for metric in additional_metrics.values():
                assert callable(metric), "Additional metrics must be callable."

        self.additional_metrics = additional_metrics

    def fit(
        self,
        X_train: Union[pd.DataFrame, ArrayLike],
        y_train: Union[pd.DataFrame, ArrayLike],
        X_val: Union[pd.DataFrame, ArrayLike],
        y_val: Union[pd.DataFrame, ArrayLike],
        target_names: Tuple[str] = None,
    ) -> None:
        """
        Optimise the alpha values for each target.

        The results of the procedure are stored in the `results_` attribute as
        a dataframe. Columns are: Run number, R^2 for each target, minimum R^2,
        score, number of included items, alpha values for each target.

        If other metrics are provided, these are also stored in the dataframe.
        The minimum value for each metric across the Y variables is also
        stored.

        Args:
            X_train (Union[pd.DataFrame, ArrayLike]): Item responses for
                training.
            y_train (Union[pd.DataFrame, ArrayLike]): Target scores for
                training.
            X_val (Union[pd.DataFrame, ArrayLike]): Item responses for
                validation.
            y_val (Union[pd.DataFrame, ArrayLike]): Target scores for
                validation.
            target_names (Tuple[str], optional): Names of the target variables.
                Defaults to `None`.

        """

        # Set up RNG
        rng = np.random.default_rng(self.seed)

        # Get number of targets
        n_targets = y_train.shape[1]

        # Check target names are correct length
        if target_names is not None:
            assert (
                len(target_names) == n_targets
            ), "Number of target names must equal number of targets"
        else:
            target_names = [
                "Variable {}".format(i + 1) for i in range(n_targets)
            ]

        # Set up alphas
        alphas = (
            rng.beta(1, 3, size=(self.n_iter, n_targets))
            * self.alpha_dist_scaling
        )

        # Use partial to set up the function with the data
        evaluate_facsimile_with_data = partial(
            evaluate_facsimile,
            X_train,
            y_train,
            X_val,
            y_val,
            fit_intercept=self.fit_intercept,
            additional_metrics=self.additional_metrics,
        )

        if self.n_jobs == 1:
            results = []
            for i in tqdm(alphas, desc="Evaluation"):
                results.append(evaluate_facsimile_with_data(i))
        else:
            with tqdm_joblib(tqdm(desc="Evaluation", total=self.n_iter)):
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(evaluate_facsimile_with_data)(i) for i in alphas
                )

        # Extract results
        scores = np.array([i[0]["score"] for i in results])
        r2s = np.stack([i[0]["r2"] for i in results])
        n_items = np.stack([i[1] for i in results])

        # Store results in a dataframe
        output_df = {
            "run": range(self.n_iter),
        }

        # Get additional metrics
        if self.additional_metrics is not None:
            for metric_name in self.additional_metrics.keys():
                metrics = np.stack([i[0][metric_name] for i in results])
                for i in range(n_targets):
                    output_df[metric_name + "_" + target_names[i]] = metrics[
                        :, i
                    ]

        # Add R2s for each target
        for i in range(n_targets):
            output_df["r2_" + target_names[i]] = r2s[:, i]

        # Add minimum R2
        output_df["min_r2"] = r2s.min(axis=1)

        # Add maximum R2
        output_df["max_r2"] = r2s.max(axis=1)

        # Add minimum and maximum for other metrics
        if self.additional_metrics is not None:
            for metric_name in self.additional_metrics.keys():
                output_df["min_" + metric_name] = np.min(
                    np.stack([i[0][metric_name] for i in results]), axis=1
                )
                output_df["max_" + metric_name] = np.max(
                    np.stack([i[0][metric_name] for i in results]), axis=1
                )

        # Add scores
        output_df["scores"] = scores

        # Add number of items
        output_df["n_items"] = n_items

        # Add alpha values
        for i in range(n_targets):
            output_df["alpha_" + target_names[i]] = alphas[:, i]

        output_df = pd.DataFrame(output_df)

        self.results_ = output_df

    def get_best_classifier(
        self, metric: str = "scores", highest_best: bool = True
    ) -> FACSIMILE:
        """
        Get the best classifier based on the optimisation results, i.e. the
        classifier with the highest score (balancing R^2 against number of
        included items).

        Args:
            metric (str, optional): Metric to use to select the best
                classifier. Defaults to `'scores'`.
            highest_best (bool, optional): Whether higher values of the metric
                are better. Defaults to `True`.

        Returns:
            FACSIMILE: Best classifier.
        """

        if not hasattr(self, "results_"):
            raise ValueError(
                "Optimisation results not available. Please run fit() first."
            )

        if metric not in self.results_.columns:
            raise ValueError(
                "Metric not available. Please choose from: {}".format(
                    ", ".join(self.results_.columns)
                )
            )

        # Get index of best classifier
        if highest_best:
            best_idx = self.results_[metric].argmax()
        else:
            best_idx = self.results_[metric].argmin()

        # Print out information about this classifier
        print("Best classifier:")
        print(
            r"Minimum R^2: {value}".format(
                value=self.results_.iloc[best_idx]["min_r2"]
            )
        )
        print(
            r"Number of included items: {value}".format(
                value=self.results_.iloc[best_idx]["n_items"]
            )
        )

        # Get alpha values for best classifier
        best_alphas = self.results_.iloc[best_idx][
            [i for i in self.results_.columns if i.startswith("alpha")]
        ].values

        # Set up model
        clf = FACSIMILE(alphas=best_alphas)

        return clf

    def get_best_classifier_max_items(
        self,
        max_items: int = 100,
        metric: str = "scores",
        highest_best: bool = True,
    ) -> FACSIMILE:
        """
        Get the best classifier based on the optimisation results, subject to a
        maximum number of items being included. For example, if `max_items ==
        100`, the best classifier with `100` or fewer items will be returned.

        Args:
            max_items (int, optional): Maximum number of items. Defaults to
                `100`.
            metric (str, optional): Metric to use to select the best
                classifier. Defaults to `'min_r2'`.
            highest_best (bool, optional): Whether higher values of the metric
                are better. Defaults to `True`.

        Returns:
            FACSIMILE: Best classifier.
        """

        if not hasattr(self, "results_"):
            raise ValueError(
                "Optimisation results not available. Please run fit() first."
            )

        if metric not in self.results_.columns:
            raise ValueError(
                "Metric not available. Please choose from: {}".format(
                    ", ".join(self.results_.columns)
                )
            )

        # Get index of best classifier
        results_subset = self.results_[self.results_["n_items"] <= max_items]
        best_idx = results_subset[results_subset["n_items"] <= max_items][
            metric
        ].argmax()

        # Get alpha values for best classifier
        best_alphas = results_subset.iloc[best_idx][
            [i for i in results_subset.columns if i.startswith("alpha")]
        ].values

        # Print out information about this classifier
        print("Best classifier:")
        print(
            r"Minimum R^2: {value}".format(
                value=self.results_.iloc[best_idx]["min_r2"]
            )
        )
        print(
            r"Number of included items: {value}".format(
                value=self.results_.iloc[best_idx]["n_items"]
            )
        )

        # Set up model
        clf = FACSIMILE(alphas=best_alphas)

        return clf

    def get_best_classifier_n_items(
        self,
        n_items: int = 100,
        metric: str = "scores",
        highest_best: bool = True,
    ) -> FACSIMILE:
        """
        Get the best classifier based on the optimisation results with a
        specific number of items. For example, if `n_items = 100`, the best
        classifier exactly `100` items will be returned.

        > **NOTE**: The optimisation procedure is stochastic, so it is possible
        that there may not be a classifier with exactly the number of items
        specified. In this case, an error will be raised.

        Args:
            n_items (int, optional): Number of items. Defaults to `100`.
            metric (str, optional): Metric to use to select the best
                classifier. Defaults to `'min_r2'`.
            highest_best (bool, optional): Whether higher values of the metric
                are better. Defaults to `True`.

        Returns:
            FACSIMILE: Best classifier.
        """

        if not hasattr(self, "results_"):
            raise ValueError(
                "Optimisation results not available. Please run fit() first."
            )

        if metric not in self.results_.columns:
            raise ValueError(
                "Metric not available. Please choose from: {}".format(
                    ", ".join(self.results_.columns)
                )
            )

        if n_items not in self.results_["n_items"].values:
            # Get the closest number of items
            closest_n_items = self.results_["n_items"].values[
                np.argmin(np.abs(self.results_["n_items"].values - n_items))
            ]
            raise ValueError(
                f"No classifier with exactly {n_items} items. Closest "
                f"number of items is {closest_n_items}."
            )

        # Get index of best classifier
        results_subset = self.results_[self.results_["n_items"] == n_items]
        best_idx = results_subset[results_subset["n_items"] == n_items][
            metric
        ].argmax()

        # Get alpha values for best classifier
        best_alphas = results_subset.iloc[best_idx][
            [i for i in results_subset.columns if i.startswith("alpha")]
        ].values

        # Set up model
        clf = FACSIMILE(alphas=best_alphas)

        return clf

    def plot_results(
        self,
        degree: Optional[int] = 3,
        figsize: Tuple[int, int] = (10, 6),
        cmap: Optional[str] = None,
        scatter_kws: Optional[Dict] = None,
        line_kws: Optional[Dict] = None,
        figure_kws: Optional[Dict] = None,
    ) -> None:
        """
        Plots the results of the optimization procedure, showing the R^2 values
        for each target variable as a function of the number of items included.

        Args:
            degree (Optional[int], optional): The degree of the polynomial for
                regression fitting. If `None`, no line is fitted or plotted.
                Defaults to `3`.
            figsize (Tuple[int, int], optional): The size of the figure
                to be plotted. Defaults to `(10,6)`.
            cmap (Optional[str], optional): The name of a colormap to generate
                colors for scatter points and lines. If `None`, uses the
                Matplotlib default color cycle. Defaults to `None`.
            scatter_kws (Optional[Dict], optional): Additional keyword
                arguments for `plt.scatter`. Defaults to `None`.
            line_kws (Optional[Dict], optional): Additional keyword arguments
                for `plt.plot`. Defaults to `None`.
            figure_kws (Optional[Dict], optional): Additional keyword arguments
                for `plt.figure`. Defaults to `None`.

        Returns:
            None: Displays the plot.
        """
        df = self.results_
        scatter_kws = {"alpha": 0.6} if scatter_kws is None else scatter_kws
        line_kws = {} if line_kws is None else line_kws
        figure_kws = {} if figure_kws is None else figure_kws

        # Creating a figure
        plt.figure(figsize=figsize, **figure_kws)

        # Inferring Y variables from DataFrame columns
        y_vars = [col for col in df.columns if col.startswith("r2_")]

        # Getting the colormap if provided
        if cmap:
            colors = cm.get_cmap(cmap, len(y_vars))

        for i, y_var in enumerate(y_vars):
            color = colors(i) if cmap else None
            # Scatter plot for each Y variable
            plt.scatter(
                df["n_items"],
                df[y_var],
                label=f'{y_var.split("r2_")[1]}',
                color=color,
                **scatter_kws,
            )

            if degree is not None:
                # Fit the model
                p = Polynomial.fit(df["n_items"], df[y_var], degree)

                # Plot the regression line for each Y variable
                x = np.linspace(df["n_items"].min(), df["n_items"].max(), 400)
                y = p(x)
                plt.plot(x, y, linewidth=2, color=color, **line_kws)

        # Labeling the plot
        plt.xlabel("Number of items")
        plt.ylabel(r"$R^2$")
        legend = plt.legend()

        # Update alpha for legend handles
        for lh in legend.legend_handles:
            lh.set_alpha(1)  # Set alpha to 1

        plt.tight_layout()
        plt.show()

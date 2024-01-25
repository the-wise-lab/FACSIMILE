from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from typing import Tuple, Union
from .model import FACSIMILE
from .utils import tqdm_joblib
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from matplotlib import cm


def evaluate_facsimile(
    X_train: Union[pd.DataFrame, ArrayLike],
    y_train: Union[pd.DataFrame, ArrayLike],
    X_val: Union[pd.DataFrame, ArrayLike],
    y_val: Union[pd.DataFrame, ArrayLike],
    alphas: Tuple[float],
    fit_intercept: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the item reduction model for a given set of alphas.

    The overall score is defined as the minimum R2 value across the factors, multiplied by 1 minus the number of
    included items divided by the total number of items. This ensures that it selects a model with a good fit, but
    also with a small number of items.

    Args:
        X_train (ArrayLike): Item responses for training.
        y_train (ArrayLike): Factor scores for training.
        X_val (ArrayLike): Item responses for validation.
        y_val (ArrayLike): Factor scores for validation.
        alphas (Tuple[float]): Alpha values for the 3 factors.
        fit_intercept (bool, optional): Whether to fit an intercept. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the score, R2 and number of included items.
    """

    # Set up model
    clf = FACSIMILE(alphas=alphas, fit_intercept=fit_intercept)

    # Fit and predict
    try:
        clf.fit(X_train, y_train)
        pred_val = clf.predict(X_val)

        # Get R2 for each variable
        r2 = r2_score(y_val, pred_val, multioutput="raw_values")

        # Store number included items
        n_items = clf.n_included_items

        # Get score accounting for n_included_items and minumum r2
        score = np.min(r2) * (1 - clf.n_included_items / X_train.shape[1])
    except Exception as e:
        print("WARNING: Fitting failed. Error:")
        print(e)
        score = n_items = np.nan
        r2 = np.ones(y_val.shape[1]) * np.nan

    return score, r2, n_items


class FACSIMILEOptimiser:
    def __init__(
        self,
        n_iter: int = 100,
        fit_intercept: bool = True,
        n_jobs: int = 1,
        seed: int = 42,
    ) -> None:
        """
        Optimise the alpha values for each factor.

        The procedure estimates a "score" for each set of alpha values which balances
        accuracy (R^2) with parsimony (number of items included). The score is defined
        as the minimum R^2 value across the 3 factors, multiplied by 1 minus the number of
        included items divided by the total number of items. This ensures that it selects
        a model with a good fit, but also with a small number of items.

        It also returns R^2 values for each factor, the minimum R^2 value, the number of
        included items, and the alpha values for each factor.

        Args:
            n_iter (int, optional): Number of iterations to run. Defaults to 100.
            fit_intercept (bool, optional): Whether to fit an intercept. Defaults to True.
            n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1.
            seed (int, optional): Random seed. Defaults to 42.
        """

        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.n_jobs = n_jobs
        self.seed = seed

    def fit(
        self,
        X_train: Union[pd.DataFrame, ArrayLike],
        y_train: Union[pd.DataFrame, ArrayLike],
        X_val: Union[pd.DataFrame, ArrayLike],
        y_val: Union[pd.DataFrame, ArrayLike],
        factor_names: Tuple[str] = None,
    ) -> None:
        """
        Optimise the alpha values for each factor.

        The results of the procedure are stored in the `results_` attribute as a
        dataframe. Columns are: Run number, R^2 for each factor, minimum R^2, score,
        number of included items, alpha values for each factor.

        Args:
            X_train (Union[pd.DataFrame, ArrayLike]): Item responses for training.
            y_train (Union[pd.DataFrame, ArrayLike]): Factor scores for training.
            X_val (Union[pd.DataFrame, ArrayLike]): Item responses for validation.
            y_val (Union[pd.DataFrame, ArrayLike]): Factor scores for validation.
            factor_names (Tuple[str], optional): Names of the factors. Defaults to None.

        """

        # Set up RNG
        rng = np.random.default_rng(self.seed)

        # Get number of factors
        n_factors = y_train.shape[1]

        # Check factor names are correct length
        if factor_names is not None:
            assert (
                len(factor_names) == n_factors
            ), "Number of factor names must equal number of factors"
        else:
            factor_names = ["Factor {}".format(i + 1) for i in range(n_factors)]

        # Set up alphas
        alphas = rng.beta(1, 3, size=(self.n_iter, n_factors))

        # Use partial to set up the function with the data
        evaluate_facsimile_with_data = partial(
            evaluate_facsimile,
            X_train,
            y_train,
            X_val,
            y_val,
            fit_intercept=self.fit_intercept,
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
        scores = np.stack([i[0] for i in results])
        r2s = np.stack([i[1] for i in results])
        n_items = np.stack([i[2] for i in results])

        output_df = {
            "run": range(self.n_iter),
        }

        # Add R2s for each factor
        for i in range(n_factors):
            output_df["r2_" + factor_names[i]] = r2s[:, i]

        # Add minimum R2
        output_df["min_r2"] = r2s.min(axis=1)

        # Add scores
        output_df["scores"] = scores

        # Add number of items
        output_df["n_items"] = n_items

        # Add alpha values
        for i in range(n_factors):
            output_df["alpha_" + factor_names[i]] = alphas[:, i]

        output_df = pd.DataFrame(output_df)

        self.results_ = output_df

    def get_best_classifier(self, metric: str = "scores") -> FACSIMILE:
        """
        Get the best classifier based on the optimisation results, i.e. the classifier
        with the highest score (balancing R^2 against number of included items).

        Args:
            metric (str, optional): Metric to use to select the best classifier. Defaults to 'scores'.

        Returns:
            FACSIMILE: Best classifier.
        """

        if not hasattr(self, "results_"):
            raise ValueError(
                "Optimisation results not available. Please run fit() first."
            )

        if not metric in self.results_.columns:
            raise ValueError(
                "Metric not available. Please choose from: {}".format(
                    ", ".join(self.results_.columns)
                )
            )

        # Get index of best classifier
        best_idx = self.results_[metric].argmax()

        # Get alpha values for best classifier
        best_alphas = self.results_.iloc[best_idx][
            [i for i in self.results_.columns if i.startswith("alpha")]
        ].values

        # Set up model
        clf = FACSIMILE(alphas=best_alphas)

        return clf

    def get_best_classifier_max_items(
        self, max_items: int = 100, metric: str = "min_r2"
    ) -> FACSIMILE:
        """
        Get the best classifier based on the optimisation results, subject to a maximum
        number of items being included. For example, if max_items = 100, the best classifier
        with 100 or fewer items will be returned.

        Args:
            max_items (int, optional): Maximum number of items. Defaults to 100.
            metric (str, optional): Metric to use to select the best classifier. Defaults to 'min_r2'.

        Returns:
            FACSIMILE: Best classifier.
        """

        if not hasattr(self, "results_"):
            raise ValueError(
                "Optimisation results not available. Please run fit() first."
            )

        if not metric in self.results_.columns:
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
        Plots the results of the optimization procedure, showing the R2 values for each
        factor as a function of the number of items included.

        Args:
            degree (Optional[int], optional): The degree of the polynomial for
                regression fitting. If None, no line is fitted or plotted.
                Defaults to 3.
            figsize (Tuple[int, int], optional): The size of the figure
                to be plotted. Defaults to (10,6).
            cmap (Optional[str], optional): The name of a colormap to generate colors for
                scatter points and lines. If None, uses the Matplotlib default color cycle.
                Defaults to None.
            scatter_kws (Optional[Dict], optional): Additional keyword arguments for plt.scatter.
                Defaults to None.
            line_kws (Optional[Dict], optional): Additional keyword arguments for plt.plot.
                Defaults to None.
            figure_kws (Optional[Dict], optional): Additional keyword arguments for plt.figure.
                Defaults to None.

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
        for lh in legend.legendHandles:
            lh.set_alpha(1)  # Set alpha to 1

        plt.tight_layout()
        plt.show()

from typing import List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso, LinearRegression
from importlib.metadata import version


class FACSIMILE(BaseEstimator):
    def __init__(self, alphas: Tuple[float], fit_intercept: bool = True) -> None:
        """
        FACtor Score IteM reductIon with Lasso Estimator (FACSIMILE) class.

        Predicts target scores from item responses using Lasso regression,
        producing a sparse model with a small number of items included.

        Args:
            alphas (Tuple[float]): Tuple of alpha values for each target
                variable.
            fit_intercept (bool, optional): Whether to fit an intercept.
                Defaults to `True`.
        """

        # Get number of target variables
        self.n_targets = len(alphas)
        # Store the alphas
        self.alphas = alphas
        # Store the included items
        self.included_items = None
        # Store whether to fit an intercept
        self.fit_intercept = fit_intercept

        # Store classifiers
        self.clf = []

        super().__init__()

    def fit(
        self,
        X: Union[pd.DataFrame, ArrayLike],
        y: Union[pd.DataFrame, ArrayLike],
    ) -> BaseEstimator:
        """
        Fit the model to the data.

        Args:
            X (Union[pd.DataFrame, ArrayLike]): Item responses. Can be provided
                as a dataframe or array of shape `(n_observations, n_items)`
            y (Union[pd.DataFrame, ArrayLike]): Target scores. Can be provided
                as a dataframe or array of shape `(n_observations, n_targets)`

        Returns:
            FACSIMILE: The fitted model
        """

        # Clear existing classifiers and included items
        self.clf = []
        self.included_items = None

        # Store column names, if provided
        if isinstance(X, pd.DataFrame):
            self.item_names = X.columns
        else:
            self.item_names = None

        # Same for y to get target variable names
        if isinstance(y, pd.DataFrame):
            self.target_names = y.columns
        else:
            # Create default target variable names
            self.target_names = ["Variable {}".format(i + 1) for i in range(y.shape[1])]

        # Check that the number of targets matches the number of alphas
        if y.shape[1] != self.n_targets:
            raise ValueError(
                "Number of targets must match number of alpha values. "
                "Got y data with {} targets but {} alpha values were "
                "specified.".format(y.shape[1], self.n_targets)
            )

        # Extract values from X and y if they are dataframes
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        # Check that X and y have the same number of rows
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        # Run the classifiers for each target to determine which items will be
        # kept This is slightly awkward because we end up running the
        # classifiers twice, once to determine which items to keep, and once to
        # fit the final model.
        clf_included_items = np.zeros((self.n_targets, X.shape[1]))

        # Loop over the target variables
        for var in range(self.n_targets):
            # Set up lasso regression with given alpha for this target
            clf = Lasso(alpha=self.alphas[var], fit_intercept=self.fit_intercept)
            # Fit the model
            clf.fit(X, y[:, var])
            # Store the included items
            clf_included_items[var, :] = clf.coef_ != 0

        # Get items included in at least one target
        self.included_items = (clf_included_items.sum(axis=0)) > 0

        # Raise an error if no items are included
        if not self.included_items.any():
            raise ValueError(
                "No items were included for any target variable. "
                "Try reducing the alpha values."
            )

        # Store column names, if provided
        if self.item_names is not None:
            self.included_item_names = self.item_names[self.included_items]

        # Store number of included items
        self.n_included_items = self.included_items.sum()

        # Fit a new, unregularised model using all the items included in at
        # least one target variable
        for var in range(self.n_targets):
            # Set up linear regression
            # We don't need regularisation here because we have already
            # selected the items
            clf = LinearRegression()

            # Fit the model
            clf.fit(X[:, self.included_items], y[:, var])

            # Store the fitted model
            self.clf.append(clf)

        return self

    def predict(self, X: Union[pd.DataFrame, ArrayLike]) -> pd.DataFrame:
        """
        Predict the target scores for a given set of item responses.

        Args:
            X (Union[pd.DataFrame, ArrayLike]): Item responses.

        Returns:
            pd.DataFrame: Predicted target variable scores.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return self.predict_reduced(X.iloc[:, self.included_items])

    def predict_reduced(self, X: Union[pd.DataFrame, ArrayLike]) -> pd.DataFrame:
        """
        Predict the target scores for a given set of item responses, using only
        the items identified by the item reduction procedure.

        Args:
            X (Union[pd.DataFrame, ArrayLike]): Responses for the items
                included in the item reduction procedure.

        Returns:
            pd.DataFrame: Predicted target variable scores.

        """

        # Extract values from X if it is a dataframe
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Array to store predictions
        pred = np.zeros((X.shape[0], self.n_targets))

        # Predict target scores
        for var in range(self.n_targets):
            # Predict target scores
            y_pred = self.clf[var].predict(X)

            # Store predictions
            pred[:, var] = y_pred

        # Convert to dataframe with target variable names
        pred = pd.DataFrame(pred, columns=self.target_names)

        return pred

    def get_weights(self, target_names: List[str] = None) -> pd.DataFrame:
        """
        Return the classifier weights as a pandas dataframe.

        Args:
            target_names (List[str], optional): List of target variable names.
            Defaults
                to `None`.

        Returns:
            pd.DataFrame: A dataframe containing the classifier weights.
                Each row corresponds to an item, and each column to a target
                variable.
        """

        # If no target variable names are provided, use default
        if target_names is None:
            target_names = self.target_names

        # Get the weights from each classifier
        weights = [clf.coef_ for clf in self.clf]

        # Use item names as column names if available, otherwise use integers
        if self.item_names is not None:
            column_names = self.included_item_names
        else:
            column_names = np.arange(weights[0].shape[0])

        # Convert to dataframe
        weights = pd.DataFrame(weights, columns=column_names)

        # Transpose so that each row is a target variable
        weights = weights.T

        # Add target variable names as columns
        weights.columns = target_names

        # Add intercept row
        weights.loc["Intercept"] = [clf.intercept_ for clf in self.clf]

        return weights

    def save(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path (str): Path to save the model to.

        """
        # Embed information about facsimile version
        self.__facsimile_version = version("facsimile")

        # Save the model to a file using joblib
        joblib.dump(self, path)

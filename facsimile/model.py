from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import r2_score
import joblib
from typing import Tuple, Union, List
from numpy.typing import ArrayLike
import pandas as pd
import numpy as np


class FACSIMILE(BaseEstimator):
    def __init__(self, alphas: Tuple[float], bias_correction: bool = True) -> None:
        """
        FACtor Score IteM reductIon with Lasso Estimator (FACSIMILE) class.

        Predicts factor scores from item responses using Lasso regression,
        producing a sparse model with a small number of items included.

        Args:
            alphas (Tuple[float]): Tuple of alpha values for each factor.
        """

        # Get number of factors
        self.n_factors = len(alphas)
        # Store the alphas
        self.alphas = alphas
        # Store the included items
        self.included_items = None

        # Store classifiers
        self.clf = []

        super().__init__()

    def fit(self, X: Union[pd.DataFrame, ArrayLike], y: Union[pd.DataFrame, ArrayLike]):
        """
        Fit the model to the data.

        Args:
            X (Union[pd.DataFrame, ArrayLike]): Item responses. Can be provided as a dataframe or array of shape
            (n_observations, n_items)
            y (Union[pd.DataFrame, ArrayLike]): Factor scores. Can be provided as a dataframe or array of shape
            (n_observations, n_factors)
        """

        # Clear existing classifiers and included items
        self.clf = []
        self.included_items = None

        # Store column names, if provided
        if isinstance(X, pd.DataFrame):
            self.item_names = X.columns
        else:
            self.item_names = None

        # Same for y to get factor names
        if isinstance(y, pd.DataFrame):
            self.factor_names = y.columns
        else:
            # Create default factor names
            self.factor_names = ["Factor {}".format(i + 1) for i in range(y.shape[1])]

        # Extract values from X and y if they are dataframes
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        # Check that X and y have the same number of rows
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        # Run the classifiers for each factor to determine which items will be kept
        # This is slightly awkward because we end up running the classifiers twice,
        # once to determine which items to keep, and once to fit the final model.
        clf_included_items = np.zeros((3, X.shape[1]))

        # Loop over the 3 factors
        for var in range(3):
            # Set up lasso regression with given alpha for this factor
            clf = Lasso(alpha=self.alphas[var])
            # Fit the model
            clf.fit(X, y[:, var])
            # Store the included items
            clf_included_items[var, :] = clf.coef_ != 0

        # Get items included in at least one factor
        self.included_items = (clf_included_items.sum(axis=0)) > 0

        # Store column names, if provided
        if self.item_names is not None:
            self.included_item_names = self.item_names[self.included_items]

        # Store number of included items
        self.n_included_items = self.included_items.sum()

        # Fit a new, unregularised model using all the items included in at least one factor
        for var in range(3):
            # Set up linear regression
            # We don't need regularisation here because we have already selected the items
            clf = LinearRegression()

            # Fit the model
            clf.fit(X[:, self.included_items], y[:, var])

            # Store the fitted model
            self.clf.append(clf)

        return self

    def predict(self, X: Union[pd.DataFrame, ArrayLike]) -> pd.DataFrame:
        """
        Predict the factor scores for a given set of item responses.

        Args:
            X (Union[pd.DataFrame, ArrayLike]): Item responses.

        Returns:
            pd.DataFrame: Predicted factor scores.
        """

        return self.predict_reduced(X.iloc[:, self.included_items])

    def predict_reduced(self, X: Union[pd.DataFrame, ArrayLike]) -> pd.DataFrame:
        """
        Predict the factor scores for a given set of item responses,
        using only the items identified by the item reduction procedure.

        Args:
            X (Union[pd.DataFrame, ArrayLike]): Responses for the items included
            in the item reduction procedure.

        Returns:
            pd.DataFrame: Predicted factor scores.

        """

        # Extract values from X if it is a dataframe
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Array to store predictions
        pred = np.zeros((X.shape[0], 3))

        # Predict factor scores
        for var in range(3):
            # Predict factor scores
            y_pred = self.clf[var].predict(X)

            # Store predictions
            pred[:, var] = y_pred

        # Convert to dataframe with factor names
        pred = pd.DataFrame(pred, columns=self.factor_names)

        return pred

    def get_weights(self, factor_names: List[str] = None) -> pd.DataFrame:
        """
        Return the classifier weights as a pandas dataframe.

        Args:
            factor_names (List[str], optional): List of factor names. Defaults to None.

        Returns:
            pd.DataFrame: A dataframe containing the classifier weights.
            Each row corresponds to an item, and each column to a factor.
        """

        # If no factor names are provided, use default
        if factor_names is None:
            factor_names = self.factor_names

        # Get the weights from each classifier
        weights = [clf.coef_ for clf in self.clf]

        # Use item names as column names if available, otherwise use integers
        if self.item_names is not None:
            column_names = self.included_item_names
        else:
            column_names = np.arange(weights[0].shape[0])

        # Convert to dataframe
        weights = pd.DataFrame(weights, columns=column_names)

        # Transpose so that each row is a factor
        weights = weights.T

        # Add factor names as columns
        weights.columns = factor_names

        # Add intercept row
        weights.loc["Intercept"] = [clf.intercept_ for clf in self.clf]
        
        return weights

    def save(self, path: str):
        """
        Save the model to a file.

        Args:
            path (str): Path to save the model to.
        """

        # Save the model to a file using joblib
        joblib.dump(self, path)

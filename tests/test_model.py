from sklearn.datasets import make_regression
from sklearn.preprocessing import scale
from facsimile.model import FACSIMILE
import numpy as np
import pandas as pd


def test_included_items():
    """Make sure that we only include predictive items"""

    x, y = make_regression(n_samples=400, n_features=10, n_targets=2, random_state=42)

    # Scale the data
    x = scale(x)
    y = scale(y)

    # Set some items to be uninformative by shuffling their values
    uninformative_idx = [0, 1, 2, 3]

    rng = np.random.RandomState(42)
    x[:, uninformative_idx] = rng.permutation(x[:, uninformative_idx])

    # Create a model
    model = FACSIMILE(alphas=[0.3, 0.3])

    # Fit the model
    model.fit(x, y)

    # Check that the uninformative items are not included
    assert not np.any(model.included_items[uninformative_idx])


def test_alpha_values():
    """Make sure that alpha values have the intended effect"""
    x, y = make_regression(n_samples=400, n_features=10, n_targets=2, random_state=42)

    # Scale the data
    x = scale(x)
    y = scale(y)

    prev_sum = 20

    for alpha in [0.001, 0.3, 0.5]:
        # Create a model
        model = FACSIMILE(alphas=[alpha, alpha])

        # Fit the model
        model.fit(x, y)

        assert model.included_items.sum() < prev_sum

        prev_sum = model.included_items.sum()


def test_included_item_names():
    """Make sure that alpha values have the intended effect"""
    x, y = make_regression(n_samples=400, n_features=10, n_targets=2, random_state=42)

    # Scale the data
    x = scale(x)
    y = scale(y)

    # Turn x into a dataframe with column names
    x = pd.DataFrame(x, columns=[f"Item {i}" for i in range(x.shape[1])])

    # Create a model
    model = FACSIMILE(alphas=[0.5, 0.5])

    # Fit the model
    model.fit(x, y)

    # Check that the item names are correct
    assert all(model.included_item_names == x.columns[model.included_items].tolist())

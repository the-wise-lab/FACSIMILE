import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from facsimile.utils import train_validation_test_split


def test_train_validation_test_split():
    # Generate some example data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

    # Generate 10 random split proportions
    rng = np.random.default_rng(42)
    split_proportions = rng.random((10, 3))
    split_proportions /= np.sum(split_proportions, axis=1, keepdims=True)

    # Test each split proportion
    for i in range(10):
        train_size, val_size, test_size = split_proportions[i]
        train_size += 1 - (train_size + val_size + test_size)

        X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(
            X, y, train_size=train_size, val_size=val_size, test_size=test_size, random_seed=42
        )

        # Check that the split proportions are correct
        assert_almost_equal(X_train.shape[0] / X.shape[0], train_size, decimal=2)
        assert_almost_equal(X_val.shape[0] / X.shape[0], val_size, decimal=2)
        assert_almost_equal(X_test.shape[0] / X.shape[0], test_size, decimal=2)
        assert_almost_equal(train_size + val_size + test_size, 1.0, decimal=6)

        # Check that the X and y data are consistent
        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
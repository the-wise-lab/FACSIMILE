import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_classification
from facsimile.utils import train_validation_test_split, simple_predict
import pytest


@pytest.mark.parametrize("random_seed", [42])
def test_train_validation_test_split(random_seed):
    # Generate some example data
    X, y = make_classification(
        n_samples=1000, n_features=10, random_state=random_seed
    )

    # Generate 10 random split proportions
    rng = np.random.default_rng(random_seed)
    split_proportions = rng.random((10, 3))
    split_proportions /= np.sum(split_proportions, axis=1, keepdims=True)

    # Test each split proportion
    for i in range(10):
        train_size, val_size, test_size = split_proportions[i]
        train_size += 1 - (train_size + val_size + test_size)

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        ) = train_validation_test_split(
            X,
            y,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            random_seed=random_seed,
        )

        # Check that the split proportions are correct
        assert_almost_equal(
            X_train.shape[0] / X.shape[0], train_size, decimal=2
        )
        assert_almost_equal(
            X_val.shape[0] / X.shape[0], val_size, decimal=2
        )
        assert_almost_equal(
            X_test.shape[0] / X.shape[0], test_size, decimal=2
        )
        assert_almost_equal(
            train_size + val_size + test_size, 1.0, decimal=6
        )

        # Check that the X and y data are consistent
        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

        # Check that the order has been randomised
        assert not np.all(X_train == X[: X_train.shape[0]])
        assert not np.all(y_train == y[: y_train.shape[0]])
        assert not np.all(X_val == X[: X_val.shape[0]])


def test_simple_predict():
    # Create example weights DataFrame
    weights_data = {
        "weights": [0.5, 1.5, 2.0, 1.0]
    }  # Last value is the intercept
    weights = pd.DataFrame(weights_data)

    # Create example input data
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Expected output
    expected_output = np.array(
        [
            [1.0 * 0.5 + 2.0 * 1.5 + 3.0 * 2.0 + 1.0],
            [4.0 * 0.5 + 5.0 * 1.5 + 6.0 * 2.0 + 1.0],
            [7.0 * 0.5 + 8.0 * 1.5 + 9.0 * 2.0 + 1.0],
        ]
    )

    # Call the function
    output = simple_predict(weights, X)

    # Assert the output is as expected
    np.testing.assert_almost_equal(output, expected_output)


if __name__ == "__main__":
    pytest.main()

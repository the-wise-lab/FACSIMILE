import joblib
import contextlib
from tqdm import tqdm
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split


# Taken from https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument

    From https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def train_validation_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float,
    val_size: float,
    test_size: float,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits X and y data into train/validation/test sets according to given split proportions, using a random seed.

    Args:
        X (numpy.ndarray): The input data.
        y (numpy.ndarray): The target data.
        train_size (float): The proportion of the data to use for training.
        val_size (float): The proportion of the data to use for validation.
        test_size (float): The proportion of the data to use for testing.
        random_seed (int, optional): The random seed to use. Defaults to 42.

    Returns:
        Tuple[numpy.ndarray]: A tuple containing the train, validation, and test sets.
    """
    assert (
        train_size + val_size + test_size == 1.0
    ), "Train, validation, and test sizes must add up to 1.0"
    assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size / (train_size + val_size),
        random_state=random_seed,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

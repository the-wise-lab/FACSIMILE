import contextlib
import os
import re
from importlib.metadata import version
from tempfile import NamedTemporaryFile
from typing import Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from fontTools import ttLib
from sklearn.model_selection import train_test_split
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """
    Context manager to patch joblib to report into tqdm progress bar given as
    argument

    From
    https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
    """

    class TqdmBatchCompletionCallback(
        joblib.parallel.BatchCompletionCallBack
    ):
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
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Splits X and y data into train/validation/test sets according to given
    split proportions, using a random seed.

    Args:
        X (numpy.ndarray): The input data.
        y (numpy.ndarray): The target data.
        train_size (float): The proportion of the data to use for training.
        val_size (float): The proportion of the data to use for validation.
        test_size (float): The proportion of the data to use for testing.
        random_seed (int, optional): The random seed to use. Defaults to `42`.

    Returns:
        Tuple[numpy.ndarray]: A tuple containing the train, validation, and
        test sets.
    """
    assert (
        train_size + val_size + test_size == 1.0
    ), "Train, validation, and test sizes must add up to 1.0"
    assert (
        X.shape[0] == y.shape[0]
    ), "X and y must have the same number of samples"

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


def load_model(model_path: str) -> object:
    """
    Load a model from disk.

    Args:
        model_path (str): Path to the model file.

    Returns:
        object: The loaded model.
    """

    # Check if the model file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load the model using joblib
    model = joblib.load(model_path)

    # Warn if the model was saved with a different version of facsimile
    if hasattr(model, "__facsimile_version"):
        if model.__facsimile_version != version("facsimile"):
            print(
                f"Warning: Model was saved with facsimile version "
                f"{model.__facsimile_version}, but the current version is "
                f"{version('facsimile')}."
            )

    return model


def simple_predict(weights: pd.DataFrame, X: np.ndarray) -> np.ndarray:
    """
    Predict target scores using a simple linear model.

    Args:
        weights (pd.DataFrame): A dataframe containing the weights for each
            item.
        X (np.ndarray): The input data.

    Returns:
        np.ndarray: The predicted target variable scores.
    """
    return X @ weights.values[:-1] + weights.values[-1]


def check_directories():
    """
    Checks if the script is being run in the root directory and if the
    required data is present.

    Raises:
        RuntimeError: If the script is not run from the root directory
            or if the `'data'` directory is empty.
    """
    # Check if the 'notebooks' directory exists
    if not os.path.isdir("docs"):
        # If we're currently in a subdirectory of the "notebooks", move
        # two directories up
        if os.path.isdir("../examples"):
            print("Changing directory to root directory of repository...")
            os.chdir("../..")
        else:
            raise RuntimeError(
                "You must run this notebook from the root directory of the "
                "repository, otherwise paths will break. You are currently "
                "in {}".format(os.getcwd())
            )

    # Check if the 'data' directory exists and is not empty
    if not os.path.isdir("data") or len(os.listdir("data")) == 0:
        raise RuntimeError(
            "You must download the data files from OSF and place them in the "
            "/data directory before running this notebook."
        )

    # Check if the 'figures' directory exists and create it if not
    if not os.path.isdir("figures"):
        os.mkdir("figures")


# Styling functions for notebooks
def download_googlefont(font: str = "Heebo") -> None:
    """
    Download a font from Google Fonts and save it in the `fonts` folder.

    This code is modified from `Opinionated`
    (https://github.com/MNoichl/opinionated), which itself is borrowed from
    https://github.com/TutteInstitute/datamapplot.

    Args:
        font (str, optional): The name of the font to download from Google
            Fonts. Defaults to `"Heebo"`.
    """

    # Replace spaces with '+' to format the font name for the API URL
    api_fontname = font.replace(" ", "+")
    # Retrieve the CSS from Google Fonts API that contains the URLs for font
    # files
    api_response = requests.get(
        f"https://fonts.googleapis.com/css?family={api_fontname}:black,"
        "bold,regular,light"
    )
    # Extract font file URLs from the response content
    font_urls = re.findall(r"https?://[^\)]+", str(api_response.content))

    # Download and process each font file found
    for font_url in font_urls:
        # Download the font file
        font_data = requests.get(font_url)
        # Create a temporary file to save the downloaded font
        with NamedTemporaryFile(delete=False, suffix=".ttf") as f:
            f.write(font_data.content)
            # Ensure the file is written and closed properly
            f.close()

            # Load the font using fontTools library
            font = ttLib.TTFont(f.name)
            # Retrieve the font family name from the font's metadata
            font_family_name = font["name"].getDebugName(1)
            # Add the font to matplotlib's font manager for future use
            matplotlib.font_manager.fontManager.addfont(f.name)
            print(f"Added new font as {font_family_name}")


def set_style(
    style_path: str = "../style.mplstyle", font: str = "Heebo"
) -> None:
    """
    Set the Matplotlib style and download the specified font from Google
    Fonts.

    Args:
        style_path (str, optional): The path to the Matplotlib style file.
            Defaults to `../style.mplstyle`.
        font (str, optional): The name of the font to download from Google
            Fonts. Defaults to `"Heebo"`.
    """

    # Check whether matplotlib already has the font
    font_names = [
        f.name for f in matplotlib.font_manager.fontManager.ttflist
    ]
    if font in font_names:
        print(f"Font {font} already available in Matplotlib.")
    else:
        download_googlefont(font)

    # Read the original style file and replace the font.family line with the
    # new font
    with open(style_path, "r") as f:
        style_lines = f.readlines()

    new_style_lines = [
        (
            line.replace("font.family: sans-serif", f"font.family: {font}")
            if line.startswith("font.family")
            else line
        )
        for line in style_lines
    ]

    # Use a temporary style file with updated font family
    with open("temp_style.mplstyle", "w") as f:
        f.writelines(new_style_lines)

    plt.style.use("temp_style.mplstyle")
    print(f"Matplotlib style set to: {style_path} with font {font}")

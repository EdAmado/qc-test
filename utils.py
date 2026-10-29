"""
utils.py
General utility functions for the Hybrid Quantum Neural Network project.
"""

from pathlib import Path
import json
import random
import cv2
import numpy as np
import torch


# Reproducibility

def set_seed(seed: int = 42):
    """
    Set all random seeds to ensure reproducibility.
    Parameters
    ----------
    seed : int
        Random seed.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# Directories

def create_directories():
    """
    Create the folders used by the project.
    """

    folders = [
        "results",
        "results/models",
        "results/metrics",
        "results/plots",
        "results/circuits",
    ]

    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)


# Image utilities

def load_grayscale_image(path):
    """
    Load a grayscale PNG image.
    Parameters
    ----------
    path : str
    Returns
    -------
    ndarray
    """

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(path)

    return image


def normalize_image(image):
    """
    Normalize image into [0,1].
    """

    return image.astype(np.float32) / 255.0


# Dataset

def load_dataset(dataset_path):
    """
    Load all images and labels.
    Expected structure

    dataset/
        train/
            0/
            1/
        test/
            0/
            1/
    """

    images = []
    labels = []
    dataset_path = Path(dataset_path)

    for label in [0, 1]:
        folder = dataset_path / str(label)
        for file in sorted(folder.glob("*.png")):
            image = load_grayscale_image(file)
            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)


# PyTorch

def to_tensor(images):
    images = np.array(images, dtype=np.float32)
    return torch.tensor(images)


def to_device(model):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    return device


# Model persistence

def save_model(model, filename):
    path = Path("results/models")
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        path / filename,
    )


def load_model(model, filename):
    path = Path("results/models") / filename
    model.load_state_dict(
        torch.load(
            path,
            map_location="cpu"
        )
    )

    model.eval()

    return model


# Metrics

def save_metrics(metrics, filename="metrics.json"):
    path = Path("results/metrics")
    path.mkdir(parents=True, exist_ok=True)
    with open(path / filename, "w") as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )


# Plots

def save_plot(fig, filename):
    path = Path("results/plots")
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path / filename,
        dpi=300,
        bbox_inches="tight",
    )
from typing import Tuple

import h5py
import numpy as np


def synthesize(
        filepath: str,
        dtype: str = 'float32',
        channels: int = 20,
        n_samples: int = 1460,
        n_timesteps: int = 1,
        dataset_name: str = 'fields',
        img_size: Tuple[int, int] = (360, 720)
) -> None:
    """
    Creates and populates a synthetic dataset with random data in an HDF5 file, intended for testing or simulation purposes.

    :param filepath: Path to the HDF5 file where the dataset will be stored.
    :param channels: Number of channels for each image/sample in the dataset.
    :param n_timesteps: Number of time steps per sample, treated as separate samples.
    :param n_samples: Number of base samples to generate.
    :param dtype: Data type of the elements in the dataset (e.g., 'float32', 'float64').
    :param dataset_name: Name of the dataset to be created within the HDF5 file.
    :param img_size: Dimensions (height, width) of each image/sample as a tuple.

    This function generates a dataset of specified dimensions and data type within an HDF5 file. Each sample is filled with
    random numbers generated according to the specified data type. The function is useful for creating datasets for development
    and testing when actual data is not available.

    Example:
        >>> synthesize('path/to/data.h5', channels=10, n_timesteps=5, n_samples=1000, dtype='float32', img_size=(224, 224))
        This creates a dataset with 5000 random images of size 224x224 with 10 channels.
    """
    with h5py.File(filepath, 'w') as f:
        shape: Tuple[int, int, int, int] = (
                n_samples * n_timesteps,
                channels,
                img_size[0],
                img_size[1]
        )

        dataset = f.create_dataset(dataset_name, shape, dtype=dtype)

        for i in range(n_samples * n_timesteps):
            dataset[i, ...] = np.random.rand(
                channels,
                img_size[0],
                img_size[1]
            ).astype(dtype)

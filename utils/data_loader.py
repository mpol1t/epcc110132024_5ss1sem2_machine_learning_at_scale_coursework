import glob
import logging
import os
from typing import Optional

import h5py
import numpy as np
import torch
import torch.utils.data.dataloader
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from utils.y_params import YParams


class ERA5Dataset(Dataset):
    def __init__(self, params: YParams, location: str, train: bool) -> None:
        self.params = params
        self.location = location
        self.train = train
        self.dt = params.dt
        self.n_in_channels = params.n_in_channels
        self.n_out_channels = params.n_out_channels
        self.normalize = True
        self.means = np.load(params.data_loader_global_means_path)[0]
        self.stds = np.load(params.data_loader_global_stds_path)[0]
        self.limit_nsamples = params.limit_nsamples if train else params.limit_nsamples_val

        self._get_files_stats()

    def _get_files_stats(self) -> None:
        """
        Collects statistics from the dataset files such as paths, years, and sample sizes.
        Initializes additional properties including the total number of samples.
        """
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()

        self.years = [
                int(os.path.splitext(os.path.basename(x))[0][-4:]) for x in self.files_paths
        ]

        self.n_years = len(self.files_paths)

        with h5py.File(self.files_paths[0], 'r') as _f:
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f['fields'].shape[0]
            self.img_shape_x = self.params.img_size[0]
            self.img_shape_y = self.params.img_size[1]

            assert (self.img_shape_x <= _f['fields'].shape[2] and self.img_shape_y <= _f['fields'].shape[
                3]), 'image shapes are greater than dataset image shapes'

        self.n_samples_total = self.n_years * self.n_samples_per_year

        if self.limit_nsamples is not None:
            self.n_samples_total = min(self.n_samples_total, self.limit_nsamples)
            logging.info("Overriding total number of samples to: {}".format(self.n_samples_total))

        self.files = [None for _ in range(self.n_years)]

        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info(
            "Found data at path {}. "
            "Number of examples: {}. "
            "Image Shape: {} x {} x {}".format(
                self.location,
                self.n_samples_total,
                self.img_shape_x,
                self.img_shape_y,
                self.n_in_channels
            )
        )

    def _open_file(self, year_idx):
        """
        Opens the HDF5 file for a given year index and caches it to reduce I/O operations.

        :param year_idx: Index of the year in the dataset to open the file for.
        """
        self.files[year_idx] = h5py.File(
            self.files_paths[year_idx], 'r'
            #            self.files_paths[year_idx], 'r', driver='core', backing_store=False
        )['fields']

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        :return: Total number of samples available for training or validation.
        """
        return self.n_samples_total

    def _normalize(self, img):
        """
        Normalizes an image using pre-computed global mean and standard deviation.

        :param img: The image tensor to normalize.
        :return: Normalized image tensor.
        """
        if self.normalize:
            img -= self.means
            img /= self.stds
        return torch.as_tensor(img)

    def __getitem__(self, global_idx):
        """
        Retrieves an item by its global index from the dataset, handling the logic to access the correct year and
        sample within that year.

        :param global_idx: The global index across all years and samples.
        :return: A tuple containing the input and target tensors for the model.
        """
        year_idx = int(global_idx / self.n_samples_per_year)  # which year
        local_idx = int(global_idx % self.n_samples_per_year)  # which sample in that year

        # open image file
        if self.files[year_idx] is None:
            self._open_file(year_idx)

        step = self.dt  # time step

        # boundary conditions to ensure we don't pull data that is not in a specific year
        local_idx = local_idx % (self.n_samples_per_year - step)
        if local_idx < step:
            local_idx += step

        return self._normalize(
            self.files[year_idx][local_idx, :, 0:self.img_shape_x, 0:self.img_shape_y]
        ), self._normalize(
            self.files[year_idx][local_idx + step, :, 0:self.img_shape_x, 0:self.img_shape_y]
        )


class ERA5DataModule(LightningDataModule):
    def __init__(self, params: YParams) -> None:
        super().__init__()

        self.params = params

        self.num_data_workers: int = params.num_data_workers

        self.val_dataset: Optional[ERA5Dataset] = None
        self.test_dataset: Optional[ERA5Dataset] = None
        self.train_dataset: Optional[ERA5Dataset] = None

        self.batch_size: int = params.global_batch_size

    @staticmethod
    def worker_init(worker_id: int):
        """
        Initializes a worker for data loading by setting the random seed to ensure reproducibility.

        :param worker_id: Identifier for the worker.
        """
        worker_seed: int = (torch.utils.data.get_worker_info().seed + worker_id) % (2 ** 32 - 1)

        np.random.seed(worker_seed)
        logging.info(f"Initializing worker {worker_id} with seed {worker_seed}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up datasets for the specified stage of the model's training or testing process.

        :param stage: Specifies the stage ('fit', 'validate', 'test', or None) for which to set up the datasets.
        """
        # Define file patterns for different stages
        if stage == 'fit' or stage is None:
            self.val_dataset = ERA5Dataset(
                train=False,
                params=self.params,
                location=self.params.data_loader_valid_data_path
            )
            self.train_dataset = ERA5Dataset(
                train=True,
                params=self.params,
                location=self.params.data_loader_train_data_path
            )
        if stage == 'test' or stage is None:
            self.test_dataset = ERA5Dataset(
                train=False,
                params=self.params,
                location=self.params.data_loader_inf_data_path
            )

    def train_dataloader(self) -> torch.utils.data.dataloader.DataLoader:
        """
        Creates a DataLoader for the training dataset.

        :return: DataLoader configured for the training dataset with shuffling and other parameters.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.params.num_data_workers,
            worker_init_fn=self.worker_init,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=self.params.data_loader_prefetch_factor
        )

    def val_dataloader(self) -> torch.utils.data.dataloader.DataLoader:
        """
        Creates a DataLoader for the validation dataset.

        :return: DataLoader configured for the validation dataset without shuffling.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.params.num_data_workers,
            worker_init_fn=self.worker_init,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=self.params.data_loader_prefetch_factor
        )

    def test_dataloader(self):
        """
        Creates a DataLoader for the test dataset.

        :return: DataLoader configured for the test dataset without shuffling.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.params.num_data_workers,
            worker_init_fn=self.worker_init,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=self.params.data_loader_prefetch_factor
        )

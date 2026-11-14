"""Picklable references to converted RGB frames; open HDF5 inside each worker."""
from dataclasses import dataclass

import h5py
import numpy as np


@dataclass(frozen=True)
class HDF5ImageFrame:
    path: str
    dataset: str
    index: int

    def __array__(self, dtype=None, copy=None):
        with h5py.File(self.path, 'r') as h:
            value = np.asarray(h[self.dataset][self.index], dtype=dtype)
        return value.copy() if copy else value


class HDF5ImageSequence:
    lazy_frames = True

    def __init__(self, path, dataset, indices=None):
        self.path, self.dataset = str(path), str(dataset)
        if indices is None:
            with h5py.File(self.path, 'r') as h:
                self.indices = np.arange(len(h[self.dataset]))
        else:
            self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        selected = self.indices[key]
        if np.ndim(selected) == 0:
            return HDF5ImageFrame(self.path, self.dataset, int(selected))
        return HDF5ImageSequence(self.path, self.dataset, selected)

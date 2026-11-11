"""Print HDF5 dataset paths, shapes and dtypes without reading array contents."""

import argparse
import h5py
from adaspevla.datasets.hdf5_common import dataset_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="One episode HDF5 file")
    args = parser.parse_args()
    with h5py.File(args.path, "r") as file:
        print(dataset_inventory(file))


if __name__ == "__main__":
    main()

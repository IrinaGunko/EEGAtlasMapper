import os
import h5py
import numpy as np
import datetime

class H5FileHandler:

    @staticmethod
    def save_h5_file(filepath, data_dict, attr_dict=None, overwrite=False):
        if os.path.exists(filepath) and not overwrite:
            print(f"File already exists: {filepath}. Skipping save.")
            return
        with h5py.File(filepath, "w") as f:
            for key, value in data_dict.items():
                f.create_dataset(key, data=value, compression="gzip", compression_opts=9)
            if attr_dict:
                for key, value in attr_dict.items():
                    f.attrs[key] = value

        print(f"Data saved successfully to: {filepath}")

    @staticmethod
    def load_h5_file(filepath):

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None

        data_dict = {}
        attr_dict = {}

        with h5py.File(filepath, "r") as f:
            for key in f.keys():
                data_dict[key] = f[key][:]
            for key in f.attrs.keys():
                attr_dict[key] = f.attrs[key]

        print(f"Loaded data from {filepath}")
        return data_dict, attr_dict

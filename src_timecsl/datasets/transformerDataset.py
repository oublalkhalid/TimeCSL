# Identifiability Guarantees For Time Series Representation via Contrastive Sparsity-inducing
# Copyright 2024, ICLR 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple
from ast import arg
from importlib.resources import path
import sys
import json
import argparse
import numpy as np
from sklearn import exceptions
import h5py
import pickle
import pandas as pd
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    ConcatDataset,
    IterableDataset,
)
import os
import time

# from constants.constants import *
# from utils.utils import *
import time
import tqdm as tq
from typing import Union
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import os
import numpy as np
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable, Tuple
import torch
import pandas as pd
from pathlib import Path


params_appliance = {
    "kettle": {
        "windowlength": 599,
        "on_power_threshold": 2000,
        "max": 3998,
        "mean": 700,
        "std": 1000,
        "s2s_length": 128,
    },
    "microwave": {
        "windowlength": 599,
        "on_power_threshold": 200,
        "max": 3969,
        "mean": 500,
        "std": 800,
        "s2s_length": 128,
    },
    "fridge": {
        "windowlength": 599,
        "on_power_threshold": 50,
        "min": 0.0,
        "max": 742.0,
        "mean": 36.63640635395875,
        "std": 54.85663130200055,
        "s2s_length": 512,
    },
    "dishwasher": {
        "windowlength": 599,
        "on_power_threshold": 10,
        "min": 0.0,
        "max": 2471.0,
        "mean": 14.583158088584735,
        "std": 146.6048618229097,
        "s2s_length": 1536,
    },
    "washing_machine": {
        "windowlength": 599,
        "on_power_threshold": 20,
        "min": 0.0,
        "max": 4423.0,
        "mean": 6.168776732249786,
        "std": 92.62301239130836,
        "s2s_length": 2000,
    },
    "oven": {
        "windowlength": 599,
        "on_power_threshold": 10,
        "min": 0.0,
        "max": 4164.0,
        "mean": 68.68922333903622,
        "std": 406.32727628192333,
        "s2s_length": 1536,
    },
    "stove": {
        "windowlength": 599,
        "on_power_threshold": 10,
        "min": 0.0,
        "max": 3739.0,
        "mean": 17.974294268605647,
        "std": 163.4856663758999,
        "s2s_length": 1536,
    },
    "clothes_dryer": {
        "windowlength": 599,
        "on_power_threshold": 10,
        "min": 0.0,
        "max": 5521.0,
        "mean": 31.96659241041726,
        "std": 365.2979005006241,
        "s2s_length": 1536,
    },
    "ev": {
        "windowlength": 256,
        "on_power_threshold": 2000,
        "min": 0,
        "max": 6000,
        "mean": 2500,
        "std": 300,
        "s2s_length": 2000,
    },
    "all_device": {
        "windowlength": 256,
        "on_power_threshold": 2000,
        "min": 0.0,
        "max": 2471.0,
        "mean": 14.583158088584735,
        "std": 146.6048618229097,
        "s2s_length": 2000,
    },
}


class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path="/tsi/data_education/Ladjal/koublal/data_cluster/ev-washing_machine-dishwasher-fridge-oven-stove-clothes_dryer_features_PQS_targets_P_dt-60S_nbHouses-443.hdf5",
        house_index: int = "ID_245",
        window_size: int = 256,
        strid: int = 1,
        appliance_to_track=["ev"],
        features=["power_active", "power_reactive", "power_apparent"],
        targets=["power_active"],
        target_seq_len: int = 256,
        **kwargs,
    ):

        super().__init__()
        self.target_seq_len = target_seq_len
        self.h5_path = h5_path
        self.id = house_index
        self.window_size = window_size
        self.stride = strid
        self.targets = targets
        self.features = features

        self.appliance_to_track = appliance_to_track
        with h5py.File(self.h5_path, "r+") as db:
            self.resolution = db[self.id].attrs["RESOLUTION"]
            self.LEN = db[self.id].attrs["LEN"]
            self.X = np.array(db[self.id]["data"][:])
            self.ground = np.array(db[self.id]["label"][:])
            print("self.ground", self.ground.shape)
            self.feature_in_dataset = db.attrs["infos_features"]
            print("self.feature_in_dataset", self.feature_in_dataset)

            if self.ground.shape[1] == 7:
                self.target_in_dataset = [
                    "washing_machine_power_active",
                    "dishwasher_power_active",
                    "fridge_power_active",
                    "oven_power_active",
                    "stove_power_active",
                    "clothes_dryer_power_active",
                    "ev_power_active",
                ]
            else:
                self.target_in_dataset = db.attrs["infos_tragets"]

            self.X = pd.DataFrame(self.X, columns=self.feature_in_dataset)
            self.ground = pd.DataFrame(self.ground, columns=self.target_in_dataset)
            self.X = self.X[features]
            self.aggregate_ = self.X
            # select only the appliance needed i.e appliance_to_track:
            all_colums = [
                m + "_{}".format(puissance)
                for m in self.appliance_to_track
                for puissance in self.targets
            ]

            self.ground = self.ground[all_colums]  # .to_numpy()
            # Create df
            self.df = pd.DataFrame(self.X, columns=self.features)
            self.df[all_colums] = self.ground.copy()
            self.ground = self.ground.to_numpy()

            self.y_mean = np.array(
                [
                    params_appliance[k]["mean"]
                    for k in self.appliance_to_track
                    for m in self.targets
                ]
            )
            self.y_std = np.array(
                [
                    params_appliance[k]["std"]
                    for k in self.appliance_to_track
                    for m in self.targets
                ]
            )
            self.y_min = np.array(
                [
                    params_appliance[k]["min"]
                    for k in self.appliance_to_track
                    for m in self.targets
                ]
            )
            self.y_max = np.array(
                [
                    params_appliance[k]["max"]
                    for k in self.appliance_to_track
                    for m in self.targets
                ]
            )

            norm_type = "min_max"

            if norm_type == "satndarization":
                print(f"norm_type {norm_type}")
                self.X_mean = np.array(
                    [params_appliance["all_device"]["mean"] for m in self.features]
                )
                self.X_std = np.array(
                    [params_appliance["all_device"]["std"] for m in self.features]
                )
                self.y = (self.ground - self.y_mean) / self.y_std
                self.X = (self.X.to_numpy() - self.X_mean) / self.X_std

            elif norm_type == "soft_satndarization":
                print(f"norm_type {norm_type}")
                self.y = (self.ground - self.y_mean) / self.y_std
                self.scaler_agg = StandardScaler()
                self.scaler_agg.fit(self.X.to_numpy())
                self.X = self.scaler_agg.transform(self.X.to_numpy())

            elif norm_type == "min_max":
                print(f"norm_type {norm_type}")
                self.y = (self.ground - self.y_mean) / (self.y_max - self.y_min)
                self.scaler_agg = MinMaxScaler()
                self.scaler_agg.fit(self.X.to_numpy())
                self.X = self.scaler_agg.transform(self.X.to_numpy())
            else:
                print("Error -> Not ``norm_type`` not supported")

            self.test = None
            self.index = np.arange(0, len(self.X))

    def __len__(self):
        return int(np.ceil((len(self.X) - self.window_size) / self.stride) + 1)

    def __getitem__(self, index):
        start_index = index * self.stride
        end_index = np.min((len(self.X), index * self.stride + self.window_size))
        end_index_trg = np.min((len(self.X), index * self.stride + self.window_size))

        src = self.padding_seqs(self.X[start_index:end_index])
        trg = self.padding_seqs(self.y[start_index - 1 : end_index - 1])
        trg_y = self.padding_seqs(self.y[start_index:end_index])
        index = self.padding_seqs(self.index[start_index:end_index])

        return index, src, trg, trg_y

    def padding_seqs(self, in_array):
        if len(in_array) == self.window_size:
            return in_array
        try:
            out_array = np.zeros((self.window_size, in_array.shape[1]))
        except:
            out_array = np.zeros(self.window_size)
        length = len(in_array)
        out_array[:length] = in_array
        return out_array

    def apply_window_in(self):
        self.X, self.y = np.concatenate(
            (np.zeros((self.window_size, self.X.shape[1])), self.X)
        ), np.concatenate((np.zeros((self.window_size, self.y.shape[1])), self.y))
        self.index = np.arange(0, len(self.X))

    def apply_window_end(self):
        self.X, self.y = np.concatenate(
            (self.X, np.zeros((self.window_size, self.X.shape[1])))
        ), np.concatenate((self.y, np.zeros((self.window_size, self.y.shape[1]))))
        self.index = np.arange(0, len(self.X))

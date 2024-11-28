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
from datasources.datasource_BankNilm import Datasource_BANKNILM, Processing_data
from datasources.preprocessing_lib import *
from constants.constants import ModelTopologie
import time
from constants.constants import *
from utils.utils import *
import time
import tqdm as tq
from typing import Union
import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib


class AddNoiseTransform:
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def __call__(self, sequence):
        noise = torch.randn_like(sequence) * self.noise_level
        return sequence + noise


class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path=None,
        house_index: int = 1,
        window_size: int = None,
        length_seq_out: int = None,
        strid: int = None,
        processMethod: SupportedPreprocessingMethods = None,
        testing_phase=False,
        appliance_to_track=None,
        features=None,
        targets=None,
        normalization_type=None,
        resampling: str = None,
        dataframe: pd.DataFrame = None,
        inference=False,
        **kwargs,
    ):
        self.flag = "test"
        self.transform = AddNoiseTransform(noise_level=10e-6)
        self.h5_path = h5_path
        self.parquet_dir = h5_path
        self.id = house_index
        self.testing_phase = testing_phase
        self.processMethod = processMethod
        self.window_size = window_size
        self.length_seq_out = length_seq_out
        self.stride = strid
        self.targets = targets
        self.features = features
        self.appliance_to_track = appliance_to_track
        self.resampling = resampling
        self.dataframe = dataframe
        self.normalization_type = normalization_type

        # Start functions
        if not inference:
            config_yaml_path = os.path.join(self.parquet_dir, "config.yaml")
            # Load the configuration from the YAML file
            with open(config_yaml_path, "r") as f:
                self.config_data = yaml.safe_load(f)
            self.load_data()

            # with h5py.File(self.h5_path, 'r+') as db:
            # self.resolution = db[self.id].attrs['RESOLUTION']
            # self.LEN = db[self.id].attrs['LEN']
            # self.X = np.array(db[self.id]['data'][:])
            # self.ground = np.array(db[self.id]['label'][:])
            # print("self.ground", self.ground.shape)
            # self.feature_in_dataset = db.attrs['infos_features']
            # print("self.feature_in_dataset", self.feature_in_dataset)
            # # if self.ground.shape[1] ==7:
            # #     self.target_in_dataset = db.attrs['infos_tragets']
            # #     #print('target_in_dataset*', self.target_in_dataset)
            # #     self.target_in_dataset = ['washing_machine_power_active',
            # #                             'dishwasher_power_active',
            # #                             'fridge_power_active',
            # #                             'oven_power_active',
            # #                             'stove_power_active',
            # #                             'clothes_dryer_power_active',
            # #                             'ev_power_active']
            # #     #print('target_in_dataset**', self.target_in_dataset)
            # # else:
            # self.target_in_dataset = db.attrs['infos_tragets']

            # self.X = pd.DataFrame(self.X, columns=self.feature_in_dataset)
            # self.ground = pd.DataFrame(self.ground, columns=self.target_in_dataset)

            # # if self.resampling:
            # #     print("Use resampling:", self.resampling)
            # #     time_index = pd.date_range(start='2023-01-01 00:00:00', periods=len(self.X), freq='1T')
            # #     self.X.index = time_index
            # #     self.ground.index = time_index
            # #     # Example: Resample to 30-minute intervals and take the mean
            # #     self.X = self.X.resample(self.resampling).asfreq()
            # #     self.ground = self.ground.resample(self.resampling).asfreq()

            # self.X = self.X[self.features]
            # self.aggregate_ = self.X
            # #select only the appliance needed i.e appliance_to_track:
            # all_colums = [m+'_{}'.format(puissance) for m in self.appliance_to_track for puissance in self.targets]
            # for cols in all_colums:
            #     if cols not in self.ground.columns:
            #         print('cols**', cols)
            #         self.ground[cols] = 0 # No data activation

            # self.ground = self.ground[all_colums] #.to_numpy()
            # #Create df
            # self.df = pd.DataFrame(self.X, columns=self.features)
            # self.df[all_colums]= self.ground.copy()
            # self.ground = self.ground.to_numpy()
            # self.y = self.ground

            # self.test = None
            # self.index = np.arange(0,len(self.X))
            # self.embedings = self.embedings_ones_zeros(self.ground)
        else:
            self.resolution = 60
            self.X = self.dataframe  # pd.read_csv(self.csv_path)
            self.ground = None
            self.y = self.X
            # self.feature_in_dataset = ['power_active','power_reactive','power_apparent']
            # print("self.feature_in_dataset", self.feature_in_dataset)

            self.X = self.X[self.features]
            self.aggregate_ = self.X
            # select only the appliance needed i.e appliance_to_track:
            all_colums = [
                m + "_{}".format(puissance)
                for m in self.appliance_to_track
                for puissance in self.targets
            ]
            self.df = pd.DataFrame(self.X, columns=self.features)
            self.test = None
            self.index = np.arange(0, len(self.X))

    # Assuming this is part of a class, and you have a method that initializes or loads the dataset
    def load_data(self):
        # Read the ID to load data
        # id = self.id  # assuming self.id is set somewhere in your clas
        # Load the parquet data
        parquet_file = os.path.join(self.parquet_dir, f"{self.id}.parquet")
        df = pd.read_parquet(parquet_file)
        df = df.replace({np.nan: 0, None: 0, np.inf: 0, -np.inf: 0})
        df.sort_values(by="timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()

        # Extract attributes from the YAML config
        self.resolution = self.config_data[self.id]["RESOLUTION"]
        self.LEN = self.config_data[self.id]["LEN"]
        self.feature_in_dataset = self.config_data["features"]
        self.target_in_dataset = self.config_data["targets"]

        # Separate features (X) and labels (ground) based on YAML config
        print(
            "self.target_in_dataset", self.target_in_dataset, "df.columns", df.columns
        )
        self.X = df[self.feature_in_dataset]

        self.ground = df[self.target_in_dataset]

        print("self.ground shape:", self.ground.shape)
        print("self.feature_in_dataset:", self.feature_in_dataset)

        # If resampling is needed
        # if self.resampling:
        #     print("Use resampling:", self.resampling)
        #     time_index = pd.date_range(start='2023-01-01 00:00:00', periods=len(self.X), freq='1T')
        #     self.X.index = time_index
        #     self.ground.index = time_index
        #     # Example: Resample to the specified interval
        #     self.X = self.X.resample(self.resampling).asfreq()
        #     self.ground = self.ground.resample(self.resampling).asfreq()

        # Select only the specified features and targets
        self.X = self.X[self.features]
        self.aggregate_ = self.X
        all_colums = [
            m + "_{}".format(puissance)
            for m in self.appliance_to_track
            for puissance in self.targets
        ]

        # Ensure all target columns exist in the ground DataFrame
        for cols in all_colums:
            if cols not in self.ground.columns:
                print("Adding missing column:", cols)
                self.ground[cols] = 0  # No data activation
                print("cols", cols)

        self.ground = self.ground[all_colums]

        # Create the final DataFrame with features and selected target columns
        self.df = pd.DataFrame(self.X, columns=self.features)
        self.df[all_colums] = self.ground.copy()

        # Convert ground truth to numpy array for further processing
        self.ground = self.ground.to_numpy()
        self.y = self.ground

    # Example usage
    # Assuming this is a method in your class and you have set necessary variables in the class
    # self.id = 'ID_0'  # Example ID, replace with the actual ID you need to load
    # self.features = ['feature1', 'feature2']  # List of features you want to use
    # self.appliance_to_track = ['appliance1', 'appliance2']  # List of appliances to track
    # self.targets = ['target1', 'target2']  # List of target powers to track
    # self.resampling = '30T'  # Example resampling interval
    # self.load_data()  # Call the method to load data

    def scaling(self, infernece: bool = False):
        if infernece:
            print(f"Type: {self.normalization_type}, Infernece: {infernece}")
            if self.normalization_type == "standarization":
                print(f"normalization_type {self.normalization_type}")
                self.X_mean = np.array(
                    [params_appliance["all_device"]["mean"] for m in self.features]
                )
                self.X_std = np.array(
                    [params_appliance["all_device"]["std"] for m in self.features]
                )
                self.X = (self.X - self.X_mean) / self.X_std
                self.y_unorm = self.X
                self.y = self.X

            elif self.normalization_type == "soft_satndarization":
                print(f"normalization_type {self.normalization_type}")
                self.scaler_agg = StandardScaler()
                self.scaler_agg.fit(self.X)
                self.X = self.scaler_agg.transform(self.X)
                # no label avilable so we set self.y by self.X to avoid torch None issue
                self.y = self.X
                self.y_unorm = self.X

            elif self.normalization_type == "min_max":
                print(f"normalization_type {self.normalization_type}")
                # #self.scaler_agg = MinMaxScaler()
                # self.scaler_agg = joblib.load('agg_scaler_min_max_train.gz')
                # self.X = self.scaler_agg.transform(self.X)
                # self.y = self.X
                self.X_min = np.array(
                    [params_appliance["all_device"]["min"] for m in self.features]
                )
                self.X_max = np.array(
                    [params_appliance["all_device"]["max"] for m in self.features]
                )
                # self.y = (self.y - self.y_min) / (self.y_max - self.y_min)
                self.X = (self.X - self.X_min) / (self.X_max - self.X_min)
                self.y = self.X

            elif self.normalization_type == "log_norm":
                self.X = np.log(self.X + 1)
                self.y = np.log(self.X + 1)

            elif self.normalization_type == "maximum_absolute":
                print(f"normalization_type {self.normalization_type}")
                self.X_min = np.array(
                    [params_appliance["all_device"]["min"] for m in self.features]
                )
                self.X_max = np.array(
                    [params_appliance["all_device"]["max"] for m in self.features]
                )
                # self.y = (self.y - self.y_min) / (self.y_max - self.y_min)
                self.X = (self.X - self.X_min) / (self.X_max - self.X_min)
                self.y = self.X

            else:
                self.y = self.y  # - self.y_mean)/self.y_std
                self.X = self.X  # - self.X_mean)/self.X_std

            self.test = None
            self.index = np.arange(0, len(self.X))
        else:
            if self.normalization_type == "standarization":
                print(f"normalization_type {self.normalization_type}")
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
                self.X_mean = np.array(
                    [params_appliance["all_device"]["mean"] for m in self.features]
                )
                self.X_std = np.array(
                    [params_appliance["all_device"]["std"] for m in self.features]
                )
                self.y_unorm = self.y
                self.y = (self.y - self.y_mean) / self.y_std
                self.X = (self.X - self.X_mean) / self.X_std

            elif self.normalization_type == "min_max":
                print(f"normalization_type {self.normalization_type}")
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
                self.X_min = np.array(
                    [params_appliance["all_device"]["min"] for m in self.features]
                )
                self.X_max = np.array(
                    [params_appliance["all_device"]["max"] for m in self.features]
                )
                self.y_unorm = self.y
                self.y = (self.y - self.y_min) / (self.y_max - self.y_min)
                self.X = (self.X - self.X_min) / (self.X_max - self.X_min)

            # elif self.normalization_type == "min_max":
            #     print(f"normalization_type {self.normalization_type}")
            #     # self.y_mean = np.array([params_appliance[k]['mean'] for k in self.appliance_to_track for m in self.targets])
            #     # self.y_std = np.array([params_appliance[k]['std'] for k in self.appliance_to_track for m in self.targets])
            #     # self.y_min = np.array([params_appliance[k]['min'] for k in self.appliance_to_track for m in self.targets])
            #     # self.y_max = np.array([params_appliance[k]['max'] for k in self.appliance_to_track for m in self.targets])
            #     #self.y = (self.y - self.y_mean)/(self.y_max - self.y_min)
            #     self.scaler_agg = MinMaxScaler()
            #     self.scaler_labels = MinMaxScaler()
            #     self.scaler_agg.fit(self.X)
            #     self.scaler_labels.fit(self.y)
            #     joblib.dump(self.scaler_agg, 'agg_scaler_min_max_train.gz')
            #     joblib.dump(self.scaler_labels, 'labels_scaler_min_max_train.gz')
            #     self.X = self.scaler_agg.transform(self.X)
            #     self.y  = self.scaler_labels.transform(self.y)

            elif self.normalization_type == "log_norm":
                print(f"Type: {self.normalization_type}, Infernece: {infernece}")
                self.y = np.log(self.y + 1e-9)
                self.X = np.log(self.X + 1e-9)

            elif self.normalization_type == "maximum_absolute":
                print(f"normalization_type {self.normalization_type}")
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
                self.X_mean = np.array(
                    [params_appliance["all_device"]["mean"] for m in self.features]
                )
                self.X_max = np.array(
                    [params_appliance["all_device"]["max"] for m in self.features]
                )
                self.X_std = np.array(
                    [params_appliance["all_device"]["std"] for m in self.features]
                )
                self.y = self.y / self.y_max
                self.X = self.X / self.X_max

            else:
                self.y = self.y
                self.X = self.X
                # raise NotImplementedError("Other types of normalization has not yet been implemented")

    def __len__(self):
        return int(np.ceil((len(self.X) - self.window_size) / self.stride) + 1)

    def __getitem__(self, index):
        # index = 0 --> 00:00 --> 01:00 (jour j)
        start_index = index * self.stride
        end_index = np.min((len(self.X), index * self.stride + self.window_size))
        # end_index_target = np.min((len(self.X), index * self.stride + self.length_seq_out))
        # print("self.length_seq_out", self.length_seq_out, "self.stride", self.stride)

        # start_index_target = index * self.stride

        end_index_target = np.min(
            (len(self.X), index * self.stride + self.length_seq_out)
        )

        # length_seq_out
        x = self.padding_seqs(
            self.X[start_index:end_index], window_size=self.window_size
        )
        y = self.padding_seqs(
            self.y[start_index:end_index_target], window_size=self.length_seq_out
        )
        # Now for Y: extract the middle `seq_output_size` points from the 256 window
        # if len(X_window) == self.window_size:
        #     mid_point = self.window_size // 2  # Midpoint of the 256 window
        #     half_output_size = self.seq_output_size // 2  # Half of the output size

        #     # Calculate the start and end indices for the target window
        #     target_start = mid_point - half_output_size
        #     target_end = mid_point + half_output_size

        #     # Adjust for cases where seq_output_size is odd
        #     if self.seq_output_size % 2 != 0:
        #         target_end += 1

        #     # Extract the middle `seq_output_size` points from Y
        #     y = self.padding_seqs(self.y[start_index + target_start:start_index + target_end])
        # else:
        #     # Handle edge cases where the window is smaller than 256 (e.g., at the end of the data)

        y_unorm = self.padding_seqs(
            self.y_unorm[start_index:end_index_target], window_size=self.length_seq_out
        )
        embedings = self.padding_seqs(
            self.y[start_index - 1 : end_index - 1], window_size=self.window_size
        )
        index = self.padding_seqs(
            self.index[start_index:end_index_target], window_size=self.window_size
        )

        # if SupportedPreprocessingMethods(self.processMethod)== SupportedPreprocessingMethods.ROLLING_WINDOW:
        #     index, x, y, embedings = index[self.window_size-1:], x, y[self.window_size-1:], embedings[self.window_size-1:]

        # elif SupportedPreprocessingMethods(self.processMethod) == SupportedPreprocessingMethods.MIDPOINT_WINDOW:
        #     index, x, y, embedings = index[self.window_size//2], x, y[self.window_size//2], embedings[self.window_size//2]

        # elif SupportedPreprocessingMethods(self.processMethod) == SupportedPreprocessingMethods.SEQ_T0_SEQ:
        #     index, x, y, embedings = index, x, y, embedings

        if len(y.shape) > 1:
            y = y.reshape(y.shape[0], y.shape[1])
            index, x, y, embedings = (
                torch.Tensor(index),
                torch.Tensor(x.transpose(1, 0)),
                torch.Tensor(y.transpose(1, 0)),
                torch.Tensor(embedings.transpose(1, 0)),
            )
            y_unorm = torch.Tensor(y_unorm.transpose(1, 0))
        else:
            y = y.reshape(y.shape[0], 1)
            y_unorm = y_unorm.reshape(y.shape[0], 1)
            y_unorm = torch.Tensor(y_unorm)
            x, y, embedings = (
                torch.Tensor(x.transpose(1, 0)),
                torch.Tensor(y),
                torch.Tensor(embedings),
            )

        # if self.flag == 'train':
        #    x, y = self.transform(x), self.transform(y)
        # x = self.transform(x) ## add some noise to the aggregate noise
        # print("embedings", embedings.shape)
        return index, x, y, y_unorm  # embedings

    def padding_seqs(self, in_array, window_size):
        if len(in_array) == window_size:
            return in_array
        try:
            out_array = np.zeros((window_size, in_array.shape[1]))
        except:
            out_array = np.zeros(window_size)
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

    def embedings_ones_zeros(self, data):
        status = np.zeros(data.shape)
        if len(data.squeeze().shape) == 1:
            columns = 1
        else:
            columns = data.squeeze().shape[-1]
        self.threshold, self.min_on, self.min_off = None, None, None
        if not self.threshold:
            self.threshold = [10 for i in range(columns)]
        if not self.min_on:
            self.min_on = [1 for i in range(columns)]
        if not self.min_off:
            self.min_off = [1 for i in range(columns)]

        for i in range(columns):
            initial_status = data[:, i] >= self.threshold[i]
            status_diff = np.diff(initial_status)
            events_idx = status_diff.nonzero()

            events_idx = np.array(events_idx).squeeze()
            events_idx += 1

            if initial_status[0]:
                events_idx = np.insert(events_idx, 0, 0)

            if initial_status[-1]:
                events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

            events_idx = events_idx.reshape((-1, 2))
            on_events = events_idx[:, 0].copy()
            off_events = events_idx[:, 1].copy()
            assert len(on_events) == len(off_events)

            if len(on_events) > 0:
                off_duration = on_events[1:] - off_events[:-1]
                off_duration = np.insert(off_duration, 0, 1000)
                on_events = on_events[off_duration > self.min_off[i]]
                off_events = off_events[np.roll(off_duration, -1) > self.min_off[i]]

                on_duration = off_events - on_events
                on_events = on_events[on_duration >= self.min_on[i]]
                off_events = off_events[on_duration >= self.min_on[i]]
                assert len(on_events) == len(off_events)

            temp_status = data[:, i].copy()
            temp_status[:] = 0
            for on, off in zip(on_events, off_events):
                temp_status[on:off] = 1
            status[:, i] = temp_status
        return status


class ParallelTraining(Datasource_BANKNILM):
    def __init__(
        self,
        list_id_source: list = None,
        rawdata=None,
        save_path: str = None,
        appliance_to_track=None,
        window_size=None,
        length_seq_out: int = 256,
        sampling_value=None,
        sampling_as_freqAcquisition=None,
        interpolation_method=None,
        strid=None,
        name=None,
        features=None,
        targets=None,
        limit=None,
        weight_data=None,
        processMethod: SupportedPreprocessingMethods = None,
        test_size=None,
        batch_size=None,
        testing_phase=False,
        hdf5_dataset=None,
        bool_testing=False,
        resampling: str = None,
        normalization_type="satndarization",
        dataframe: pd.DataFrame = None,
        inference: bool = False,
        **kwargs,
    ):

        self.list_id_source = list_id_source
        self.hdf5_dataset = hdf5_dataset
        self.strid = strid
        self.window_size = window_size
        self.length_seq_out = length_seq_out
        self.processMethod = processMethod
        self.targets = targets
        self.appliance_to_track = appliance_to_track
        self.features = features
        self.normalization_type = normalization_type
        self.testing_phase = testing_phase
        self.train_val_split = 1 - test_size
        self.batch_size = batch_size
        self.resampling = resampling
        self.dataframe = dataframe
        t = tq.tqdm(self.list_id_source)
        self.iterator = iter(self.list_id_source)
        self.dic_houses, self.combination = {}, []
        for indice in t:
            ID = next(self.iterator)
            dataset = TimeseriesDataset(
                h5_path=self.hdf5_dataset,
                house_index=indice,
                window_size=self.window_size,
                length_seq_out=self.length_seq_out,
                strid=self.strid,
                processMethod=self.processMethod,
                testing_phase=self.testing_phase,
                appliance_to_track=self.appliance_to_track,
                features=self.features,
                targets=self.targets,
                normalization_type=self.normalization_type,
                resampling=self.resampling,
                dataframe=self.dataframe,
                inference=inference,
            )

            self.aggregate = dataset.aggregate_
            self.df = dataset.df
            # self.test = dataset.test
            self.y = dataset.y
            dataset.apply_window_in()
            dataset.apply_window_end()
            dataset.scaling(infernece=inference)
            self.combination.append(dataset)
            t.set_description("TimeseriesDataset: {}".format(ID))
            t.refresh()
            time.sleep(0.01)
        # add window to beginin and end
        print("Begin Concat dataset")
        self.dataset = ConcatDataset(self.combination)
        print("end Concat dataset:", len(self.dataset))
        print("end Concat dataset")
        self.train_size = int(self.train_val_split * len(self.dataset))
        self.val_size = len(self.dataset) - self.train_size

    def inverse_transform(self, agg_data, label_data=None):
        if label_data is None:
            agg_data_inverse = self.combination[0].scaler_agg.inverse_transform(
                agg_data
            )
            return agg_data_inverse
        else:
            agg_data_inverse = self.combination[0].scaler_agg.inverse_transform(
                agg_data
            )
            y_min = self.combination[0].y_mean
            y_max = self.combination[0].y_max
            y_min = self.combination[0].y_min
            y_std = self.combination[0].y_std
            label_data_inverse = agg_data * (y_max - y_min) + y_min
            return agg_data_inverse, label_data_inverse

    def prams(self):
        self.DATASET = h5py.File(self.backend_dataset, "r+")
        self.resolution = self.DATASET.attrs["RESOLUTION"]
        self.testing_phase = False
        if self.testing_phase:
            print("######## TEST PARMAS ###############")
            (
                self.maxs_agg,
                self.means_agg,
                self.sigma_agg,
                self.maxs_disag,
                self.means_disag,
                self.sigma_disag,
            ) = get_params(
                path_dataset=self.backend_dataset,
                targets=self.targets,
                infos_agg="infos_agg_test",
                infos_disag="infos_disag_test",
            )
        else:
            print("######## TRAIN PARMAS ###############")
            (
                self.maxs_agg,
                self.means_agg,
                self.sigma_agg,
                self.maxs_disag,
                self.means_disag,
                self.sigma_disag,
            ) = get_params(
                path_dataset=self.backend_dataset,
                targets=self.targets,
                infos_agg="infos_agg_train",
                infos_disag="infos_disag_train",
            )
        self.DATASET.close()

        return (
            self.resolution,
            self.maxs_agg,
            self.means_agg,
            self.sigma_agg,
            self.maxs_disag,
            self.means_disag,
            self.sigma_disag,
        )

    def train_val_dataloader(self):
        train_dataset, val_dataset = random_split(
            self.dataset,
            [self.train_size, self.val_size],
            generator=torch.Generator().manual_seed(1234),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )  # 0, pin_memory=False)
        print("Successfully : PREPARE TRAIN LOADER")
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )  # 0, pin_memory=False)
        print("Successfully : PREPARE VALIDATION LOADER")
        return train_loader, val_loader

    def train_val_dataset(self):
        train_dataset, val_dataset = random_split(
            self.dataset,
            [self.train_size, self.val_size],
            generator=torch.Generator().manual_seed(1234),
        )
        return train_dataset, val_dataset

    def test_dataloader(self):
        print("batch_size=self.batch_size:", self.batch_size)
        test_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        print("Successfully : PREPARE TEST LOADER", len(test_loader))
        return test_loader


class TimeseriesDataset_csv(torch.utils.data.Dataset, Processing_data):
    def __init__(
        self,
        window_size=None,
        strid=None,
        appliance_to_track=None,
        targets=["power_active"],
        features=["power_active", "power_reactive", "power_apparent"],
        csv__path: str = None,
        NormStdMethod=None,
        fillna_method=None,
        noise_factor=None,
        infos=None,
        **kwargs,
    ):

        super(Processing_data, self).__init__()
        self.h5_or_csv_path = csv__path
        self.window_size = window_size
        self.strid = strid
        self.appliance_to_track = appliance_to_track
        self.targets = targets
        self.features = features
        self.create_dataset()

    def __len__(self):
        return int(np.ceil((len(self.X) - self.window_size) / self.strid) + 1)

    def __getitem__(self, index):
        start_index = index * self.strid
        end_index = np.min((len(self.X), index * self.strid + self.window_size))
        x = self.padding_seqs(
            self.X[start_index:end_index], window_size=self.window_size
        )
        ind = self.padding_seqs(
            self.index[start_index:end_index], window_size=self.window_size
        )
        x = torch.tensor(x.transpose(1, 0))

        return torch.Tensor(ind), x, x, torch.Tensor(ind)

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

    def create_dataset(self):
        self.df = pd.read_csv(self.h5_or_csv_path, low_memory=False)
        self.ID = self.df["prm"][0]
        self.X = self.df[self.features]
        self.aggregate_ = self.X

        if len(self.appliance_to_track) > 1:
            X_mean = np.array(
                [params_appliance["all_device"]["mean"] for k in self.features]
            )
            X_std = np.array(
                [params_appliance["all_device"]["std"] for k in self.features]
            )
        else:
            X_mean = np.array(
                [
                    params_appliance[k]["mean"]
                    for k in self.appliance_to_track
                    for m in self.features
                ]
            )
            X_std = np.array(
                [
                    params_appliance[k]["std"]
                    for k in self.appliance_to_track
                    for m in self.features
                ]
            )
        self.X = (self.X - X_mean) / X_std
        self.X = self.X.to_numpy()
        self.test = None
        print("Type self.X,", type(self.X))
        print("self.X,", self.X.shape)
        self.index = np.arange(0, len(self.X))

    def apply_window_in(self):
        self.X = np.concatenate((np.zeros((self.window_size, self.X.shape[1])), self.X))
        self.index = np.arange(0, len(self.X))

    def apply_window_end(self):
        self.X = np.concatenate((self.X, np.zeros((self.window_size, self.X.shape[1]))))
        self.index = np.arange(0, len(self.X))

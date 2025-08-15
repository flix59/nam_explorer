

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor


class HousingDataset:
    def __init__(self, csv_file, target_column = "median_house_value", train_split=0.8, validation_split=0.1, test_split=0.1):
        if isinstance(csv_file, (str, Path)):
            self.data = pd.read_csv(csv_file)
        else:
            self.data = csv_file

        # self.features = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
        if target_column not in self.data.columns:
            raise ValueError(f"Target column {target_column} not in csv {self.data.columns}")
        self.features = []
        for column in self.data.columns:
            try:
                float(self.data[column][0])
                if column == target_column:
                    continue
                self.features.append(column)
            except:
                if column == target_column:
                    raise ValueError(f"Target column {target_column} can not be parsed to float: first value: {self.data[target_column][0]}")
                pass 
        self.data = self.data[self.features + [target_column]].dropna()
        data = torch.tensor(self.data[self.features].values, dtype=torch.float32)
        targets = torch.tensor(self.data[target_column].values, dtype=torch.float32)
        self.train_data, self.validation_data, self.test_data = self.split_data(data, train_split, validation_split, test_split)
        self.train_targets, self.validation_targets, self.test_targets = self.split_data(targets, train_split, validation_split, test_split)
        # normalize features
        self.scaler = StandardScaler()
        self.train_data = torch.tensor(self.scaler.fit_transform(self.train_data), dtype=torch.float32)
        self.validation_data = torch.tensor(self.scaler.transform(self.validation_data), dtype=torch.float32)
        self.test_data = torch.tensor(self.scaler.transform(self.test_data), dtype=torch.float32)
        self.target_scaler = StandardScaler()
        self.train_targets = torch.tensor(self.target_scaler.fit_transform(self.train_targets.view(-1, 1)).flatten(), dtype=torch.float32)
        self.validation_tarrets = torch.tensor(self.target_scaler.transform(self.validation_targets.view(-1, 1)).flatten(), dtype=torch.float32)
        self.test_targets = torch.tensor(self.target_scaler.transform(self.test_targets.view(-1, 1)).flatten(), dtype=torch.float32)


    def split_data(self, data: Tensor, train_split, validation_split, test_split):
        assert train_split + validation_split + test_split == 1.0, "Splits must sum to 1"
        total_size = data.size(0)
        train_size = int(total_size * train_split)
        validation_size = int(total_size * validation_split)
        return (data[:train_size], data[train_size:train_size + validation_size], data[train_size + validation_size:])
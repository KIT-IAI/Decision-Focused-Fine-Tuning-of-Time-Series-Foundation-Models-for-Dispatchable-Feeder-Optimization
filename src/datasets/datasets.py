import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch import from_numpy
import pickle
import glob

from src.configuration import DATASET_CONFIG, DATATSET_BOUNDS

class DFRTrainableBuildingDataset(Dataset):
    """
    A dataset class for the DFR trainable building dataset.
    This class is used to load the data for a specific building and prepare it for training.
    """
    
    def __init__(self, data, start, end, building_id, dataset, verbose=False, soe="simulated", scale_soe=True, soe_stats_only_small=False):
        if verbose:
            print(f"Creating Dataset {dataset}")
            print(f"Building ID: {building_id}")

        self.data_stats = data[DATATSET_BOUNDS["train"][0]:DATATSET_BOUNDS["train"][1]]
        self.data = data[start:end]
        self.context_length = DATASET_CONFIG["ctx"]
        self.prediction_length = DATASET_CONFIG["pdt"]

        _data = self.data[self.context_length:-self.prediction_length]
        self.valid_idx = _data.index[_data.index.hour==12]
        stats_data = self.data_stats[self.data_stats.index.hour==12]

        # only needed for exactly equal execution as in DFR
        if soe_stats_only_small:
            # only use the data which is used in training with cutting the context and prediction length # Would be needed for exakt equal execution as in DFR
            _subset = self.data_stats[self.context_length:-self.prediction_length]
            noon_indices = _subset.index[_subset.index.hour == 12]
            stats_data = stats_data.loc[noon_indices]


        min = stats_data.min()
        max = stats_data.max()
        mean = stats_data.mean()
        std = stats_data.std()

        # create a dataframe with the statistical information
        self.statistical_information = pd.DataFrame({"min": min, "max": max, "mean": mean, "std": std}, index=[0])

        # get the soe
        if soe == "simulated":
            path = glob.glob(f"data/data_res/optimisations_and_forecasts/training/results_optimisation/{building_id}/original/13.5/{dataset}/MAE/*/*")[0]
            if verbose:
                print("Simulated SoE")
        else:
            raise ValueError("SoE not implemented")

        with open(path, 'rb') as f:
            self.soe = pickle.load(f).results_online_final["e"]["house0"][0]

        self.soe = self.soe[self.valid_idx + pd.Timedelta(hours=12)]   

        # Concat SOE and statistical information
        stats_tmp = np.array([np.full((len(self.soe)),x) for x in self.statistical_information.values[0]])
        self.soe_stats = np.concatenate((self.soe.values.reshape((-1,1)), stats_tmp.T), axis=1)

        with open("models/surrogate/prosumption_1_0/feature_scaler/feature_scaler.pkl", 'rb') as f:
            soe_stats_scaler = pickle.load(f)

        if scale_soe:
            self.soe_stats = soe_stats_scaler.transform(self.soe_stats)
        else:
            self.soe_stats = self.soe_stats

        # Only for debugging purposes
        if verbose:
            print(f"Dataset {dataset} created, Shapes:", self.data.shape, self.soe.shape, self.soe_stats.shape)
            print(f"Dataset {dataset} created, Timestamps:", self.data.index[0], self.data.index[-1], self.soe.index[0], self.soe.index[-1], self.valid_idx[0], self.valid_idx[-1])
            last_index = self.valid_idx[-1]
            print(f"Last entry: {self.get_gt(len(self)-1)}")
            print("First entry" , self.get_gt(0))
    
    def __len__(self):
        return len(self.valid_idx)

    def get_naive_shift(self, idx, shift, length):
        return self.data.loc[self.valid_idx[idx] - pd.Timedelta(hours=shift):self.valid_idx[idx] - pd.Timedelta(hours=shift)+ pd.Timedelta(hours=1)*(length-1)].values
    
    def get_gt(self, idx):
        return self.data.loc[self.valid_idx[idx] : self.valid_idx[idx]+pd.Timedelta(hours=self.prediction_length-1)].values
    
    def get_time_by_index(self, idx):
        return self.valid_idx[idx]

    def __getitem__(self, idx):

        return ({ "past_target": from_numpy(self.data.loc[self.valid_idx[idx] - pd.Timedelta(hours=self.context_length):self.valid_idx[idx] - pd.Timedelta(hours=1)].values.reshape(-1,1)),    
                 "past_observed_target": from_numpy(np.ones((self.context_length,1)).astype(int)),
                 "past_is_pad": from_numpy(np.full((self.context_length), False)),
                 },
                   from_numpy(self.get_gt(idx)), 
                   from_numpy(self.soe_stats[idx]))
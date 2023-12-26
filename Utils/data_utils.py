import os
import torch
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def load_data(
    data_root,
    dataset_name,
    batch_size,
    random_seed,
    rm_filename=None
):
    if dataset_name == 'abilene':
        n_train = 15*7
        n_test = 1*7
        train_size = int(n_train * 288)
        test_size = int(n_test * 288)
        scale = 10**9
        tm_filename = 'abilene_tm.csv'
        if rm_filename is None:
            rm_filename = 'abilene_rm.csv'
    elif dataset_name == 'geant':
        n_train = 10*7
        n_test = 1*7
        train_size = int(n_train * 96)
        test_size = int(n_test * 96)
        scale = 10**7
        tm_filename = 'geant_tm.csv'
        if rm_filename is None:
            rm_filename = 'geant_rm.csv'
    else:
        raise NotImplementedError

    tm_filepath = os.path.join(data_root, tm_filename)
    rm_filepath = os.path.join(data_root, rm_filename)

    train_dataset = TMEDataset(tm_filepath, rm_filepath, train_size, test_size, period='train', scale=scale, seed=random_seed)
    test_dataset = TMEDataset(tm_filepath, rm_filepath, train_size, test_size, period='test', scale=scale, seed=random_seed)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader, train_dataset.rm


class TMEDataset(Dataset):
    def __init__(
        self, 
        tm_filepath, 
        rm_filepath,
        train_size, 
        test_size,
        period='train',
        scale=10**9,
        seed=2023
    ):
        super(TMEDataset, self).__init__()

        self.traffic_noisy_id = None
        self.link_noisy_id = None
        
        assert period in ['train', 'test'], ''
        self.period = period

        self.traffic, self.link, self.traffic_clean, self.rm =\
            self.read_data(tm_filepath, rm_filepath, train_size, test_size, scale, period, seed)
        
        _, self.dim_1 = self.link.shape
        self.len, self.dim_2 = self.traffic.shape

    def read_data(
        self,
        tm_filepath, 
        rm_filepath, 
        train_size, 
        test_size, 
        scale, 
        period='train',
        seed=2023,
    ):
        """Reads a single .csv
        """
        df = pd.read_csv(tm_filepath, header=None)
        df.drop(df.columns[-1], axis=1, inplace=True)
        traffic = df.values[:(train_size+test_size)] / scale

        if period == 'test':
            traffic = traffic[train_size:, :]
        else:
            traffic = traffic[:train_size, :]

        traffic = torch.from_numpy(traffic).float()
        rm_df = pd.read_csv(rm_filepath, header=None)
        rm_df.drop(rm_df.columns[-1], axis=1, inplace=True)

        rm = torch.from_numpy(rm_df.values).float()
        link = traffic @ rm

        return traffic, link, traffic.clone(), rm
    
    def __getitem__(self, ind):
        y = self.link[ind, :]
        x = self.traffic[ind, :]
        return x, y

    def __len__(self):
        return self.len
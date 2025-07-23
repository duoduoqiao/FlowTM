import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def load_data(args):
    # parse args
    data_root, dataset_name, batch_size, known_rate = args.data_root, args.dataset, args.batch_size, args.known_rate
    rm_filename = None

    if dataset_name == 'abilene':
        n_train = 15 * 7
        n_test = 1 * 7
        train_size = int(n_train * 288)
        test_size = int(n_test * 288)
        scale = 10 ** 9
        tm_filename = 'abilene_tm.csv'
        if rm_filename is None:
            rm_filename = 'abilene_rm.csv'
    elif dataset_name == 'geant':
        n_train = 10 * 7
        n_test = 1 * 7
        train_size = int(n_train * 96)
        test_size = int(n_test * 96)
        scale = 10 ** 7
        tm_filename = 'geant_tm.csv'
        if rm_filename is None:
            rm_filename = 'geant_rm.csv'
    else:
        raise NotImplementedError

    tm_filepath = os.path.join(data_root, tm_filename)
    rm_filepath = os.path.join(data_root, rm_filename)

    # build dataset
    traffic, link = read_data(tm_filepath, rm_filepath, train_size, test_size, scale)
    traffic_train, link_train, unknown_index_train = split_dataset(traffic, link, train_size, 'train', known_rate)
    traffic_test, link_test, unknown_index_test = split_dataset(traffic, link, train_size, 'test', known_rate)
    if args.use_conv3d:
        train_dataset = TMEDataset3D(traffic_train, link_train, unknown_index_train, period='train', window=args.window)
    else:
        train_dataset = TMEDataset(traffic_train, link_train, unknown_index_train, period='train')
    test_dataset = TMEDataset(traffic_test, link_test, unknown_index_test, period='test')

    # build dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def rand_mask(b, d, known_rate):
    known_num = int(b * d * known_rate)
    mask = torch.zeros((b, d), dtype=torch.float)
    known_index = torch.randperm(b * d)[:known_num]  # randomly select the known flow indices
    mask[known_index // d, known_index % d] = 1
    unknown_index = (mask == 0)
    return mask, unknown_index


def get_x_mask(x, mask, unknown_index):
    b, d = x.shape[0], x.shape[-1]
    x_mask = x * mask
    x_mean = torch.sum(x_mask, dim=0) / torch.clamp(torch.sum(mask, dim=0), min=1)
    x_mean = x_mean.repeat(b, 1)
    x_mask[unknown_index] = x_mean[unknown_index]
    return x_mask


def split_dataset(traffic, link, train_size, period, known_rate=1.):
    if period == 'test':
        traffic = traffic[train_size:, :]
        link = link[train_size:, :]
        mask = torch.ones_like(traffic)
        unknown_index = (mask == 0.)
    else:
        traffic = traffic[:train_size, :]
        link = link[:train_size, :]
        if known_rate < 1.:
            mask, unknown_index = rand_mask(traffic.shape[0], traffic.shape[1], known_rate)
            traffic = get_x_mask(traffic, mask, unknown_index)
        else:
            mask = torch.ones_like(traffic)
            unknown_index = (mask == 0.)
        print(f'known_rate = {known_rate}')
    return traffic, link, unknown_index


def read_data(
        tm_filepath,
        rm_filepath,
        train_size,
        test_size,
        scale
):
    # read a single csv
    df = pd.read_csv(tm_filepath, header=None)
    df.drop(df.columns[-1], axis=1, inplace=True)
    traffic = df.values[:(train_size + test_size)] / scale

    rm_df = pd.read_csv(rm_filepath, header=None)
    rm_df.drop(rm_df.columns[-1], axis=1, inplace=True)
    rm = rm_df.values

    traffic = torch.from_numpy(traffic).float()
    rm = torch.from_numpy(rm).float()
    link = traffic @ rm

    return traffic, link


class TMEDataset(Dataset):
    def __init__(
            self,
            traffic,
            link,
            unknown_index,
            period='train'
    ):
        super(TMEDataset, self).__init__()

        assert period in ['train', 'test'], ''
        self.period = period
        self.traffic, self.link, self.unknown_index = traffic, link, unknown_index

        self.dim_1 = self.link.shape[-1]
        self.len, self.dim_2 = self.traffic.shape[0], self.traffic.shape[-1]

    def __getitem__(self, ind):
        y = self.link[ind]
        x = self.traffic[ind]
        unknown_index = self.unknown_index[ind]
        return x, y, unknown_index

    def __len__(self):
        return self.len


class TMEDataset3D(Dataset):
    def __init__(
            self,
            traffic,
            link,
            unknown_index,
            period='train',
            window=12,
            stride=1
    ):
        super(TMEDataset3D, self).__init__()

        assert period in ['train', 'test'], ''
        self.period = period
        self.window = window
        self.stride = stride

        self.traffic, self.link, self.unknown_index = self.get_x_3d(traffic, link, unknown_index)

        self.dim_1 = self.link.shape[-1]
        self.len, self.dim_2 = self.traffic.shape[0], self.traffic.shape[-1]

    def get_x_3d(self, x, y, unknown_index):
        stride = self.stride if self.period == 'train' else self.window
        t = x.shape[0]

        # check
        if t < self.window:
            raise ValueError(f"The length of dataset {t} < the window size {self.window}.")
        if self.period == 'test' and t % self.window != 0:
            raise ValueError("The sequence length in the test phase must be divisible by the window size.")

        x_ = x.unfold(0, self.window, stride).contiguous().transpose(1, 2)  # (window_size, traffic_dim)
        unknown_index_ = unknown_index.unfold(0, self.window, stride).contiguous().transpose(1, 2)
        y_ = y.unfold(0, self.window, stride).contiguous().transpose(1, 2)

        return x_, y_, unknown_index_

    def __getitem__(self, ind):
        y = self.link[ind]
        x = self.traffic[ind]
        unknown_index = self.unknown_index[ind]
        return x, y, unknown_index

    def __len__(self):
        return self.len
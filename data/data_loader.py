import os
import warnings

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from utils.EMD import EMD_Reconstruct, EMD_Find_Freq, EMD_Predict
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
                 target='OT', ori_target='OT', EMD=True):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.EMD = EMD

        self.features = features
        self.target = target
        self.ori_target = ori_target
        self.target_index = 0
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.input_len, 12 * 30 * 24 + 4 * 30 * 24 - self.input_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove(self.ori_target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.ori_target] + cols]
        var_list = [self.ori_target] + cols
        self.target_index = var_list.index(self.target)
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.data = self.scaler.transform(df_value)
        self.data_x = self.data[border1:border2]

        if self.EMD:
            self.sin_waves = np.zeros([self.input_len + 1, self.input_len + self.pred_len])
            self.cos_waves = np.zeros([self.input_len + 1, self.input_len + self.pred_len])
            for i in range(self.input_len + 1):
                if i == 0:
                    continue
                for j in range(self.input_len + self.pred_len):
                    cur_period = self.input_len / i
                    self.sin_waves[i, j] = np.sin((2 * np.pi / cur_period) * j)
                    self.cos_waves[i, j] = np.cos((2 * np.pi / cur_period) * j)

            emd_dataset_path = './EMD/' + self.data_path[:-4]
            if not os.path.exists(emd_dataset_path):
                os.makedirs(emd_dataset_path)

            freq_path = ('./EMD/' + self.data_path[:-4] + '/' + '{}_{}_freq.npy'.
                         format(self.flag, self.input_len))
            reconw_path = ('./EMD/' + self.data_path[:-4] + '/' + '{}_{}_reconw.npy'.
                           format(self.flag, self.input_len))
            s_freq_path = ('./EMD/' + self.data_path[:-4] + '/' + '{}_{}_sfreq.npy'.
                           format(self.flag, self.input_len))

            if not os.path.exists(freq_path):
                self.emd_freq = np.zeros([len(self.data_x) - self.input_len, 40, self.data_x.shape[1]])
                self.emd_reconstructw = np.zeros([len(self.data_x) - self.input_len, 40, self.data_x.shape[1]])
                self.emd_sfreq = np.zeros([len(self.data_x) - self.input_len, self.data_x.shape[1]])
                print('Finding EMD-based freqs')
                for i in range(len(self.data_x) - self.input_len):
                    print('{}/{}'.format(i, len(self.data_x) - self.input_len))
                    for j in range(self.data_x.shape[1]):
                        local_freq, s_freq = EMD_Find_Freq(self.data_x[i: i + self.input_len, j], self.input_len)
                        self.emd_freq[i, :local_freq.shape[0], j] = local_freq
                        self.emd_sfreq[i, j] = s_freq
                        if local_freq.shape[0] > 0:
                            current_sin_waves = self.sin_waves[local_freq, :self.input_len]
                            current_cos_waves = self.cos_waves[local_freq, :self.input_len]
                            input_features = np.concatenate([current_sin_waves, current_cos_waves], axis=0)
                        else:
                            input_features = None
                        coef = EMD_Reconstruct(self.data_x[i: i + self.input_len, j], self.input_len, input_features)
                        self.emd_reconstructw[i, :coef.shape[0], j] = coef

                print('Saving EMD-based freqs ...')
                np.save(freq_path, self.emd_freq)
                np.save(reconw_path, self.emd_reconstructw)
                np.save(s_freq_path, self.emd_sfreq)
            else:
                self.emd_freq = np.load(freq_path)
                self.emd_reconstructw = np.load(reconw_path)
                self.emd_sfreq = np.load(s_freq_path)

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len + self.pred_len
        if self.features == 'M':
            seq_x = self.data_x[r_begin:r_end]
            pred_x_initial = np.zeros([self.input_len + self.pred_len, seq_x.shape[1]])
            for i in range(seq_x.shape[1]):
                pred_x_initial[:, i] = np.mean(seq_x[:self.input_len, i])

            var_sp_matrix = np.zeros([seq_x.shape[1], seq_x.shape[1]])
            if self.EMD:
                for i in range(seq_x.shape[1]):
                    current_freq = self.emd_freq[index, :, i]
                    current_weight = self.emd_reconstructw[index, :, i]
                    pred_x_initial[:, i] = EMD_Predict(
                        current_weight, self.input_len, self.pred_len, current_freq, self.sin_waves, self.cos_waves)
                if self.set_type == 0:
                    per = np.random.permutation(seq_x.shape[1])
                    ones_matrix = np.ones([1, seq_x.shape[1]])
                    zeros_matrix = np.zeros([1, seq_x.shape[1]])
                    var_speriod = self.emd_sfreq[index, per].reshape(1, seq_x.shape[1])
                    for i in range(seq_x.shape[1]):
                        shared_period = var_speriod[0, i]
                        var_sp_matrix[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)
                    return seq_x[:, per], pred_x_initial[-self.pred_len:, per], var_sp_matrix.astype(bool)
                else:
                    ones_matrix = np.ones([1, seq_x.shape[1]])
                    zeros_matrix = np.zeros([1, seq_x.shape[1]])
                    var_speriod = self.emd_sfreq[index, :].reshape(1, seq_x.shape[1])
                    for i in range(seq_x.shape[1]):
                        shared_period = var_speriod[0, i]
                        var_sp_matrix[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)
                    return seq_x, pred_x_initial[-self.pred_len:, :], var_sp_matrix.astype(bool)

            else:
                if self.set_type == 0:
                    per = np.random.permutation(seq_x.shape[1])
                    return seq_x[:, per], pred_x_initial[-self.pred_len:, :], var_sp_matrix.astype(bool)
                else:
                    return seq_x, pred_x_initial[-self.pred_len:, :], var_sp_matrix.astype(bool)
        else:
            seq_x = self.data_x[r_begin:r_end, self.target_index: self.target_index + 1]
            pred_x_initial = np.zeros([self.input_len + self.pred_len, seq_x.shape[1]])
            pred_x_initial[:, 0] = np.mean(seq_x[:self.input_len, 0])
            var_sp_matrix = np.zeros([seq_x.shape[1], seq_x.shape[1]])
            if self.EMD:
                current_freq = self.emd_freq[index, :, self.target_index]
                current_weight = self.emd_reconstructw[index, :, self.target_index]
                pred_x_initial[:, 0] = EMD_Predict(
                    current_weight, self.input_len, self.pred_len, current_freq, self.sin_waves, self.cos_waves)
                return seq_x, pred_x_initial[-self.pred_len:, :], var_sp_matrix.astype(bool)
            else:
                return seq_x, pred_x_initial[-self.pred_len:, :], var_sp_matrix.astype(bool)

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1


class Dataset_ETT_min(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv', target='OT', ori_target='OT', EMD=True):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.EMD = EMD

        self.features = features
        self.target = target
        self.ori_target = ori_target
        self.target_index = 0
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.input_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.input_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove(self.ori_target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.ori_target] + cols]
        var_list = [self.ori_target] + cols
        self.target_index = var_list.index(self.target)
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_value)

        self.data_x = data[border1:border2]

        if self.EMD:
            self.sin_waves = np.zeros([self.input_len + 1, self.input_len + self.pred_len])
            self.cos_waves = np.zeros([self.input_len + 1, self.input_len + self.pred_len])
            for i in range(self.input_len + 1):
                if i == 0:
                    continue
                for j in range(self.input_len + self.pred_len):
                    cur_period = self.input_len / i
                    self.sin_waves[i, j] = np.sin((2 * np.pi / cur_period) * j)
                    self.cos_waves[i, j] = np.cos((2 * np.pi / cur_period) * j)

            emd_dataset_path = './EMD/' + self.data_path[:-4]
            if not os.path.exists(emd_dataset_path):
                os.makedirs(emd_dataset_path)

            freq_path = ('./EMD/' + self.data_path[:-4] + '/' + '{}_{}_freq.npy'.
                         format(self.flag, self.input_len))
            reconw_path = ('./EMD/' + self.data_path[:-4] + '/' + '{}_{}_reconw.npy'.
                           format(self.flag, self.input_len))
            s_freq_path = ('./EMD/' + self.data_path[:-4] + '/' + '{}_{}_sfreq.npy'.
                           format(self.flag, self.input_len))

            if not os.path.exists(freq_path):
                self.emd_freq = np.zeros([len(self.data_x) - self.input_len, 40, self.data_x.shape[1]])
                self.emd_reconstructw = np.zeros([len(self.data_x) - self.input_len, 40, self.data_x.shape[1]])
                self.emd_sfreq = np.zeros([len(self.data_x) - self.input_len, self.data_x.shape[1]])
                print('Finding EMD-based freqs')
                for i in range(len(self.data_x) - self.input_len):
                    print('{}/{}'.format(i, len(self.data_x) - self.input_len))
                    for j in range(self.data_x.shape[1]):
                        local_freq, s_freq = EMD_Find_Freq(self.data_x[i: i + self.input_len, j], self.input_len)
                        self.emd_freq[i, :local_freq.shape[0], j] = local_freq
                        self.emd_sfreq[i, j] = s_freq
                        if local_freq.shape[0] > 0:
                            current_sin_waves = self.sin_waves[local_freq, :self.input_len]
                            current_cos_waves = self.cos_waves[local_freq, :self.input_len]
                            input_features = np.concatenate([current_sin_waves, current_cos_waves], axis=0)
                        else:
                            input_features = None
                        coef = EMD_Reconstruct(self.data_x[i: i + self.input_len, j], self.input_len, input_features)
                        self.emd_reconstructw[i, :coef.shape[0], j] = coef

                print('Saving EMD-based freqs ...')
                np.save(freq_path, self.emd_freq)
                np.save(reconw_path, self.emd_reconstructw)
                np.save(s_freq_path, self.emd_sfreq)
            else:
                self.emd_freq = np.load(freq_path)
                self.emd_reconstructw = np.load(reconw_path)
                self.emd_sfreq = np.load(s_freq_path)

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len + self.pred_len
        if self.features == 'M':
            seq_x = self.data_x[r_begin:r_end]
            pred_x_initial = np.zeros([self.input_len + self.pred_len, seq_x.shape[1]])
            for i in range(seq_x.shape[1]):
                pred_x_initial[:, i] = np.mean(seq_x[:self.input_len, i])

            var_sp_matrix = np.zeros([seq_x.shape[1], seq_x.shape[1]])
            if self.EMD:
                for i in range(seq_x.shape[1]):
                    current_freq = self.emd_freq[index, :, i]
                    current_weight = self.emd_reconstructw[index, :, i]
                    pred_x_initial[:, i] = EMD_Predict(
                        current_weight, self.input_len, self.pred_len, current_freq, self.sin_waves, self.cos_waves)
                if self.set_type == 0:
                    per = np.random.permutation(seq_x.shape[1])
                    ones_matrix = np.ones([1, seq_x.shape[1]])
                    zeros_matrix = np.zeros([1, seq_x.shape[1]])
                    var_speriod = self.emd_sfreq[index, per].reshape(1, seq_x.shape[1])
                    for i in range(seq_x.shape[1]):
                        shared_period = var_speriod[0, i]
                        var_sp_matrix[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)
                    return seq_x[:, per], pred_x_initial[-self.pred_len:, per], var_sp_matrix.astype(bool)
                else:
                    ones_matrix = np.ones([1, seq_x.shape[1]])
                    zeros_matrix = np.zeros([1, seq_x.shape[1]])
                    var_speriod = self.emd_sfreq[index, :].reshape(1, seq_x.shape[1])
                    for i in range(seq_x.shape[1]):
                        shared_period = var_speriod[0, i]
                        var_sp_matrix[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)
                    return seq_x, pred_x_initial[-self.pred_len:, :], var_sp_matrix.astype(bool)

            else:
                if self.set_type == 0:
                    per = np.random.permutation(seq_x.shape[1])
                    return seq_x[:, per], pred_x_initial[-self.pred_len:, :], var_sp_matrix.astype(bool)
                else:
                    return seq_x, pred_x_initial[-self.pred_len:, :], var_sp_matrix.astype(bool)
        else:
            seq_x = self.data_x[r_begin:r_end, self.target_index: self.target_index + 1]
            pred_x_initial = np.zeros([self.input_len + self.pred_len, seq_x.shape[1]])
            pred_x_initial[:, 0] = np.mean(seq_x[:self.input_len, 0])
            var_sp_matrix = np.zeros([seq_x.shape[1], seq_x.shape[1]])
            if self.EMD:
                current_freq = self.emd_freq[index, :, self.target_index]
                current_weight = self.emd_reconstructw[index, :, self.target_index]
                pred_x_initial[:, 0] = EMD_Predict(
                    current_weight, self.input_len, self.pred_len, current_freq, self.sin_waves, self.cos_waves)
                return seq_x, pred_x_initial[-self.pred_len:, :], var_sp_matrix.astype(bool)
            else:
                return seq_x, pred_x_initial[-self.pred_len:, :], var_sp_matrix.astype(bool)

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ECL.csv', target='MT_321', ori_target='MT_321', EMD=True):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.EMD = EMD

        self.data_path = data_path
        self.features = features
        self.target = target
        self.ori_target = ori_target
        self.target_index = 0
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.input_len, len(df_raw) - num_test - self.input_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove(self.ori_target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.ori_target] + cols]
        var_list = [self.ori_target] + cols
        self.target_index = var_list.index(self.target)
        if self.set_type == 0:
            print('Current target:', self.target)
            print('Current index:', self.target_index)
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_value)

        self.data_x = data[border1:border2]
        if self.EMD:
            self.sin_waves = np.zeros([self.input_len + 1, self.input_len + self.pred_len])
            self.cos_waves = np.zeros([self.input_len + 1, self.input_len + self.pred_len])
            for i in range(self.input_len + 1):
                if i == 0:
                    continue
                for j in range(self.input_len + self.pred_len):
                    cur_period = self.input_len / i
                    self.sin_waves[i, j] = np.sin((2 * np.pi / cur_period) * j)
                    self.cos_waves[i, j] = np.cos((2 * np.pi / cur_period) * j)

            emd_dataset_path = './EMD/' + self.data_path[:-4]
            if not os.path.exists(emd_dataset_path):
                os.makedirs(emd_dataset_path)

            freq_path = ('./EMD/' + self.data_path[:-4] + '/' + '{}_{}_freq.npy'.
                         format(self.flag, self.input_len))
            reconw_path = ('./EMD/' + self.data_path[:-4] + '/' + '{}_{}_reconw.npy'.
                           format(self.flag, self.input_len))
            s_freq_path = ('./EMD/' + self.data_path[:-4] + '/' + '{}_{}_sfreq.npy'.
                           format(self.flag, self.input_len))

            if not os.path.exists(freq_path):
                self.emd_freq = np.zeros([len(self.data_x) - self.input_len, 40, self.data_x.shape[1]])
                self.emd_reconstructw = np.zeros([len(self.data_x) - self.input_len, 40, self.data_x.shape[1]])
                self.emd_sfreq = np.zeros([len(self.data_x) - self.input_len, self.data_x.shape[1]])
                print('Finding EMD-based freqs')
                for i in range(len(self.data_x) - self.input_len):
                    print('{}/{}'.format(i, len(self.data_x) - self.input_len))
                    for j in range(self.data_x.shape[1]):
                        local_freq, s_freq = EMD_Find_Freq(self.data_x[i: i + self.input_len, j], self.input_len)
                        self.emd_freq[i, :local_freq.shape[0], j] = local_freq
                        self.emd_sfreq[i, j] = s_freq
                        if local_freq.shape[0] > 0:
                            current_sin_waves = self.sin_waves[local_freq, :self.input_len]
                            current_cos_waves = self.cos_waves[local_freq, :self.input_len]
                            input_features = np.concatenate([current_sin_waves, current_cos_waves], axis=0)
                        else:
                            input_features = None
                        coef = EMD_Reconstruct(self.data_x[i: i + self.input_len, j], self.input_len, input_features)
                        self.emd_reconstructw[i, :coef.shape[0], j] = coef

                print('Saving EMD-based freqs ...')
                np.save(freq_path, self.emd_freq)
                np.save(reconw_path, self.emd_reconstructw)
                np.save(s_freq_path, self.emd_sfreq)
            else:
                self.emd_freq = np.load(freq_path)
                self.emd_reconstructw = np.load(reconw_path)
                self.emd_sfreq = np.load(s_freq_path)

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len + self.pred_len
        if self.features == 'M':
            seq_x = self.data_x[r_begin:r_end]
            pred_x_initial = np.zeros([self.input_len + self.pred_len, seq_x.shape[1]])
            for i in range(seq_x.shape[1]):
                pred_x_initial[:, i] = np.mean(seq_x[:self.input_len, i])

            var_sp_matrix = np.zeros([seq_x.shape[1], seq_x.shape[1]])
            if self.EMD:
                for i in range(seq_x.shape[1]):
                    current_freq = self.emd_freq[index, :, i]
                    current_weight = self.emd_reconstructw[index, :, i]
                    pred_x_initial[:, i] = EMD_Predict(
                        current_weight, self.input_len, self.pred_len, current_freq, self.sin_waves, self.cos_waves)
                if self.set_type == 0:
                    per = np.random.permutation(seq_x.shape[1])
                    ones_matrix = np.ones([1, seq_x.shape[1]])
                    zeros_matrix = np.zeros([1, seq_x.shape[1]])
                    var_speriod = self.emd_sfreq[index, per].reshape(1, seq_x.shape[1])
                    for i in range(seq_x.shape[1]):
                        shared_period = var_speriod[0, i]
                        var_sp_matrix[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)
                    return seq_x[:, per], pred_x_initial[-self.pred_len:, per], var_sp_matrix.astype(bool)
                else:
                    ones_matrix = np.ones([1, seq_x.shape[1]])
                    zeros_matrix = np.zeros([1, seq_x.shape[1]])
                    var_speriod = self.emd_sfreq[index, :].reshape(1, seq_x.shape[1])
                    for i in range(seq_x.shape[1]):
                        shared_period = var_speriod[0, i]
                        var_sp_matrix[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)
                    return seq_x, pred_x_initial[-self.pred_len:, :], var_sp_matrix.astype(bool)

            else:
                if self.set_type == 0:
                    per = np.random.permutation(seq_x.shape[1])
                    return seq_x[:, per], pred_x_initial[-self.pred_len:, :], var_sp_matrix.astype(bool)
                else:
                    return seq_x, pred_x_initial[-self.pred_len:, :], var_sp_matrix.astype(bool)
        else:
            seq_x = self.data_x[r_begin:r_end, self.target_index: self.target_index + 1]
            pred_x_initial = np.zeros([self.input_len + self.pred_len, seq_x.shape[1]])
            pred_x_initial[:, 0] = np.mean(seq_x[:self.input_len, 0])
            var_sp_matrix = np.zeros([seq_x.shape[1], seq_x.shape[1]])
            if self.EMD:
                current_freq = self.emd_freq[index, :, self.target_index]
                current_weight = self.emd_reconstructw[index, :, self.target_index]
                pred_x_initial[:, 0] = EMD_Predict(
                    current_weight, self.input_len, self.pred_len, current_freq, self.sin_waves, self.cos_waves)
                return seq_x, pred_x_initial[-self.pred_len:, :], var_sp_matrix.astype(bool)
            else:
                return seq_x, pred_x_initial[-self.pred_len:, :], var_sp_matrix.astype(bool)

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1

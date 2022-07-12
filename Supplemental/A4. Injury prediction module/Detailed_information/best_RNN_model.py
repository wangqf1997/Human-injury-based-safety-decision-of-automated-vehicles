# -*- coding: utf-8 -*-
'''
-------------------------------------------------------------------------------------------------
This code accompanies the paper titled "Human injury-based safety decision of automated vehicles"
Author: Qingfan Wang, ***,
Corresponding author: Bingbing Nie (nbb@tsinghua.edu.cn)
-------------------------------------------------------------------------------------------------
'''


import argparse
import random
import numpy as np

import torch
from torch import nn
import torch.utils.data as Data

from imblearn.metrics import geometric_mean_score
from imblearn.metrics import classification_report_imbalanced

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


__author__ = "Qingfan Wang"


class RNN(nn.Module):
    ''' RNN-based injury prediction model. '''

    def __init__(self, in_dim, hid_dim, n_layers, flag_LSTM, bidirectional, dropout):
        super(RNN, self).__init__()
        ''' Develop the model structure. '''

        # Develop embedding layers for eight input valuables.
        # The size of embedding tables depends on the discretization of valuables.
        self.embed_1 = nn.Linear(1, in_dim)  # collision delta-v
        self.embed_2 = nn.Linear(1, in_dim)  # collision angle
        self.embed_3 = nn.Embedding(22, in_dim)  # POI of the ego vehicle
        self.embed_4 = nn.Embedding(22, in_dim)  # POI of the opposing vehicle
        self.embed_5 = nn.Embedding(4, in_dim)  # occupant age
        self.embed_6 = nn.Embedding(2, in_dim)  # occupant gender
        self.embed_7 = nn.Embedding(2, in_dim)  # belt usage
        self.embed_8 = nn.Embedding(2, in_dim)  # airbag usage
        self.embed_9 = nn.Embedding(5, in_dim)  # vehicle mass ratio

        # Determine the usage of LSTM or GRU.
        self.flag_LSTM = flag_LSTM

        # The key data processing module.
        if flag_LSTM:
            self.encoder = nn.LSTM(in_dim, hid_dim, n_layers, batch_first=True, bidirectional=bidirectional,
                                   dropout=dropout)
        else:
            self.encoder = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, bidirectional=bidirectional,
                                  dropout=dropout)

        # Determine the usage of bidirectional mechanism.
        if bidirectional:
            # Map the outputs to the four defined injury levels.
            self.linear = nn.Linear(2 * hid_dim, 4)
        else:
            # Map the outputs to the four defined injury levels.
            self.linear = nn.Linear(hid_dim, 4)

    def forward(self, x):
        ''' The forward propagation of information. '''

        # Embedding the nine input valuables into a high-dimensional space based on embedding lookup tables.
        e_1 = self.embed_1(x[:, 0].unsqueeze(1)).unsqueeze(1)
        e_2 = self.embed_2(x[:, 1].unsqueeze(1)).unsqueeze(1)
        e_3 = self.embed_3(x[:, 2].long()).unsqueeze(1)
        e_4 = self.embed_4(x[:, 3].long()).unsqueeze(1)
        e_5 = self.embed_5(x[:, 4].long()).unsqueeze(1)
        e_6 = self.embed_6(x[:, 5].long()).unsqueeze(1)
        e_7 = self.embed_7(x[:, 6].long()).unsqueeze(1)
        e_8 = self.embed_8(x[:, 7].long()).unsqueeze(1)
        e_9 = self.embed_9(x[:, 8].long()).unsqueeze(1)

        # Concatenate the nine Embedded input valuables in the high-dimensional space.
        z = torch.cat([e_1, e_2, e_3, e_4, e_5, e_6, e_7, e_8, e_9], dim=1)

        # Process the input data with RNN.
        if self.flag_LSTM:
            output, (hidden, cell) = self.encoder(z)
        else:
            output, _ = self.encoder(z)

        # Map the outputs to the four defined injury levels.
        outputs = self.linear(output[:, -1, :])

        return outputs


def load_data(data):
    ''' Load and process the crash data. '''

    # Divide the dataset into three parts: training, validation, and testing.
    shuffle = np.random.permutation(len(data))
    data = data[shuffle]
    data_train = data[:int(len(data) * 0.7)]
    data_test = data[int(len(data) * 0.7):int(len(data) * 0.85)]
    data_val = data[int(len(data) * 0.85):]

    # tranform the data into torch.
    data_val = torch.from_numpy(data_val).float()
    data_test = torch.from_numpy(data_test).float()

    return data_val, data_test


def evaluate_model(model, loader_eva, Num_data):
    ''' Evaluate the model. '''

    # Switch to the evaluation mode.
    model.eval()

    # Evaluation batch.
    test_true = np.zeros(Num_data)
    test_pred = np.zeros(Num_data)
    for step, (batch_x,) in enumerate(loader_eva):
        prediction = model(batch_x[:, :-1])

        # Record the target and predicted injuries.
        if step < len(loader_eva) - 1:
            test_true[(step) * 256:(step + 1) * 256] = batch_x[:, -1].cpu().numpy()
            test_pred[(step) * 256:(step + 1) * 256] = prediction.data.max(1)[1].cpu().numpy()
        else:
            test_true[(step) * 256:] = batch_x[:, -1].cpu().numpy()
            test_pred[(step) * 256:] = prediction.data.max(1)[1].cpu().numpy()

    # Obtain the prediction performance.
    accu = 100. * (1 - np.count_nonzero(test_true - test_pred) / float(len(test_pred)))
    G_mean = geometric_mean_score(test_true, test_pred)
    conf_mat = confusion_matrix(test_true, test_pred)
    report = classification_report_imbalanced(test_true, test_pred, digits=3)

    return np.around(accu, 1), np.around(G_mean, 3), conf_mat, report


def main():
    ''' Show the optimal RNN-based occupant injury prediction model with the best prediction performance. '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--rand_seed', type=int, default=123, help='Random seed')
    parser.add_argument('--re_samp', type=str, default='US', help='Re-sampling methods: US, OS, CS')
    parser.add_argument('--print_inf', action='store_false', help='print the information of the training process')
    opt = parser.parse_args()

    # Define the random seed.
    seed = opt.rand_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Load the real-world crash data.
    data = np.load('dataset/data_pro.npy')
    data_val, data_test = load_data(data)

    # Load the datasets.
    dataset_val = Data.TensorDataset(data_val)
    loader_val = Data.DataLoader(dataset=dataset_val, batch_size=256, shuffle=True)
    dataset_test = Data.TensorDataset(data_test)
    loader_test = Data.DataLoader(dataset=dataset_test, batch_size=256, shuffle=True)

    # Obtain the optimal hyper-parameters.
    if opt.re_samp == 'US':
        flag, bidirectional, Batch_size, In_dim, Hid_dim, Layer, dropout, Learning_rate = True, False, 64, 32, 64, 1, 0.5, 0.003
    elif opt.re_samp == 'OS':
        flag, bidirectional, Batch_size, In_dim, Hid_dim, Layer, dropout, Learning_rate = True, True, 64, 16, 32, 2, 0.5, 0.003
    elif opt.re_samp == 'CS':
        flag, bidirectional, Batch_size, In_dim, Hid_dim, Layer, dropout, Learning_rate = False, False, 256, 32, 64, 2, 0.2, 0.001
    else:
        print('Wrong re-sampling method!')
        return

    # Load the model with the optimal parameters.
    model = RNN(in_dim=In_dim, hid_dim=Hid_dim, n_layers=Layer, flag_LSTM=flag, bidirectional=bidirectional,
                dropout=dropout)
    # model.load_state_dict(torch.load('Saved_Model_params\Best_params\RNN_%s_best_params.pkl' % opt.re_samp))
    model.load_state_dict(torch.load('Saved_Model_params\Best_params\RNN_%s_best_params.pkl' % opt.re_samp, map_location='cpu'))

    # Validate the model.
    accu, G_mean, conf_mat, report = evaluate_model(model, loader_val, len(data_val))
    if opt.print_inf:
        print('Validation | Accuracy: ' + str(accu) + '%')
        print('Validation | G-mean: ' + str(G_mean))
        print(conf_mat)
        print(report)

    # Test the model.
    accu, G_mean, conf_mat, report = evaluate_model(model, loader_test, len(data_test))
    if opt.print_inf:
        print('Test | Accuracy: ' + str(accu) + '%')
        print('Test | G-mean: ' + str(G_mean))
        print(conf_mat)
        print(report)


if __name__ == "__main__":
    main()

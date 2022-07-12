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


class FNN(nn.Module):
    ''' Develop FNN-based occupant injury prediction model. '''

    def __init__(self, in_dim, n_hidden, dropout, flag):
        super(FNN, self).__init__()
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

        self.flag = flag

        # The key data processing module.
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(), nn.Dropout(dropout))
        if flag:
            self.layer4 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(), nn.Dropout(dropout))
            self.layer5 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(), nn.Dropout(dropout))
            self.layer6 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(), nn.Dropout(dropout))

        # Map the outputs to the four defined injury levels.
        self.layer = nn.Sequential(nn.Linear(n_hidden, 4))

    def forward(self, x):
        ''' The forward propagation of information. '''

        # Embedding the nine input valuables into a high-dimensional space based on embedding lookup tables.
        e_1 = self.embed_1(x[:, 0].unsqueeze(1))
        e_2 = self.embed_2(x[:, 1].unsqueeze(1))
        e_3 = self.embed_3(x[:, 2].long())
        e_4 = self.embed_4(x[:, 3].long())
        e_5 = self.embed_5(x[:, 4].long())
        e_6 = self.embed_6(x[:, 5].long())
        e_7 = self.embed_7(x[:, 6].long())
        e_8 = self.embed_8(x[:, 7].long())
        e_9 = self.embed_9(x[:, 8].long())

        # Concatenate the nine Embedded input valuables in the high-dimensional space.
        z = e_1 + e_2 + e_3 + e_4 + e_5 + e_6 + e_7 + e_8 + e_9

        # Process the input data with FNN.
        z = self.layer3(self.layer2(self.layer1(z)))
        if self.flag:
            z = self.layer6(self.layer5(self.layer4(z)))

        # Map the outputs to the four defined injury levels.
        outputs = self.layer(z)

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
    ''' Show the optimal FNN-based occupant injury prediction model with the best prediction performance. '''

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
        flag, Batch_size, In_dim, Hid_dim, dropout, Learning_rate = False, 64, 32, 64, 0.2, 0.01
    elif opt.re_samp == 'OS':
        flag, Batch_size, In_dim, Hid_dim, dropout, Learning_rate = False, 32, 64, 128, 0.5, 0.01
    elif opt.re_samp == 'CS':
        flag, Batch_size, In_dim, Hid_dim, dropout, Learning_rate = False, 32, 128, 128, 0.2, 0.003
    else:
        print('Wrong re-sampling method!')
        return

    # Load the model with the optimal parameters.
    model = FNN(in_dim=In_dim, n_hidden=Hid_dim, dropout=dropout, flag=flag)
    # model.load_state_dict(torch.load('Saved_Model_params\Best_params\FNN_%s_best_params.pkl' % opt.re_samp))
    model.load_state_dict(torch.load('Saved_Model_params\Best_params\FNN_%s_best_params.pkl' % opt.re_samp, map_location='cpu'))

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

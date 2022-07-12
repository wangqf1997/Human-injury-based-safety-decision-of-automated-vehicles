'''
-------------------------------------------------------------------------------------------------
This code accompanies the paper titled "Human injury-based safety decision of automated vehicles"
Author: Qingfan Wang, Qing Zhou, Miao Lin, Bingbing Nie
Corresponding author: Bingbing Nie (nbb@tsinghua.edu.cn)
-------------------------------------------------------------------------------------------------
'''


import torch
import numpy as np
from torch import nn
from torch.nn.utils import weight_norm


__author__ = "Qingfan Wang"


class RNN(nn.Module):
    ''' Establish the occupant injury prediction model. '''

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

        # Concatenate the eight Embedded input valuables in the high-dimensional space.
        z = torch.cat([e_1, e_2, e_3, e_4, e_5, e_6, e_7, e_8, e_9], dim=1)

        # Process the input data with RNN.
        if self.flag_LSTM:
            output, (hidden, cell) = self.encoder(z)
        else:
            output, _ = self.encoder(z)

        # Map the outputs to the four defined injury levels.
        outputs = self.linear(output[:, -1, :])

        return outputs

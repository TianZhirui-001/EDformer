import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import TensorDataset
from tqdm import tqdm
import os
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_data = TensorDataset(input_train1, output_train1,input_train_extend1,
                           input_train2, output_train2,input_train_extend2,
                           input_train3, output_train3,input_train_extend3,
                           input_train4, output_train4,input_train_extend4,
                           input_train5, output_train5,input_train_extend5,
                           input_train6, output_train6,input_train_extend6)

train_loader = torch.utils.data.DataLoader(train_data,
                                           bath_size,
                                           False)


class Encoder1(nn.Module):
    def __init__(self, time_step_feature):
        super(Encoder1, self).__init__()
        self.conv1d1 = nn.Conv1d(time_step_feature, out_channels=32, kernel_size=2, padding=1)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.conv1d2 = nn.Conv1d(32, out_channels=64, kernel_size=2, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=16)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)

    def forward(self, x):
        x_c = x.permute(0, 2, 1)
        x_c = self.conv1d1(x_c)
        x_c = self.relu1(x_c)
        x_c = self.pool1(x_c)
        x_c = self.conv1d2(x_c)
        x_c = self.relu2(x_c)
        x_c = self.pool2(x_c)
        x_c = x_c.permute(0, 2, 1)
        x = x_c.permute(1, 0, 2)
        x_encoder1 = self.transformer_encoder(x)
        return x_encoder1



class Encoder2(nn.Module):
    def __init__(self, time_step_feature):
        super(Encoder2, self).__init__()
        self.fc1 = nn.Linear(time_step_feature, 128)
        self.lstm1 = nn.LSTM(128, 128, bidirectional=True, num_layers=1, batch_first=True)


    def forward(self, x):
        x = self.fc1(x)
        output, (h_n, c_n) = self.lstm1(x)
        h_n = h_n[0:1]
        c_n = c_n[0:1]
        return output, h_n, c_n

class Decoder(nn.Module):
    def __init__(self, feature_size, output_size):
        super(Decoder, self).__init__()
        ##Add any attention here based on yourself
        self.fc = nn.Linear(feature_size*2+ n(based on ICEMDAN), 64)
        self.lstm1 = nn.LSTM(64, 64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(64, 64, num_layers=1, batch_first=True)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=64, nhead=16)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=1)
        self.dropout3 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x_extend, x_new, h_0, c_0,x_old):
        x = torch.cat((x_extend,x_new), dim=2)
        x = self.fc(x)
        x, _ = self.lstm1(x, (h_0, c_0))
        x,_ = self.lstm2(x)
        x,_ = self.lstm3(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_decoder(x,x_old)
        x = x.permute(1, 0, 2)
        output = self.fc2(x)
        return output[:, -1, :]

class EncoDeco(nn.Module):
    def __init__(self, time_step_feature, feature_size, output_size):
        super(EncoDeco, self).__init__()
        self.encoder1 = Encoder1(time_step_feature)
        self.encoder2 = Encoder2(time_step_feature)
        self.decoder = Decoder(feature_size, output_size)

    def forward(self, x,t):
        x_encoder1 = self.encoder1(x)
        x_encoder2, h_n, c_n = self.encoder2(x)
        output = self.decoder(t,x_encoder2,h_n, c_n,x_encoder1)
        return output


class Multitask(nn.Module):
    def __init__(self, time_step_feature, output_size):
        super(Multitask, self).__init__()
        self.EncoDeco1 = EncoDeco(time_step_feature, output_size)
        self.EncoDeco2 = EncoDeco(time_step_feature, output_size)
        self.EncoDeco3 = EncoDeco(time_step_feature, output_size)
        self.EncoDeco4 = EncoDeco(time_step_feature, output_size)
        self.EncoDeco5 = EncoDeco(time_step_feature, output_size)
        self.EncoDeco6 = EncoDeco(time_step_feature, output_size)

    def forward(self, x1,t1,x2,t2,x3,t3,x4,t4,x5,t5,x6,t6):
        output1 = self.EncoDeco1(x1,t1)
        output2 = self.EncoDeco2(x2, t2)
        output3 = self.EncoDeco3(x3, t3)
        output4 = self.EncoDeco4(x4, t4)
        output5 = self.EncoDeco5(x5, t5)
        output6 = self.EncoDeco6(x6, t6)
        return output1,output2,output3,output4,output5,output6

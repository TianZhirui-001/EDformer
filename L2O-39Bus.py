import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import TensorDataset
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_sliding_windows(data, window_size=24):
    windows = []

    for start in range(len(data) - window_size + 1):
        end = start + window_size
        window = data[start:end]  # Directly use the slice of numpy array
        windows.append(window)

    return np.array(windows)

feature = 42
bath_size = 32
max_epoch = 500
num_test = 100


data = pd.read_excel('')
data = np.array(data)
Data1 = data[:,:feature]
Data2 = data[:,feature].reshape(-1,1)
Data1 = create_sliding_windows(Data1, window_size=24)
Data2 = Data2[~np.isnan(Data2).any(axis=1)]
Data3 = Data2

train_size = len(Data1) - num_test
input_train, output_train = Data1[:train_size], Data2[:train_size]
input_test, output_test = Data1[train_size:], Data2[train_size:]

A = MinMaxScaler()
A.fit(output_train)
output_train = A.transform(output_train)


input_train = torch.from_numpy(input_train).to(torch.float32)
output_train = torch.from_numpy(output_train).to(torch.float32)
input_test = torch.from_numpy(input_test).to(torch.float32)
output_test = torch.from_numpy(output_test).to(torch.float32)


train_data = TensorDataset(input_train, output_train)
train_loader = torch.utils.data.DataLoader(train_data,
                                           bath_size,
                                           True)

class DeepLearing(nn.Module):
    def __init__(self, feature, output_size):
        super().__init__()
        self.fc1 = nn.Linear(feature, 2048)
        self.silu1 = nn.SiLU()
        self.drop1 = nn.Dropout(0.05)
        self.fc2 = nn.Linear(2048, 2048)
        self.silu2 = nn.SiLU()
        self.drop2 = nn.Dropout(0.05)
        self.fc3 = nn.Linear(2048, 1)

        self.fc4 = nn.Linear(24, 2048)
        self.silu3 = nn.SiLU()
        self.drop3 = nn.Dropout(0.05)
        self.fc5 = nn.Linear(2048, 2048)
        self.silu4 = nn.SiLU()
        self.drop4 = nn.Dropout(0.05)
        self.fc6 = nn.Linear(2048, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.silu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.silu2(x)
        x = self.drop2(x)
        x = self.fc3(x)

        x = x.permute(0, 2, 1)
        x = self.fc4(x)
        x = self.silu3(x)
        x = self.drop3(x)
        x = self.fc5(x)
        x = self.silu4(x)
        x = self.drop4(x)
        x = self.fc6(x)
        x = x.squeeze(2)
        return x

model = DeepLearing(feature,1).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.000001,weight_decay=0.00001)


for epoch in range(max_epoch):
    model.train()
    running_loss = 0
    train_bar = tqdm(train_loader)  # 形成进度条
    for data in train_bar:
        input_train, output_train = data  # 解包迭代器中的X和Y
        input_train = input_train.to(device)  # 将输入数据移动到GPU上
        output_train = output_train.to(device)  # 将目标输出移动到GPU上
        optimizer.zero_grad()
        input_train_pred = model(input_train)
        loss = loss_function(input_train_pred, output_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 max_epoch,loss)


## Testing
model.eval()
input_test = input_test.to(device)
output_test_pred = model(input_test)
output_test_pred = output_test_pred.cpu().detach().numpy()
output_test_pred = A.inverse_transform(output_test_pred)


Real= Data3[-num_test:].reshape(-1,1)
output_test_pred = output_test_pred[-num_test:].reshape(-1,1)

plt.figure()
plt.plot(Real)
plt.plot(output_test_pred)
plt.show()

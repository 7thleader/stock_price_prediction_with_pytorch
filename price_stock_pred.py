import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ticker = "XU100.IS"
df = yf.download(ticker, "2020-01-01")
df

df.Close.plot(figsize=(12,8))

scaler = StandardScaler()
df["Close"] = scaler.fit_transform(df[["Close"]])
df.Close

seq_length = 30
data = []

for i in range(len(df) - seq_length):
    data.append(df.Close[i:i+seq_length])

data= np.array(data).reshape(-1,seq_length,1)

train_size = int(0.8 * len(data))

x_train = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(device)
y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
x_test = torch.from_numpy(data[train_size:, :-1, :]).type(torch.Tensor).to(device)
y_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(device)

class PredictionModel(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,output_dim):
        super(PredictionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.lstm = nn.LSTM(input_dim,hidden_dim,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim,output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_dim,device=device)
        c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_dim,device=device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out
        

model = PredictionModel(input_dim=1,hidden_dim=32,num_layers=2,output_dim=1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

num_epochs = 200
for i in range(num_epochs):
    y_train_pred = model(x_train)
    
    loss = criterion(y_train_pred,y_train)

    if i % 25 == 0:
        print(i,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


model.eval()

with torch.no_grad():
    y_train_pred = model(x_train)
    y_test_pred = model(x_test)

y_train_pred_inv = scaler.inverse_transform(y_train_pred.cpu().numpy())
y_train_inv = scaler.inverse_transform(y_train.cpu().numpy())

y_test_pred_inv = scaler.inverse_transform(y_test_pred.cpu().numpy())
y_test_inv = scaler.inverse_transform(y_test.cpu().numpy())


train_rmse = root_mean_squared_error(y_train_inv[:, 0], y_train_pred_inv[:, 0])
test_rmse = root_mean_squared_error(y_test_inv[:, 0], y_test_pred_inv[:, 0])

fig = plt.figure(figsize=(12,10))

gs = fig.add_gridspec(4,1)


ax1 = fig.add_subplot(gs[:3,0])
ax1.plot(df.iloc[-len(y_test_inv):].index, y_test_inv, color="blue", label="Actual Price")
ax1.plot(df.iloc[-len(y_test_inv):].index, y_test_pred_inv, color="green", label="Predicted Price")
ax1.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")

ax2 = fig.add_subplot(gs[3,0])
ax2.axhline(test_rmse, color="blue", linestyle="--", label="RMSE")

ax2.plot(df.iloc[-len(y_test_inv):].index, abs(y_test_inv - y_test_pred_inv), "r", label="Prediction Error")
ax2.legend()
plt.title("Prediction Error")
plt.xlabel("Date")
plt.ylabel("Error")
plt.tight_layout()
plt.show()
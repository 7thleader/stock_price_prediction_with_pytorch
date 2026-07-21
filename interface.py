import streamlit as st
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

class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


st.title("📈 Yapay Zeka ile Hisse Tahminleme")
st.write("Bu uygulama LSTM modeli kullanarak geçmiş fiyatlardan geleceği tahmin eder.")


ticker = st.text_input("Hisse/Endeks Kodu Girin (Örn: AAPL, XU100.IS):", "AAPL")
num_epochs = st.slider("Eğitim Turu (Epoch) Seçin:", min_value=50, max_value=300, value=150, step=50)


if st.button("Modeli Eğit ve Tahmin Et"):
    
    with st.spinner(f"{ticker} verileri işleniyor ve model eğitiliyor. Lütfen bekleyin..."):
        
        
        df = yf.download(ticker, "2020-01-01")
        
        if len(df) == 0:
            st.error("Bu hisse koduyla veri bulunamadı! Lütfen kodu kontrol edin.")
        else:
            
            seq_length = 30
            total_windows = len(df) - seq_length
            train_size = int(0.8 * total_windows)
            split_index = train_size + seq_length
            
            scaler = StandardScaler()
            scaler.fit(df[["Close"]].iloc[:split_index])
            df["Close"] = scaler.transform(df[["Close"]])
            
            data = []
            for i in range(len(df) - seq_length):
                data.append(df["Close"].iloc[i:i+seq_length].values)
            
            data = np.array(data).reshape(-1, seq_length, 1)
            
            x_train = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(device)
            y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
            x_test = torch.from_numpy(data[train_size:, :-1, :]).type(torch.Tensor).to(device)
            y_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(device)

           
            model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            
            progress_bar = st.progress(0)
            
            for i in range(num_epochs):
                y_train_pred = model(x_train)
                loss = criterion(y_train_pred, y_train)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
                progress_bar.progress((i + 1) / num_epochs)

            
            model.eval()
            with torch.no_grad():
                y_train_pred = model(x_train)
                y_test_pred = model(x_test)

            y_test_pred_inv = scaler.inverse_transform(y_test_pred.cpu().numpy())
            y_test_inv = scaler.inverse_transform(y_test.cpu().numpy())

            test_rmse = root_mean_squared_error(y_test_inv[:, 0], y_test_pred_inv[:, 0])

            
            st.success(f"Eğitim Tamamlandı! Hata Payı (Test RMSE): {test_rmse:.2f}")

            
            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(4, 1)

            ax1 = fig.add_subplot(gs[:3, 0])
            ax1.plot(df.iloc[-len(y_test_inv):].index, y_test_inv, color="blue", label="Gerçek Fiyat")
            ax1.plot(df.iloc[-len(y_test_inv):].index, y_test_pred_inv, color="green", label="Tahmin Edilen")
            ax1.legend()
            ax1.set_title(f"{ticker} Hisse Fiyat Tahmini")
            ax1.set_xlabel("Tarih")
            ax1.set_ylabel("Fiyat")

            ax2 = fig.add_subplot(gs[3, 0])
            ax2.axhline(test_rmse, color="blue", linestyle="--", label="RMSE")
            ax2.plot(df.iloc[-len(y_test_inv):].index, abs(y_test_inv - y_test_pred_inv), "r", label="Tahmin Hatası")
            ax2.legend()
            ax2.set_title("Tahmin Hatası Payı")
            ax2.set_xlabel("Tarih")
            ax2.set_ylabel("Hata")

            plt.tight_layout()
            
            
            st.pyplot(fig)
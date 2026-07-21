# stock_price_prediction_with_pytorch
An end-to-end Deep Learning project for financial time series forecasting using PyTorch (LSTM) and Streamlit.

[TR]
# 📈 Deep Learning Tabanlı Finansal Zaman Serisi Tahminlemesi (PyTorch LSTM)

Bu proje, finansal piyasalardaki (hisse senetleri, endeksler) tarihsel fiyat hareketlerini analiz ederek gelecekteki kapanış fiyatlarını tahmin etmek amacıyla geliştirilmiş uçtan uca bir Derin Öğrenme (Deep Learning) uygulamasıdır. 

Projede yalnızca bir model eğitilmekle kalınmamış; zaman serisi analizlerindeki en büyük risklerden biri olan **veri sızıntısı (data leakage)** engellenmiş, tensör boyutlandırmaları optimize edilmiş ve model interaktif bir web arayüzü ile canlıya alınmıştır.

## 🧠 Teknik Mimari ve Geliştirme Süreci

### 1. Veri Hazırlığı ve İstatistiksel Önlemler
Zaman serisi projelerinde verinin modele doğru formatta ve doğru istatistiksel varsayımlarla verilmesi, modelin kendisi kadar kritiktir.
* **Veri Çekimi:** `yfinance` kullanılarak istenen finansal varlığın 2020'den günümüze kadarki günlük kapanış (Close) fiyatları çekildi.
* **Data Leakage (Veri Sızıntısı) Çözümü:** `StandardScaler` işlemi uygulanırken, modelin gelecekten "kopya çekmesini" (look-ahead bias) engellemek adına kritik bir yaklaşım izlendi. Scaler tüm veri setine değil, **yalnızca eğitim (train) setine fit edildi** ve ardından tüm veri setine uygulandı.
* **Sliding Window (Kaydırılan Pencere) Yaklaşımı:** 1 boyutlu zaman serisi verisi, modelin geçmişi öğrenebilmesi için 30 günlük pencerelere (sequence_length=30) bölündü.
* **Tensör Dönüşümleri:** PyTorch'un LSTM mimarisinin beklediği `(Batch Size, Sequence Length, Input Dimension)` formatını sağlamak amacıyla Numpy dizileri `.reshape(-1, 30, 1)` ile 3 boyutlu matrislere dönüştürülerek GPU/CPU işlem birimlerine (tensör olarak) aktarıldı.

### 2. PyTorch LSTM Model Mimarisi
PyTorch `nn.Module` sınıfından türetilen özel bir sinir ağı mimarisi inşa edildi.
* **LSTM Katmanı:** Zaman içindeki uzun ve kısa vadeli bağımlılıkları (trendleri) yakalamak için `hidden_dim=32` ve `num_layers=2` parametrelerine sahip bir LSTM ağı kullanıldı.
* **Tam Bağlı (Fully Connected) Katman:** LSTM'den çıkan matrisin sadece en son zaman adımındaki (`out[:, -1, :]`) özellikler alınarak, tek boyutlu bir çıktı (tahmin edilen fiyat) üretecek bir `nn.Linear` katmanından geçirildi.

### 3. Eğitim Döngüsü (Training Loop) ve Optimizasyon
* **Kayıp Fonksiyonu:** Sürekli bir değişken tahmini (regresyon) yapıldığı için `nn.MSELoss` (Ortalama Karesel Hata) kullanıldı.
* **Optimizasyon:** Gradyan inişi (gradient descent) için esnek öğrenme oranına sahip `Adam Optimizer` (lr=0.01) tercih edildi.
* **Geri Yayılım (Backpropagation):** Her epoch'ta `optimizer.zero_grad()`, `loss.backward()` ve `optimizer.step()` işlemleriyle model ağırlıkları güncellendi.

### 4. Değerlendirme (Evaluation) ve Gerçek Dünya Metrikleri
* **Evaluation Modu:** Tahmin aşamasında ağın ağırlıklarının güncellenmesini durdurmak için `model.eval()` ve gereksiz bellek kullanımını önlemek için `with torch.no_grad():` bloğu kullanıldı.
* **Inverse Transform:** Modelin ürettiği ölçeklendirilmiş (-1 ile 1 arasındaki) tahminler, `scaler.inverse_transform()` ile tekrar gerçek dünya fiyatlarına dönüştürüldü.
* **Performans Ölçümü:** Başarı oranı, orijinal fiyat ölçeğinde Kök Ortalama Karesel Hata (RMSE) kullanılarak hesaplandı ve görselleştirildi.

### 5. Streamlit ile Arayüz Tasarımı ve Deployment
Geliştirilen bu arka plan mimarisi, teknik olmayan kullanıcıların da deneyimleyebilmesi için Streamlit kullanılarak interaktif bir web uygulamasına dönüştürüldü.
* Kullanıcıdan dinamik olarak hisse kodu (Örn: AAPL, XU100.IS, VOO) ve eğitim turu (Epoch) sayısı alınır.
* Arayüz üzerinde model anlık olarak eğitilir ve süreç bir "Progress Bar" ile gösterilir.
* Eğitim sonucunda Matplotlib ile gerçek vs. tahmin edilen fiyatlar grafiksel olarak ekrana yansıtılır.
* Uygulama, `requirements.txt` konfigürasyonu ile Streamlit Community Cloud üzerinde canlıya (deploy) alınmıştır.

## ⚙️ Kurulum ve Lokal Çalıştırma

Projeyi kendi ortamınızda test etmek için:

1. Depoyu klonlayın:
   ```bash
   git clone [https://github.com/KULLANICI_ADINIZ/DEPO_ADINIZ.git](https://github.com/KULLANICI_ADINIZ/DEPO_ADINIZ.git)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
[EN]
# 📈 Deep Learning-Based Financial Time Series Forecasting (PyTorch LSTM)

This project is an end-to-end Deep Learning application developed to predict future closing prices by analyzing historical price movements in financial markets (stocks, indices). 

The project goes beyond merely training a model; it addresses major risks in time series analysis by preventing **data leakage**, optimizing tensor dimensions, and deploying the model live with an interactive web interface.

## 🧠 Technical Architecture and Development Process

### 1. Data Preparation and Statistical Precautions
Providing data in the correct format and with proper statistical assumptions is as critical as the model itself in time series projects.
* **Data Extraction:** Daily closing (Close) prices of the desired financial asset from 2020 to the present were extracted using `yfinance`.
* **Data Leakage Solution:** A critical approach was taken to prevent the model from "cheating" by looking ahead (look-ahead bias) during the `StandardScaler` application. The scaler was **fitted solely on the training set** rather than the entire dataset, and then applied to the whole dataset.
* **Sliding Window Approach:** The 1D time series data was divided into 30-day windows (`sequence_length=30`) so the model could learn from past sequences.
* **Tensor Transformations:** Numpy arrays were reshaped into 3D matrices using `.reshape(-1, 30, 1)` to meet the `(Batch Size, Sequence Length, Input Dimension)` format expected by PyTorch's LSTM architecture, and then transferred to GPU/CPU as tensors.

### 2. PyTorch LSTM Model Architecture
A custom neural network architecture inheriting from PyTorch's `nn.Module` class was built.
* **LSTM Layer:** An LSTM network with `hidden_dim=32` and `num_layers=2` was used to capture long and short-term dependencies (trends) over time.
* **Fully Connected Layer:** The matrix output from the LSTM was passed through an `nn.Linear` layer, taking only the features at the last time step (`out[:, -1, :]`) to produce a 1D output (the predicted price).

### 3. Training Loop and Optimization
* **Loss Function:** `nn.MSELoss` (Mean Squared Error) was used as it is a continuous variable prediction (regression) task.
* **Optimization:** `Adam Optimizer` (`lr=0.01`) with a flexible learning rate was preferred for gradient descent.
* **Backpropagation:** Model weights were updated at each epoch using `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`.

### 4. Evaluation and Real-World Metrics
* **Evaluation Mode:** `model.eval()` was used to stop weight updates during prediction, and the `with torch.no_grad():` block was utilized to prevent unnecessary memory allocation.
* **Inverse Transform:** Scaled predictions (between -1 and 1) produced by the model were converted back to real-world prices (e.g., USD or TRY) using `scaler.inverse_transform()`.
* **Performance Measurement:** The success rate was calculated and visualized using Root Mean Squared Error (RMSE) on the original price scale.

### 5. UI Design and Deployment with Streamlit
This backend architecture was transformed into an interactive web application using Streamlit so that non-technical users can also experience it.
* Dynamically receives stock tickers (e.g., AAPL, XU100.IS, VOO) and the number of training rounds (Epochs) from the user.
* The model is trained in real-time on the interface, and the process is shown with a "Progress Bar".
* Upon completion, actual vs. predicted prices are plotted graphically using Matplotlib.
* The application was deployed live on Streamlit Community Cloud with the `requirements.txt` configuration.

## ⚙️ Installation and Local Setup

To test the project in your own environment:

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)

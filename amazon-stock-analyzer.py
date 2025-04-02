#!/usr/bin/env python
# coding: utf-8

# In[65]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.graph_objs as go
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score,f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout,Input, SimpleRNN, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf


# In[66]:


# Apple'ın verilerini yfinance ile alma
data = yf.download('AMZN', start='2021-05-15', end='2024-05-20')

# Tarih formatını eşleştirme
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date']).dt.date


# In[67]:


# Veriyi yfinance ile çekme
data = yf.download('AMZN', start='2021-05-15', end='2024-05-20')
data.reset_index(inplace=True)

# Figure nesnesini oluşturma
fig = go.Figure()

# Candlestick grafiğini figüre ekleme
fig.add_trace(go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'], name='Fiyat'))

# Grafiğin layout ayarlarını yapma
fig.update_layout(
    title='Amazon Hisse Fiyatı',
    xaxis_title='Tarih',
    yaxis_title='Fiyat (USD)',
    margin=dict(l=20, r=20, t=50, b=20)
)

# Grafiği gösterme (Jupyter Notebook için)
fig.show()


# In[68]:


#Tarih formatını eşleştirme ve indeks olarak ayarlama
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)


# In[69]:


# Teknik Göstergeleri Hesaplayan Fonksiyonlar
def MACD(data, period_short=12, period_long=26, period_signal=9):
    data['short_ema'] = data['Close'].ewm(span=period_short, adjust=False).mean()
    data['long_ema'] = data['Close'].ewm(span=period_long, adjust=False).mean()
    data['macd'] = data['short_ema'] - data['long_ema']
    data['signal_line'] = data['macd'].ewm(span=period_signal, adjust=False).mean()
    return data

def RSI(data, period=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    data['RSI'] = 100 - (100 / (1 + RS))
    return data

def Stochastic(data, period=14):
    low_min = data['Low'].rolling(window=period).min()
    high_max = data['High'].rolling(window=period).max()
    data['%K'] = (data['Close'] - low_min) * 100 / (high_max - low_min)
    data['%D'] = data['%K'].rolling(window=3).mean()
    return data

def Momentum(data, period=10):
    data['Momentum'] = data['Close'] - data['Close'].shift(period)
    return data


# In[70]:


# Genişletilmiş Alım-Satım Stratejisi Fonksiyonu
def trading_strategy(data):
    # Teknik göstergeleri hesaplama
    data = MACD(data)
    data = RSI(data)
    data = Stochastic(data)
    data = Momentum(data)
    
    # Alım-Satım Kararları
    signals = []
    
    for i in range(1, len(data)):
        # MACD ve RSI ile alım-satım sinyalleri
        if (data['RSI'].iloc[i] < 30 and data['macd'].iloc[i] > data['signal_line'].iloc[i]):
            signals.append({'Date': data.index[i], 'Signal': 'Alış Sinyali (RSI, MACD)'})
        elif (data['RSI'].iloc[i] > 70 and data['macd'].iloc[i] < data['signal_line'].iloc[i]):
            signals.append({'Date': data.index[i], 'Signal': 'Satış Sinyali (RSI, MACD)'})

        # Stochastic %K ve %D ile alım-satım sinyalleri
        if (data['%K'].iloc[i] < 20 and data['%K'].iloc[i] > data['%D'].iloc[i]):
            signals.append({'Date': data.index[i], 'Signal': 'Alış Sinyali (Stochastic)'})
        elif (data['%K'].iloc[i] > 80 and data['%K'].iloc[i] < data['%D'].iloc[i]):
            signals.append({'Date': data.index[i], 'Signal': 'Satış Sinyali (Stochastic)'})
        
        # Momentum ile alım-satım sinyalleri
        if (data['Momentum'].iloc[i] > 0 and data['Momentum'].iloc[i-1] < 0):
            signals.append({'Date': data.index[i], 'Signal': 'Alış Sinyali (Momentum)'})
        elif (data['Momentum'].iloc[i] < 0 and data['Momentum'].iloc[i-1] > 0):
            signals.append({'Date': data.index[i], 'Signal': 'Satış Sinyali (Momentum)'})

    # Sinyalleri DataFrame olarak dönüştürüp, görselleştirme
    signals_df = pd.DataFrame(signals)
    if not signals_df.empty:
        print(signals_df)
    else:
        print("Hiç sinyal üretilmedi.")
    return data


# In[71]:


pd.set_option('display.max_rows', None)  # Tüm satırları göster

# Tabloyu oluşturma
data = trading_strategy(data)



# In[72]:


# Sonuçları stil ile yazdırma - Sadece son 30 satırı al
last_30_data = data[['Close', 'short_ema', 'long_ema', 'macd', 'signal_line', 'RSI', '%K', '%D', 'Momentum']].tail(30)
styled_data = last_30_data.style.format("{:.2f}")
styled_data = styled_data.background_gradient(cmap='coolwarm').set_properties(**{'border': '1.3px solid black', 'color': 'black'})

styled_data


# In[73]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import yfinance as yf

app = dash.Dash(__name__)

# Kullanıcı Arayüzü Layout'u
app.layout = html.Div([
    html.H1("Stock Analysis with Technical Indicators"),
    
    dcc.Dropdown(
        id='indicator-dropdown',
        options=[
            {'label': 'MACD', 'value': 'MACD'},
            {'label': 'RSI', 'value': 'RSI'},
            {'label': 'Stochastic', 'value': 'Stochastic'},
            {'label': 'Momentum', 'value': 'Momentum'}
        ],
        value='MACD'
    ),
    
    dcc.Graph(id='stock-graph'),
    
])


# In[74]:


# Grafikleri güncellemek için callback fonksiyonu
@app.callback(
    Output('stock-graph', 'figure'),
    Input('indicator-dropdown', 'value')
)
def update_graph(selected_indicator):
    # Veriyi yfinance ile çek
    data = yf.download('AMZN', start='2021-05-15', end='2024-05-17')
    data.reset_index(inplace=True)
    
    # Figure nesnesi oluştur
    fig = go.Figure()
    
    if selected_indicator == 'MACD':
        ShortEMA = data['Close'].ewm(span=12, adjust=False).mean()
        LongEMA = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ShortEMA - LongEMA
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Signal_Line'], mode='lines', name='Signal Line'))
        
    elif selected_indicator == 'RSI':
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        RS = gain / loss
        data['RSI'] = 100 - (100 / (1 + RS))
        
        fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], mode='lines', name='RSI'))
        fig.add_shape(type="line", x0=data['Date'].min(), x1=data['Date'].max(), y0=30, y1=30, line=dict(color="Red", width=2, dash="dashdot"))
        fig.add_shape(type="line", x0=data['Date'].min(), x1=data['Date'].max(), y0=70, y1=70, line=dict(color="Red", width=2, dash="dashdot"))
        
    elif selected_indicator == 'Stochastic':
        low_min = data['Low'].rolling(window=14).min()
        high_max = data['High'].rolling(window=14).max()
        data['%K'] = (data['Close'] - low_min) * 100 / (high_max - low_min)
        data['%D'] = data['%K'].rolling(window=3).mean()
        
        fig.add_trace(go.Scatter(x=data['Date'], y=data['%K'], mode='lines', name='%K'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['%D'], mode='lines', name='%D'))
        fig.add_shape(type="line", x0=data['Date'].min(), x1=data['Date'].max(), y0=20, y1=20, line=dict(color="Green", width=2, dash="dashdot"))
        fig.add_shape(type="line", x0=data['Date'].min(), x1=data['Date'].max(), y0=80, y1=80, line=dict(color="Green", width=2, dash="dashdot"))
        
    elif selected_indicator == 'Momentum':
        data['Momentum'] = data['Close'] - data['Close'].shift(10)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Momentum'], mode='lines', name='Momentum'))
    
    fig.update_layout(
        title=f"{selected_indicator} Indicator for AAPL",
        xaxis_title='Date',
        yaxis_title='Indicator Value',
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8062)


# In[57]:


# NaN değerleri olan satırları kaldır
data.dropna(inplace=True)


# In[58]:


# Veriyi ölçeklendir
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume', 'macd', 'signal_line', 'RSI', '%K', '%D', 'Momentum']])


# In[20]:


# Veriyi hazırla
def prepare_data(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 3])  # Close sütunu genellikle index 3'te yer alır
    return np.array(X), np.array(y)


# In[21]:


window_size = 60  
X, y = prepare_data(scaled_data, window_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[22]:


def create_model(neurons=50, dropout_rate=0.2, optimizer='adam', learn_rate=0.001):
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_initializer=HeNormal()))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(neurons, return_sequences=False, kernel_initializer=HeNormal()))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer=HeNormal()))
    model.compile(optimizer=Adam(learning_rate=learn_rate) if optimizer == 'adam' else RMSprop(learning_rate=learn_rate), loss='mean_squared_error')
    return model


# In[23]:


from tensorflow.keras.callbacks import LearningRateScheduler

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch > 10:
        lr = lr * 0.5
    return lr

lr_callback = LearningRateScheduler(lr_scheduler)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
callbacks = [
    early_stopping_callback,
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
    lr_callback
]


# In[24]:


from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import HeNormal
# Hiperparametre kombinasyonları
params = {
    'neurons': [50, 100, 150],
    'dropout_rate': [0.0, 0.1, 0.2],
    'learn_rate': [0.001, 0.01],
    'optimizer': ['adam', 'rmsprop']
}

# En iyi model ve loss değeri
best_model = None
best_loss = float('inf')
best_params = {}

# Hiperparametre kombinasyonlarını deneyerek en iyi modeli bulma
for neurons in params['neurons']:
    for dropout_rate in params['dropout_rate']:
        for learn_rate in params['learn_rate']:
            for optimizer in params['optimizer']:
                print(f"Training model with neurons={neurons}, dropout_rate={dropout_rate}, learn_rate={learn_rate}, optimizer={optimizer}")
                model = create_model(neurons=neurons, dropout_rate=dropout_rate, optimizer=optimizer, learn_rate=learn_rate)
                history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks, verbose=0)
                val_loss = min(history.history['val_loss'])
                print(f"Validation loss: {val_loss}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = model
                    best_params = {
                        'neurons': neurons,
                        'dropout_rate': dropout_rate,
                        'learn_rate': learn_rate,
                        'optimizer': optimizer
                    }

print(f"Best parameters: {best_params}")
print(f"Best validation loss: {best_loss}")


# In[25]:


# En iyi parametrelerle modeli yeniden oluşturma ve eğitme
best_optimizer = Adam(learning_rate=best_params['learn_rate']) if best_params['optimizer'] == 'adam' else RMSprop(learning_rate=best_params['learn_rate'])
best_model = create_model(neurons=best_params['neurons'], dropout_rate=best_params['dropout_rate'], optimizer=best_optimizer, learn_rate=best_params['learn_rate'])
history = best_model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.2, callbacks=callbacks)



# In[ ]:





# In[26]:


# Kayıp değerlerini görselleştirme
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()


# In[27]:


# Karesel Hata
    
# Modeli değerlendirme
mse = best_model.evaluate(X_test, y_test)
print("Mean Squared Error:", mse)

mse_lstm = best_model.evaluate(X_test, y_test)
predictions_lstm = best_model.predict(X_test)
print("LSTM Mean Squared Error:", mse_lstm)


# In[75]:


# Sadece Close fiyatını geri ölçeklendirme
def inverse_transform_close(scaler, data, column_index):
    inverse_data = np.zeros((len(data), scaler.n_features_in_))
    inverse_data[:, column_index] = data[:, 0]
    return scaler.inverse_transform(inverse_data)[:, column_index]


# In[76]:


predicted_prices_lstm = predictions_lstm.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)

predicted_prices_lstm = inverse_transform_close(scaler, predicted_prices_lstm, 3)
actual_prices = inverse_transform_close(scaler, y_test_reshaped, 3)


# In[77]:


# Plot predictions
plt.figure(figsize=(14, 7))
plt.plot(actual_prices, color='blue', label='Actual AAPL Price')
plt.plot(predicted_prices_lstm, color='red', label='Predicted AAPL Price (lstm)')
plt.title('AAPL Stock Price Prediction (lstm)')
plt.xlabel('Time')
plt.ylabel('AAPL Stock Price')
plt.legend()
plt.show()


# In[ ]:





# In[78]:


# Test setinde kullanılan tarihlerle eşleşen tarihleri ekleme
test_dates = data.index[-len(y_test):].to_list()
actual_prices_df = pd.DataFrame(actual_prices, columns=['Adj Close'], index=test_dates)
predicted_prices_df = pd.DataFrame(predicted_prices_lstm, columns=['Predictions'], index=test_dates)

# Belirtilen tarih aralığında sonuçları gösterme
start_date = '2023-12-28'
end_date = '2024-05-20'

mask = (actual_prices_df.index >= start_date) & (actual_prices_df.index <= end_date)
filtered_actual_prices_df = actual_prices_df.loc[mask]
filtered_predicted_prices_df = predicted_prices_df.loc[mask]

# Birleştirilen sonuçları gösterme
result_df = pd.merge(filtered_actual_prices_df, filtered_predicted_prices_df, left_index=True, right_index=True)
result_df.index.name = 'Date'
result_df.reset_index(inplace=True)

print(result_df)


# In[79]:


# Stop loss ve take profit seviyeleri
stop_loss_threshold = 0.02  # %2 kayıp
take_profit_threshold = 0.03  # %3 kazanç
initial_capital = 10000  # Başlangıç sermayesi
shares = 15  # Alınacak hisse adedi

def execute_trades_strategy(data, initial_capital, shares, stop_loss_threshold, take_profit_threshold):
    capital = initial_capital
    initial_price = data['Close'].iloc[0]

    for i in range(1, len(data)):
        current_price = data['Close'].iloc[i]
        macd_signal = data['macd'].iloc[i]
        signal_line = data['signal_line'].iloc[i]
        rsi = data['RSI'].iloc[i]
        stochastic_k = data['%K'].iloc[i]
        stochastic_d = data['%D'].iloc[i]
        momentum = data['Momentum'].iloc[i]

        # Mevcut sermaye ve hisse adedi ile işlem yapma
        current_value = capital
        shares_value = shares * current_price
        portfolio_value = current_value + shares_value

        # Stop loss
        if (current_price - initial_price) / initial_price <= -stop_loss_threshold:
            capital -= shares_value
            print(f"Stop loss triggered at {current_price}. Remaining capital: {capital}")
            break

        # Take profit
        if (current_price - initial_price) / initial_price >= take_profit_threshold:
            capital += shares_value
            print(f"Take profit triggered at {current_price}. Total capital: {capital}")
            break

        # Sinyallere göre alım-satım
        if (rsi < 30 and macd_signal > signal_line) or (stochastic_k < 20 and stochastic_k > stochastic_d) or (momentum > 0):
            # Alış sinyali
            capital -= shares_value
            shares += shares
        elif (rsi > 70 and macd_signal < signal_line) or (stochastic_k > 80 and stochastic_k < stochastic_d) or (momentum < 0):
            # Satış sinyali
            capital += shares_value
            shares = 0

        # Güncellenmiş sermaye
        capital = portfolio_value
        initial_price = current_price

    return capital

# Stratejiye göre sermaye hesaplaması
final_capital_strategy = execute_trades_strategy(data, initial_capital, shares, stop_loss_threshold, take_profit_threshold)

print("Strategy Final Capital:", final_capital_strategy)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[80]:


# F1 Skoru Hesaplama ve Yazdırma
def calculate_f1_score(predicted, actual, threshold=0.01):
    predicted_signals = np.where(predicted > actual * (1 + threshold), 1, np.where(predicted < actual * (1 - threshold), -1, 0))
    actual_signals = np.where(actual[1:] > actual[:-1], 1, np.where(actual[1:] < actual[:-1], -1, 0))
    actual_signals = np.insert(actual_signals, 0, 0)  # İlk değeri nötr olarak ekle
    f1 = f1_score(actual_signals[:-1], predicted_signals[1:], average='macro', zero_division=1)
    return f1


# In[81]:


# Ensure the arrays are of the same length
if len(predicted_prices_lstm) != len(actual_prices):
    min_len = min(len(predicted_prices_lstm), len(actual_prices))
    predicted_prices_lstm = predicted_prices_lstm[:min_len]
    actual_prices = actual_prices[:min_len]

# Calculate F1 Score for LSTM
predicted_signals_lstm = np.where(predicted_prices_lstm > actual_prices * (1 + 0.01), 1, 
                                  np.where(predicted_prices_lstm < actual_prices * (1 - 0.01), -1, 0))
actual_signals = np.where(actual_prices[1:] > actual_prices[:-1], 1, 
                          np.where(actual_prices[1:] < actual_prices[:-1], -1, 0))

f1_lstm = f1_score(actual_signals, predicted_signals_lstm[1:], average='macro', zero_division=1)


# In[82]:


#f1_lstm = calculate_f1_score(predicted_prices_lstm, actual_prices)
f1_lstm = f1_score(np.sign(actual_prices[1:] - actual_prices[:-1]), np.sign(predicted_prices_lstm[1:] - predicted_prices_lstm[:-1]), average='macro', zero_division=1)

print("LSTM F1 Score:", f1_lstm)


# In[83]:


# Accuracy Hesaplama ve Yazdırma
def calculate_accuracy(predicted, actual, threshold=0.01):
    predicted_signals = np.where(predicted > actual * (1 + threshold), 1, np.where(predicted < actual * (1 - threshold), -1, 0))
    actual_signals = np.where(actual[1:] > actual[:-1], 1, np.where(actual[1:] < actual[:-1], -1, 0))
    actual_signals = np.insert(actual_signals, 0, 0)  # İlk değeri nötr olarak ekle
    accuracy = accuracy_score(actual_signals[:-1], predicted_signals[1:])
    return accuracy

#accuracy_lstm = calculate_accuracy(predicted_prices_lstm, actual_prices)
accuracy_lstm = accuracy_score(np.sign(actual_prices[1:] - actual_prices[:-1]), np.sign(predicted_prices_lstm[1:] - predicted_prices_lstm[:-1]))

print("LSTM Accuracy:", accuracy_lstm)                          


# In[84]:


# Gelecek günleri tahmin etme fonksiyonu
def predict_next_day(model, previous_data, window_size):
    next_days = []
    last_window = previous_data[-window_size:]
    
    for _ in range(1):
        X_pred = last_window.reshape((1, window_size, previous_data.shape[1]))
        pred_price = model.predict(X_pred)
        
        # Dummy array for inverse_transform
        pred_full = np.zeros((1, previous_data.shape[1]))
        pred_full[0, 3] = pred_price  # 'Close' fiyatı için tahmin
        
        # Update last_window with the predicted price
        last_window = np.append(last_window[1:], pred_full, axis=0)
        
        # Store the predicted price
        next_days.append(pred_full[0, 3])

    # Return the inverse transformed prices
    next_days = np.array(next_days).reshape(-1, 1)
    dummy = np.zeros((len(next_days), previous_data.shape[1]))
    dummy[:, 3] = next_days[:, 0]  # 'Close' sütunu
    return scaler.inverse_transform(dummy)[:, 3]

# Gelecek 1 günü tahmin etme
next_day_prediction = predict_next_day(best_model, scaled_data, window_size)
print("LSTM Next Day Prediction:", next_day_prediction)


# In[85]:


###############  P/E  ROE ORANLARI #########

# Apple'ın temel istatistiklerini ve mali bilgilerini yfinance ile çekme
ticker = yf.Ticker("AMZN")

# Temel istatistikleri ve mali bilgileri çekme
info = ticker.info
financials = ticker.financials
balancesheet = ticker.balance_sheet

# EPS ve Net Income değerlerini çekme
eps = info['trailingEps'] if 'trailingEps' in info else None  # Son dönem EPS değeri

# Mali bilgilerden Net Income değerini çekme
if 'Net Income' in financials.index:
    net_income = financials.loc['Net Income'].iloc[0]
elif 'Net Income Common Stockholders' in financials.index:
    net_income = financials.loc['Net Income Common Stockholders'].iloc[0]
else:
    net_income = None

# Bilançodan Total Stockholder Equity değerini çekme
if 'Total Stockholder Equity' in balancesheet.index:
    total_equity = balancesheet.loc['Total Stockholder Equity'].iloc[0]
elif 'Stockholders Equity' in balancesheet.index:
    total_equity = balancesheet.loc['Stockholders Equity'].iloc[0]
else:
    total_equity = None

# Şirketin güncel kapanış fiyatını almak için son veriyi çekme
data = yf.download('AMZN', start='2024-05-01', end='2024-05-14')
latest_price = data['Close'].iloc[-1]

# P/E Oranını Hesaplama
if eps:  # EPS bilgisi varsa P/E oranını hesapla
    pe_ratio_real_time = latest_price / eps
    print(f"Gerçek Zamanlı P/E Oranı: {pe_ratio_real_time:.2f}")
else:
    print("EPS bilgisi bulunamadı.")

# P/E Oranına Göre Yorum Yapma
if eps:
    if pe_ratio_real_time < 10:
        print(f"P/E oranı {pe_ratio_real_time:.2f}, çok düşük. Şirketin hisseleri düşük değerlendirilmiş olabilir, potansiyel bir değer yatırımı olabilir.")
    elif pe_ratio_real_time < 15:
        print(f"P/E oranı {pe_ratio_real_time:.2f}, düşük. Şirketin hisseleri makul bir değerlemeye sahip, orta düzeyde büyüme bekleniyor.")
    elif pe_ratio_real_time < 25:
        print(f"P/E oranı {pe_ratio_real_time:.2f}, orta. Şirketin hisseleri adil bir değerlemeye sahip, ortalama büyüme oranı bekleniyor.")
    elif pe_ratio_real_time < 40:
        print(f"P/E oranı {pe_ratio_real_time:.2f}, yüksek. Şirketin hisseleri yüksek değerlendirilmiş, yüksek büyüme beklentileri var ama aşırı değerlenme riski de taşıyor.")
    else:
        print(f"P/E oranı {pe_ratio_real_time:.2f}, çok yüksek. Şirketin hisseleri aşırı değerlendirilmiş, spekülatif bir bölgede ve yüksek fiyat düzeltmeleri riski barındırıyor.")

# ROE Hesaplama
if net_income and total_equity:  # Net Income ve Total Equity bilgisi varsa ROE hesapla
    roe_real_time = (net_income / total_equity) * 100  # Yüzde (%) olarak hesapla
    print(f"Gerçek Zamanlı ROE: {roe_real_time:.2f}%")

    # ROE'ye Göre Yorum Yapma
    if roe_real_time < 5:
        print(f"ROE {roe_real_time:.2f}%, çok düşük. Şirket sermayesini etkili kullanmıyor, finansal zorluklar veya düşük karlılık sektörleri işaret edebilir.")
    elif roe_real_time < 10:
        print(f"ROE {roe_real_time:.2f}%, düşük. Şirket sermayesini ortalama düzeyde kullanıyor, bazı operasyonel verimsizlikler olabilir.")
    elif roe_real_time < 20:
        print(f"ROE {roe_real_time:.2f}%, orta. Şirket sermayesini iyi kullanıyor, sağlıklı bir oranda kar elde ediyor.")
    elif roe_real_time < 40:
        print(f"ROE {roe_real_time:.2f}%, yüksek. Şirket sermayesini çok etkili kullanarak yüksek karlılık elde ediyor.")
    else:
        print(f"ROE {roe_real_time:.2f}%, çok yüksek. Olağanüstü yüksek karlılık gösteriyor, ancak yüksek borç seviyeleri veya riskler barındırabilir.")
else:
    print("Net Income veya Total Equity bilgisi bulunamadı.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





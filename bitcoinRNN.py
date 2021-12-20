# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 12:47:02 2021

@author: ytrop
"""

import requests
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import btalib
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from sklearn.preprocessing import StandardScaler 
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split 
from tqdm import trange 
from sklearn.preprocessing import MinMaxScaler


pio.renderers.default = 'browser'


# Carga de datos

url = "https://rest.coinapi.io/v1/exchangerate/BTC/USD/history?period_id=5MIN&time_start=2021-12-04T00:00:00&time_end=2021-12-10T00:00:00&limit=100000"


headers = {
  'X-CoinAPI-Key': '3623AB15-E80B-4FDA-A3A4-704E7E5A8AD3'
}

data = pd.read_json(r"C:\Users\ytrop\Desktop\datosCriptoOHLCV")

# Limpieza columnas y formateo de dataframe

data.drop(["time_period_end","time_open","time_close"],axis=1,inplace=True) 

data = data.rename( columns = {"time_period_start": "Time",
                        "price_open":"Open",
                        "price_high": "High",
                        "price_low" : "Low",
                        "price_close": "Close",
                        "volume_traded": "Volume",
                        "trades_count": "Count"})


group = data.groupby('Time')

data['Time'] =  pd.to_datetime(data['Time'],format = '%Y-%m-%d %H:%M')

data.set_index('Time',inplace = True)
print (data.dtypes)

"""sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.1, vmin=0)
plt.show()"""

# Comprobacion de null

data.isnull().any()

# Tamaño dataset

data.shape

data.head()

#Visualizacion candle sticks

"""candlestick = go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close']
                            )

fig = go.Figure(data=[candlestick])

fig.update_layout(
    width=800, height=600,
    title="Bitcoin Nebulova",
    yaxis_title='Bitcoin Prices'
)


fig.show()"""

### Indicadores metricos ###


# Analisis Bollinger bands

data['MA20'] = data['Close'].rolling(window=20).mean()
data['20dSTD'] = data['Close'].rolling(window=20).std() 

data['Upper'] = data['MA20'] + (data['20dSTD'] * 2)
data['Lower'] = data['MA20'] - (data['20dSTD'] * 2)

# Visualizacion Bollinger bands

"""fig = go.Figure(data=[go.Candlestick(x =data.index,
                                     open = data['Open'],
                                     high = data['High'],
                                     low = data['Low'],
                                     showlegend = False,
                                     close = data['Close'])])


parametros = ['MA20', 'Lower', 'Upper']
colores = ['blue', 'orange', 'orange']
for param,c in zip(parametros, colores):
    fig.add_trace(go.Scatter(
        x = data.index,
        y = data[param],
        showlegend = False,
        line_color = c,
        mode='lines',
        line={'dash': 'solid'},
        marker_line_width=2, 
        marker_size=10,
        opacity = 0.8))
    
fig.show()"""    
# Calculo RSI y MACD
 
rsi = btalib.rsi(data, period=14)
print(rsi.df.rsi[-1])
 
macd = btalib.macd(data, pfast=20, pslow=50, psignal=13)

data = data.join([rsi.df, macd.df])
print(data.tail())

#Visualizaciones de indicadores metricos RsI

"""fig = make_subplots(rows=2, cols=1)

fig.append_trace(
    go.Scatter(
        x=data.index,
        y=data['Open'],
        line=dict(color='#ff9900', width=1),
        name='open',
        legendgroup='1',
    ), row=1, col=1
)

fig.append_trace(
    go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='#ff9900',
        decreasing_line_color='black',
        showlegend=False
    ), row=1, col=1
)

fig.append_trace(
    go.Scatter(
        x=data.index,
        y=data['macd'],
        line=dict(color='#ff9900', width=2),
        name='macd',
        legendgroup='2',
    ), row=2, col=1
)

colors = np.where(data['histogram'] < 0, '#000', '#ff9900')


fig.append_trace(
    go.Bar(
        x=data.index,
        y=data['histogram'],
        name='histogram',
        marker_color=colors,
    ), row=2, col=1
)

layout = go.Layout(
    plot_bgcolor='#efefef',
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    xaxis=dict(
        rangeslider=dict(
            visible=False
        )
    )
)

fig.update_layout(layout)
fig.show()"""


#Normalizacion de datos

data = data.dropna()

scaler = StandardScaler() 

scaler.fit(data.values)  # sólo una vez y se guarda como si fuera el modelo 

dataset = scaler.transform(data.values) 



FEATURES_X = list(range(14))   

FEATURES_Y = list(range(4)) 

N_FEATURES_X = len(FEATURES_X)  # dataset.shape[0] 

N_FEATURES_Y = len(FEATURES_Y)   

LOOK_BACK = 50 
  

X = list() 

y = list() 

for i in trange(data.shape[0] - LOOK_BACK): 

    X.append(dataset[i : i + LOOK_BACK, FEATURES_X].tolist()) 

    y.append(dataset[i + LOOK_BACK, FEATURES_Y].tolist()) 



X = np.array(X) 

X = X.reshape(-1, LOOK_BACK, N_FEATURES_X) 

y = np.array(y) 

y = y.reshape(-1, N_FEATURES_Y) 

  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, shuffle=False) 


# Construcción del modelo

num_units = 4
activation_function = 'relu'
optimizer = 'adam'
loss_function = 'mean_squared_error'
batch_size = 5
num_epochs = 100


earlyStop=EarlyStopping(monitor="loss",patience=3)

regressor = Sequential ()

regressor.add(LSTM(units = num_units, activation = activation_function, input_shape=(LOOK_BACK,N_FEATURES_X, )))

regressor.add(Dense(units = N_FEATURES_Y  ))

regressor.compile(optimizer = optimizer, loss = loss_function)

regressor.fit(X_train, y_train, batch_size = batch_size, epochs = num_epochs,callbacks=[earlyStop])

regressor.summary()

tb=TensorBoard(log_dir='C:/temp/tensorflow_logs/fig0')

history = regressor.fit(X_train, y_train, epochs=300, batch_size=100, validation_data=(X_train, y_train),
                        verbose=0, callbacks = [tb],shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


results = regressor.evaluate(X_test,y_test)

train_preds = regressor.predict(X_test)


train_preds.shape
y_test.shape

scaler_y = StandardScaler()
scaler_y.mean_ = scaler.mean_[:4]
scaler_y.var_ = scaler.var_ [:4]
scaler_y.scale_ = scaler.scale_[:4]

predictions = scaler_y.inverse_transform(train_preds)
real_values = scaler_y.inverse_transform(y_test)


"""train_preds = train_preds.reshape(-1, N_FEATURES_Y) 
predictions = scaler.inverse_transform(train_preds)
real_values = scaler.inverse_transform(y_test)"""

plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()  
plt.plot(real_values[3] , color = 'red', label = 'Real Price')
plt.plot(predictions[3], color = 'blue', label = 'Predicted Price')
plt.title('BTC Prediction', fontsize=40)
df_test = dataset.reset_index()
x=df_test.index
labels = df_test['date']
plt.xticks(x, labels, rotation = 'vertical')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
plt.xlabel('Time', fontsize=40)
plt.ylabel('BTC', fontsize=40)
plt.legend(loc=2, prop={'size': 25})
plt.show()
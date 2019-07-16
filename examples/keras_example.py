# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:26:22 2019
"""
import __init__
import numpy as np

from pyetf.algos import diffMA, slopeMA 
def addFeatures(dataset):
    dataset['weekday'] = dataset.index.weekday
    dataset['diff'] = diffMA(dataset.price)
    dataset['slope'] = slopeMA(dataset.price)
    return dataset.dropna()

from pyetf.algos import future_mean_var
def addTarget(dataset, futureDays=30):
    prices = dataset.price
    m = futureDays
    y_var = []
    for t in range(0, len(prices)-m+1):
        p = prices.iloc[t:t+m]
        _, var = future_mean_var(p)
        y_var.append(var*10000)
    y_min = min(y_var)
    y_max = max(y_var)
    y_distance = y_max-y_min
    for t in range(len(y_var)):
        y_var[t] = (y_var[t]-y_min)/y_distance
    for t in range(len(prices)-m+1, len(prices)):
        y_var.append(np.nan)
    dataset['y'] = y_var
    return dataset.dropna()

def buildXY(dataset, pastDays=30):
    """
    Result -> numpy
    """
    m = pastDays
    x_dataset = dataset.drop(columns='y').values
    y_dataset = dataset['y'].values
    dataX, dataY = [], []
    for t in range(0, len(dataset)-m+1):
        dataX.append(x_dataset[t:(t+m)])
        dataY.append(y_dataset[t+m-1])
    return np.array(dataX), np.array(dataY)

from sklearn.preprocessing import MinMaxScaler
def normalise_windows(window_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalised_data = []
    for window in window_data:
        normalised_window = scaler.fit_transform(window)
        normalised_data.append(normalised_window)
    return normalised_data

def splitDataset(x_dataset, y_dataset, train_size_ratio=0.6):
    train_size = int(len(x_dataset) * train_size_ratio)
    x_train, x_test = x_dataset[0:train_size], x_dataset[train_size:len(x_dataset)]
    y_train, y_test = y_dataset[0:train_size], y_dataset[train_size:len(y_dataset)]
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def shuffleDataset(x, y):
    np.random.seed(10)
    randomList = np.arange(x.shape[0])
    np.random.shuffle(randomList)
    return x[randomList], y[randomList]
    
# 1. Data Process
import ffn
from pyetf.data import eod
# 1.1 read data
etf_tickers = ['SHY']
model_filename = 'est_var(' + etf_tickers[0].lower() + ').h5'
start_date_str = '2002-01-01'
prices = ffn.get(tickers=etf_tickers, market='US', 
                 provider=eod, 
                 start=start_date_str)
dataset = prices.copy()
dataset = dataset.rename({etf_tickers[0].lower():'price'}, axis=1)
# 1.2 add features to X
dataset = addFeatures(dataset)
# 1.3 add targets to Y
dataset = addTarget(dataset)
# 1.4 structure train and test data
x_dataset, y_dataset = buildXY(dataset)
# 1.5 normalization
x_dataset = normalise_windows(x_dataset)
# 1.6 split train set and test set
x_train, y_train, x_test, y_test = splitDataset(x_dataset, y_dataset)
# 1.7 shuttle train set
x_train, y_train = shuffleDataset(x_train, y_train)

# 2. Build Model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
# 2.1 setup model
model = Sequential()
model.add(LSTM(20, input_length=x_train.shape[1], input_dim=x_train.shape[2]))
model.add(Dense(1))
model.add(Dropout(0.2))
model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
model.summary()
# 2.2 train model
model.fit(x_train, y_train, epochs=1000, batch_size=min(1000,x_train.shape[0]), verbose=1)
# 2.3 save model
model.save(model_filename)

# 3. Prediction and Evaluation
from keras.models import load_model
# 3.1 load model
model_load = load_model(model_filename)
# 3.2 predication
#trainPredict = model_load.predict(x_train)
testPredict = model_load.predict(x_test)
y_predict = model_load.predict(np.array(x_dataset))
# 3.3 evaluation
trainScore = model_load.evaluate(x_train, y_train)
testScore = model_load.evaluate(x_test, y_test)
print('Train Score Loss: %.4f' % (trainScore[0]))
print('Test Score Loss: %.4f' % (testScore[0]))

# 4. Plot Results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
#plt.plot(y_dataset)
#plt.plot(y_predict)
plt.plot(y_test)
plt.plot(testPredict)
plt.show()
#upload file
from google.colab import files
uploaded = files.upload()

#import data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#np.random.seed(3)
#Data = pd.read_csv('filename.csv',header=0, usecols=['Date','Close'],parse_dates=True,index_col='Date')
Data = pd.read_csv('train.csv')
Data = Data.T
Data=Data.drop('id',0)
#plt.figure(figsize=(10,5))
#plt.plot(Data)
#plt.show()

##Data Analysis
#DataPCh = Data.pct_change()
#LogReturns = np.log(1 + DataPCh) 
#print(LogReturns.tail(10))
#plt.figure(figsize=(10,5))
#plt.plot(LogReturns)
#plt.show()

#Data Size Adjustment
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
DataScaled = scaler.fit_transform(Data)

#Data Division
TestLen = 12
TrainLen = int(len(DataScaled) - TestLen)
TrainData = DataScaled[0:TrainLen, :]
TestData = DataScaled[TrainLen:len(DataScaled), :]
print(len(TrainData), len(TestData))

#LSTM Dataset Creation
def DatasetCreation(dataset, TimeStep=1):
    DataX, DataY = [], []
    for i in range(len(dataset)- TimeStep -1):
            a = dataset[i:(i+ TimeStep), 0]
            DataX.append(a)
            DataY.append(dataset[i + TimeStep, 0])
    return np.array(DataX), np.array(DataY)

TimeStep = 1
TrainX, TrainY = DatasetCreation(TrainData, TimeStep)
TestX, TestY = DatasetCreation(TestData, TimeStep)
TrainX = np.reshape(TrainX, (TrainX.shape[0], 1, TrainX.shape[1]))
TestX = np.reshape(TestX, (TestX.shape[0], 1, TestX.shape[1]))

#Keras LSTM model generation
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import tensorflow as tf
#tf.random.set_seed(3)

model = tf.keras.Sequential()
model.add(layers.LSTM(256, input_shape=(1, TimeStep)))
model.add(layers.Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
model.fit(TrainX, TrainY, epochs=10, batch_size=1, verbose=1)
model.summary()

score = model.evaluate(TrainX, TrainY, verbose=0)
print('Keras Model Loss = ',score[0])
print('Keras Model Accuracy = ',score[1])

#Prediction
TrainPred = model.predict(TrainX)
TestPred = model.predict(TestX)

TrainPred = scaler.inverse_transform(TrainPred)
TrainY = scaler.inverse_transform([TrainY])
TestPred = scaler.inverse_transform(TestPred)
TestY = scaler.inverse_transform([TestY])

TrainPredictPlot = np.empty_like(DataScaled)
TrainPredictPlot[:, :] = np.nan
TrainPredictPlot[1:len(TrainPred)+1, :] = TrainPred

TestPredictPlot = np.empty_like(DataScaled)
TestPredictPlot[:, :] = np.nan
TestPredictPlot[len(TrainPred)+(1*2)+1:len(DataScaled)-1, :] = TestPred

plt.plot(scaler.inverse_transform(DataScaled))
plt.plot(TrainPredictPlot)
plt.plot(TestPredictPlot)
plt.show()

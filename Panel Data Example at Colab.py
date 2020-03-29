#upload file
from google.colab import files
uploaded = files.upload()

#import data
classNames = ['A','B','C','D']
data = pd.read_csv('filename.csv', names=classNames)

#Removing missing values
import numpy as np
newData = data.replace('?', np.nan)
print(dataNew.info())
print(dataNew.describe())
print(dataNew.isnull().sum())
newData = newData.dropna()
print(newData.info())
print(newData.isnull().sum())

#Divide DataFrame
InputNames = classNames
InputNames.pop()
columnRange = len(newData.columns)-1
Input = pd.DataFrame(newData.iloc[:, 0:columnRange], columns=InputNames)
Target = pd.DataFrame(newData.iloc[:, columnRange], columns=['D'])

#Data scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(Input))
InputScaled = scaler.fit_transform(Input)
InputScaled = pd.DataFrame(InputScaled,columns=InputNames)
summary = InputScaled.describe()
summary = summary.transpose()
print(summary)

#Split the data
from sklearn.model_selection import train_test_split
Input_train, Input_test, Target_train, Target_test = train_test_split(InputScaled, Target, test_size = 0.30, random_state = 5)
print(Input_train.shape)
print(Input_test.shape)
print(Target_train.shape)
print(Target_test.shape)

# learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
model = tf.keras.Sequential()
model.add(layers.Dense(30, input_dim=13, activation='tanh'))
model.add(layers.Dense(20, activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(Input_train, Target_train, epochs=1000, verbose=1)
model.summary()

score = model.evaluate(Input_test, Target_test, verbose=0)
print('Keras Model Accuracy = ',score[1])




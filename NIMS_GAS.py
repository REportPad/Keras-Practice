#https://www.tensorflow.org/tutorials/structured_data/time_series
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

csv_path = 'C:/ML/city gas/train2.csv'

#(Nx125)
df = pd.read_csv(csv_path)

#transpose
df = df.T
df = df.drop('id',0)#(125 x 78587)

#history_size is the size of the past window of information. 
#The target_size is how far in the future does the model need to learn to predict. 
#The target_size is the label that needs to be predicted.
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False): #univariate: dataset, start_index, end_index, history_size, target_size
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

#the first 113 rows of the data will be the training dataset, and there remaining will be the validation dataset.
TRAIN_SPLIT = 113 #125-12

#Setting seed to ensure reproducibility.
#tf.random.set_seed(13)

##Part 2: Forecast a multivariate time series
#features_considered = ['0':'78525']#['p (mbar)', 'T (degC)', 'rho (g/m**3)']
#features = df[features_considered]
#features.index = df['Date Time']

#the first step will be to standardize the dataset using the mean and standard deviation of the training data.
#dataset = features.values
data_mean = df[:TRAIN_SPLIT].mean(axis=0)#dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = df[:TRAIN_SPLIT].std(axis=0)#dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (df-data_mean)/data_std

#Multi-step model
#the network is shown data from the last five (5) days, i.e. 720 observations that are sampled every hour. 
#The sampling is done every one hour since a drastic change is not expected within 60 minutes. 
#Thus, 120 observation represent history of the last five days.
#For the single step prediction model, the label for a datapoint is the temperature 12 hours into the future. 
#In order to create a label for this, the temperature after 72(12*6) observations is used.
future_target = 12 #the number of predicted point
past_history = 12 #In order to make this prediction, you choose to use 5 days of observations.
STEP = 1
dataset = dataset.to_numpy()
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1],           0, TRAIN_SPLIT, past_history, future_target, STEP)
x_val_multi, y_val_multi     = multivariate_data(dataset, dataset[:, 1], TRAIN_SPLIT,        None, past_history, future_target, STEP)

#train data, test data
BATCH_SIZE = 8#??
#BUFFER_SIZE = 10000 #shuffle will initially select a random element from only the first 1,0000 elements in the buffer.
train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
#train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_data_multi = train_data_multi.batch(BATCH_SIZE).repeat()
val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

#LSTM model
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape = x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(12)) #since 12 predictions are made, the dense layer outputs 12 predictions.
multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

EVALUATION_INTERVAL = 200 #Train for 200 steps
EPOCHS = 10
multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)



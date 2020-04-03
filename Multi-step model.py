#https://www.tensorflow.org/tutorials/structured_data/time_series
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

#about graph
#mpl.rcParams['figure.figsize'] = (8, 6)
#mpl.rcParams['axes.grid'] = False

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

#(420551x14)
df = pd.read_csv(csv_path)

#history_size is the size of the past window of information. 
#The target_size is how far in the future does the model need to learn to predict. 
#The target_size is the label that needs to be predicted.
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
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

#the first 300,000 rows of the data will be the training dataset, and there remaining will be the validation dataset.
TRAIN_SPLIT = 300000

#Setting seed to ensure reproducibility.
tf.random.set_seed(13)

##Part 2: Forecast a multivariate time series
features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
features = df[features_considered]
features.index = df['Date Time']

#the first step will be to standardize the dataset using the mean and standard deviation of the training data.
dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset-data_mean)/data_std

#Multi-step model
#the network is shown data from the last five (5) days, i.e. 720 observations that are sampled every hour. 
#The sampling is done every one hour since a drastic change is not expected within 60 minutes. 
#Thus, 120 observation represent history of the last five days.
#For the single step prediction model, the label for a datapoint is the temperature 12 hours into the future. 
#In order to create a label for this, the temperature after 72(12*6) observations is used.
future_target = 72 #(12*6)
past_history = 720 #(5days * 24hour * 6times)
STEP = 6
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)
#train data, test data
BATCH_SIZE = 256
BUFFER_SIZE = 10000 #shuffle will initially select a random element from only the first 1,0000 elements in the buffer.

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

#LSTM model
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(72))
multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

EVALUATION_INTERVAL = 200 #Train for 200 steps
EPOCHS = 10
multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)

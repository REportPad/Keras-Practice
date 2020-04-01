import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('C:/ML/mltoolkit-master/data/aptsellindex_gangnamgu.csv',parse_dates =["date"], index_col ="date") 

split_date = pd.Timestamp('01-01-2017')
train = df.loc[:split_date, ['trade_price_idx_value']]
test = df.loc[split_date:, ['trade_price_idx_value']]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

train_sc_df = pd.DataFrame(train_sc, columns=['trade_price_idx_value'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['trade_price_idx_value'], index=test.index)

for s in range(1, 13):
    train_sc_df['shift_{}'.format(s)] = train_sc_df['trade_price_idx_value'].shift(s)
    test_sc_df['shift_{}'.format(s)] = test_sc_df['trade_price_idx_value'].shift(s)

X_train = train_sc_df.dropna().drop('trade_price_idx_value', axis=1)
y_train = train_sc_df.dropna()[['trade_price_idx_value']]

X_test = test_sc_df.dropna().drop('trade_price_idx_value', axis=1)
y_test = test_sc_df.dropna()[['trade_price_idx_value']]

X_train = X_train.values
X_test= X_test.values

y_train = y_train.values
y_test = y_test.values

X_train_t = X_train.reshape(X_train.shape[0], 12, 1)
X_test_t = X_test.reshape(X_test.shape[0], 12, 1)

from keras.layers import LSTM 
from keras.models import Sequential 
from keras.layers import Dense 
import keras.backend as K 
from keras.callbacks import EarlyStopping

K.clear_session()
    
model = Sequential()
model.add(LSTM(20, input_shape=(12, 1))) # (timestep, feature) 
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam') 
model.summary()
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

model.fit(X_train_t, y_train, epochs=100, batch_size=30, verbose=1, callbacks=[early_stop])
          
df_y = pd.DataFrame(y_train)
df_y.plot()


#model 저장하기
from keras.models import load_model
model.save('lstm_model.h5')

#model 불러오기
from keras.models import load_model
model = load_model('lstm_model.h5')

#model 사용하기
yhat = model.predict_classes(X_test_t)

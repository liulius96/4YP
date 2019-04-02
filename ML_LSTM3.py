from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np
import pandas as pd
import datetime

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# goodness of fit
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# define some functions to find the test statistics
def rmse(actual_val, predicted_val):
    return sqrt(mean_squared_error(actual_val,predicted_val))

def r_sq(actual_val, predicted_val):
    return r2_score(actual_val,predicted_val)


# load data
btc = pd.read_pickle('C:/Users/tliu/Documents/4YP/2013_btc_data.pkl')
df = btc.loc[:datetime.date(year=2018, month=1,day=1)]
# define input sequence
raw_seq = df.BTC.values
# choose a number of time steps
n_steps = 1
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)


# demonstrate prediction
df_predict = btc.loc[datetime.date(year=2018, month=1,day=1):datetime.date(year=2018, month=1,day=31)]
x_input = array(df_predict.BTC.values)
# reshape into 3d array
x_input = x_input.reshape(-1,1,1)
yhat = model.predict(x_input, verbose=0)
print(yhat)

# plot
forecast = pd.DataFrame(yhat, index=btc.loc[datetime.date(year=2018,month=1,day=1):datetime.date(year=2018,month=1,day=31)].index, columns=['Prediction'])
result=pd.merge(btc.loc[datetime.date(year=2017,month=9,day=1):datetime.date(year=2018,month=1,day=31)], forecast, how='outer', left_index=True, right_index=True)
# plot
ax = result.plot(x='Date', y=['BTC', 'Prediction'], title='Bitcoin Price Forecast')
ax.set_ylabel("Price (USD)")
# comparisons
valid = df_predict.BTC.values
btc_r2 = r_sq(valid, yhat)
btc_rmse = rmse(valid, yhat)
# set up text box
textstr = '\n'.join((
    r'$RMSE=%.2f$' % (btc_rmse, ),
    r'$R squared=%.2f$' % (btc_r2, )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05,0.95, textstr, transform=ax.transAxes, fontsize=16,
        verticalalignment='top', bbox=props)
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.plt as plt
import statsmodels.api as sm
from sklearn.ensemble.forest import RandomForestRegressor

btc_df = pd.read_csv('C:/Users/Liuli/OneDrive/Documents/4YP/crypto_sentiment_daily_df.csv')
btc_df = btc_df[['Date', 'BTC', 'BVOL24H', 'BTC_Volume', 'SPX']]
btc_df['Date'] = pd.to_datetime(btc_df['Date'])
btc_df.set_index('Date', inplace=True)
btc_df = np.log(btc_df) - np.log(btc_df.shift(1))
btc_df = btc_df.sort_index(ascending=True)
btc_df = btc_df.dropna()

# split into input and output?
X = np.asarray(btc_df.drop(columns='BTC'))
Y = np.asarray(btc_df.BTC)

# fit random forest model
model = RandomForestRegressor(n_estimators=500, random_state=1)
model.fit(X,Y)

# show importance scores
print(model.feature_importances_)

# plot importance scores
names = btc_df.drop(columns='BTC').columns
ticks = [i for i in range(len(names))]
plt.bar(ticks, model.feature_importances_)
plt.xticks(ticks, names)
plt.show()


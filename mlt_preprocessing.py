import pandas as pd
import numpy as np

# load the raw o/p from LDA models
df = pd.read_pickle('C:/Users/Tony/Documents/4YP/data/pickles/per-doc-topics/MLT_df_alpha-1_5-topics.pkl')

# create a series where the most likely topics are grouped by date
df_mlt = df['Most Likely Topic'].groupby('Date').value_counts()

# reset the index (this turns it into a dataframe)
df_mlt = df_mlt.reset_index(name='Count')

# pivot the values
df_mlt = df_mlt.pivot(index='Date', columns='Most Likely Topic', values='Count')


# load the bitcoin df
btc_df = pd.read_csv('C:/Users/Tony/Documents/4YP/data/crypto_sentiment_daily_df.csv')
btc_df = btc_df[['Date', 'BTC', 'BVOL24H', 'BTC_Volume', 'SPX']]
# btc_df['Daily_Return'] = btc_df['BTC'].pct_change()

# set the date column as the index
btc_df.set_index('Date', inplace=True)
btc_df.index = pd.to_datetime(btc_df.index)

# merge the two dataframes
btc_df = btc_df.merge(df_mlt, how='inner', left_index=True,right_index=True)

# change the dataframe to pct_change per day so everything is normalised
btc_dr = btc_df.pct_change(1)

# change the dataframe to log diff
btc_ld = np.log(btc_df) - np.log(btc_df.shift(1))
btc_ld = btc_ld.dropna()

# save the dataframes
btc_dr.to_pickle('C:/Users/Tony/Documents/4YP/data/pickles/per-doc-topics/btc_dr_with_mlt_alpha-1_topics-5.pkl')
btc_ld.to_pickle('C:/Users/Tony/Documents/4YP/data/pickles/per-doc-topics/btc_ld_with_mlt_alpha-1_topics-5.pkl')

# create a correlation matrix for every column of the pct_change (daily returns) dataframe
corr_matrix = btc_dr.corr(method='pearson')


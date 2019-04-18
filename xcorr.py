import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

btc_dr = pd.read_pickle('C:/Users/Tony/Documents/4YP/data/pickles/per-doc-topics/btc_dr_with_mlt_alpha-1_topics-5.pkl')
btc_ld = pd.read_pickle('C:/Users/Tony/Documents/4YP/data/pickles/per-doc-topics/btc_ld_with_mlt_alpha-1_topics-5.pkl')

# plt.acorr(btc_dr['BTC'].dropna().values, maxlags=20)

def plot_xcorr_acorr(x, name_x, y ,name_y ,max_lags ,save_path ):
    fig, [ax1, ax2] = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)
    sig_val = 2 / (len(x)) ** (1 / 2)
    ax1.axhline(sig_val, color='b')
    ax1.axhline(-sig_val, color='b')
    ax1.xcorr(x, y, usevlines=True, maxlags=max_lags, normed=True, lw=2)
    # ax1.grid(True)
    ax1.axhline(0, color='black', lw=2)
    ax1.set_title('Cross Correlation between ' + str(name_x) + ' and ' +str(name_y))

    ax2.acorr(x, usevlines=True, normed=True, maxlags=max_lags, lw=2)
    ax2.grid(True)
    ax2.axhline(0, color='black', lw=2)
    ax2.set_title('Autocorrelation for ' + str(name_x))

    plt.savefig(save_path + '/' +str(name_x) + '_and_' +str(name_y)+'_xcorr.pdf', bbox_inches='tight')

s1 = btc_dr[0].dropna()
s2 = btc_dr.BTC.dropna()
path = 'C:/Users/Tony/Documents/4YP/Figures/cross_corr'

plot_xcorr_acorr(s1.values,'Topic 0', s2.values, 'BTC', 20, path)

for x in range(0,5):
    s1 = btc_dr[x].dropna()
    s1_name = 'Topic '+str(x)
    s2 = btc_dr.BTC.dropna()
    s2_name = 'BTC'
    plot_xcorr_acorr(s1,s1_name,s2,s2_name,20,path)
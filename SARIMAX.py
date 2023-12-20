import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

file = 'dataset/coldplay_grouped_by.csv'
coldplay = pd.read_csv(file)

# ACF, PACF graphs to help determine order of ARIMA model, again statsmodel has these handy functions built-in
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(coldplay['Streams'][1:], lags=40, ax=ax1)   # first value of diff is NaN
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(coldplay['Streams'][1:], lags=40, ax=ax2)
plt.show()

file = 'dataset/coldplay.csv'
coldplay = pd.read_csv(file)
coldplay = coldplay.groupby(['Track Name'])['Streams'].sum().reset_index()
print(coldplay.sort_values(by=['Streams']))
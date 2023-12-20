import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tabulate import tabulate

file = 'dataset/coldplay_grouped_by.csv'
coldplay = pd.read_csv(file)

coldplay['Date'] = pd.to_datetime(coldplay['Date'])
coldplay.set_index('Date', inplace=True)

streams = coldplay['Streams']

# Construct the model
mod = sm.tsa.SARIMAX(streams, order=(1, 0, 0), trend='c')
# Estimate the parameters
res = mod.fit(disp=0)

print(res.summary())

# Forecasting with five-step prevision
fcast_result = res.get_forecast(steps=5)
fcast_result_summary = fcast_result.summary_frame()

fcast_result_summary.index.name = 'Step'

print("Summary Frame for Forecast Result (Next 5 Steps):")
print(tabulate(fcast_result_summary, headers='keys', tablefmt='pretty'))

# Plot result
fig, ax = plt.subplots(figsize=(15, 5))

# Plot the data
streams.plot(ax=ax, label='Historical Data', color='g')

# Construct the forecast
forecast_steps = 50
fcast = res.get_forecast(steps=forecast_steps).summary_frame()
forecast_index = pd.date_range(start=streams.index[-1] + pd.DateOffset(1), periods=forecast_steps, freq='D')
fcast.index = forecast_index

fcast['mean'].plot(ax=ax, style='k--', label='Forecast')
ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1)

plt.xlabel('Date')
plt.ylabel('Streams')
plt.title('SARIMAX Forecast')

plt.legend()
plt.savefig('plots/sarimax_forecast.png', bbox_inches='tight')
plt.show()

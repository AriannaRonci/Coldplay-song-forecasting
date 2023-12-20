import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

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

# The default is to get a one-step-ahead forecast
forecast_result = res.forecast()
print(f"Forecast Result: {forecast_result}\n")

fcast_res1 = res.get_forecast()
# Most results are collected in the `summary_frame` attribute.
# Here we specify that we want a confidence level of 90%
print("Summary Frame with 90% Confidence Level:")
print(fcast_res1.summary_frame(alpha=0.1))

fcast_res2 = res.forecast(steps=2)
print(f"\nForecast Result for the Next 2 Steps: \n{fcast_res2}")

# Plot result
fig, ax = plt.subplots(figsize=(15, 5))

# Plot the data
streams.plot(ax=ax, label='Historical Data')

# Construct the forecast
forecast_steps = 56
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

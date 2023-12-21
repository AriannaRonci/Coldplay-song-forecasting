import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime as dt
import sklearn.metrics as met
from ARIMA import ADF_test
from ARIMA import ACF_PACF

file = 'dataset/coldplay_grouped_by.csv'
coldplay = pd.read_csv(file)

coldplay['Date'] = pd.to_datetime(coldplay['Date'])
coldplay.set_index('Date', inplace=True)
coldplay = coldplay.iloc[:, 1:2]

streams = coldplay['Streams']

'''coldplay_date = coldplay.iloc[:, 1:3]
coldplay_date.set_index(pd.DatetimeIndex(coldplay.Date), inplace=True)
coldplay_date.drop(columns='Date', inplace=True)'''

seasonal_diff = seasonal_decompose(coldplay, model='additive', extrapolate_trend='freq').seasonal


def next_prediction(p, d, q, P, D, Q, m, df):
    # Construct the model
    mod = sm.tsa.SARIMAX(df.values, order=(p, d, q), seasonal_order=(P, D, Q, m), trend='c')
    # Estimate the parameters
    res = mod.fit(disp=0)

    fcast_result = res.get_forecast(steps=5)
    fcast_result_summary = fcast_result.summary_frame()

    fcast_result_summary.index.name = 'Step'

    # Plot result
    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot the data
    streams.plot(ax=ax, label='Historical Data', color='skyblue')

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
    plt.savefig('SARIMAX/next_prediction.png', bbox_inches='tight')
    plt.show()


def SARIMAX_model(p, d, q, P, D, Q, m, df):
    model = SARIMAX(df.values, order=(p, d, q), seasonal_order=(P, D, Q, m))
    results = model.fit()

    ax = plt.gca()
    plt.plot(df)
    plt.plot(pd.DataFrame(results.fittedvalues, columns=['Streams']).set_index(df.index), color='red')
    ax.legend(['Streams', 'Forecast'])
    plt.savefig('SARIMAX/Predictions', bbox_inches='tight')
    ax.set_xlim(dt.datetime(2017, 1, 1), dt.datetime(2018, 1, 9))
    plt.show()
    print(results.summary())

    #residual error
    residuals = pd.DataFrame(results.resid)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.savefig('SARIMAX/Residual_Error', bbox_inches='tight')
    plt.show()

    results.plot_diagnostics(figsize=(12, 8))
    plt.savefig('SARIMAX/Diagnostics', bbox_inches='tight')
    plt.show()


def train_SARIMAX(p, d, q, P, D, Q, m, df):
    # Create Training and Test
    train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)
    index_list = list(df.index)
    train_len = int(len(index_list) * 0.8)
    date_split = str(index_list[train_len + 1])
    ax = df.plot(color='b', label='Train')
    df.loc[date_split:].plot(color='r', label='Test', ax=ax)

    # Fit SARIMAX model on training data
    model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, m))
    sarimax_model = model.fit()

    # Forecast
    pred_uc = sarimax_model.get_forecast(steps=len(test_data))
    pred_ci = pred_uc.conf_int()

    pd.DataFrame(pred_uc.predicted_mean).set_index(pd.DatetimeIndex(test_data.index))['predicted_mean'].plot(ax=ax, label='Forecast', style='k--')
    ax.fill_between(pred_ci.set_index(pd.DatetimeIndex(test_data.index).to_period('D')).index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.05)

    ax.set_xlabel('Date')
    ax.set_ylabel('Streams')
    ax.set_xlim(dt.datetime(2017, 1, 1), dt.datetime(2018, 1, 9))

    plt.legend()
    plt.savefig('SARIMAX/Forecasting', bbox_inches='tight')
    plt.show()

    return pred_uc, test_data


ADF_test(seasonal_diff)
ACF_PACF(seasonal_diff, 'SARIMAX/ACF_PACF.png')
SARIMAX_model(1, 0, 2, 1, 0, 1, 7, coldplay['Streams'])
train_SARIMAX(1, 0, 2, 1, 0, 1, 7, coldplay)
next_prediction(1, 0, 2, 1, 0, 1, 7, streams)

pred_uc, test_data = train_SARIMAX(1, 0, 2, 1, 0, 1, 7, coldplay)

predicted = pred_uc.predicted_mean
mape = met.mean_absolute_percentage_error(test_data, predicted)
sqe = met.mean_squared_error(test_data.squeeze(), predicted)
mae = met.mean_absolute_error(test_data, predicted)
r2 = met.r2_score(test_data, predicted)
print(mape)
print(sqe)
print(mae)
print(r2)

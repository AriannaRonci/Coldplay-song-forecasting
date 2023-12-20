import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
from statsmodels.tsa.seasonal import seasonal_decompose

file_coldplay = 'dataset/coldplay_grouped_by.csv'
coldplay = pd.read_csv(file_coldplay)

file_song = 'dataset/song_grouped_by.csv'
song = pd.read_csv(file_song)


def streams_by_date(dataset, name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot_date(dataset['Date'], dataset['Streams'], fmt='g-', color='skyblue')  # x = array of dates, y = array of numbers
    fig.autofmt_xdate()
    # For tickmarks and ticklabels every week
    ax.xaxis.set_major_locator(MonthLocator())

    # For tickmarks and ticklabels every other week
    # ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MO, interval=2))

    plt.xlabel("Date")
    plt.ylabel("Streams")
    plt.title("Number of Streams by Date")

    plt.grid(True)
    plt.savefig(f"plots/streams_by_date_{name}", bbox_inches='tight')
    plt.show()


def streams_by_day(dataset, name):
    day = dataset['Day of week'].unique()
    plt.figure(figsize=(10, 5))
    plt.bar(day, dataset.groupby('Day of week')['Streams'].sum(), width=0.5, color='skyblue')
    plt.xlabel("Day of Week")
    plt.ylabel("Total Stream")
    plt.title("Number of Streams by day of week")
    plt.savefig(f"plots/streams_for_day_{name}", bbox_inches='tight')
    plt.show()


def decompose(dataset, model):

    dataset = dataset.iloc[:,1:3]
    tdi = pd.DatetimeIndex(dataset.Date)
    dataset.set_index(tdi, inplace=True)
    dataset.drop(columns='Date', inplace=True)
    dataset.index.name = 'datetimeindex'
    time_series = pd.Series(dataset['Streams'], index=tdi)
    result = seasonal_decompose(time_series, model=model, period=7, extrapolate_trend='freq')

    # Visualizza i componenti decomposti
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    plt.figure(figsize=(8,8))

    plt.subplot(411)
    plt.title(f'{model.capitalize()} decomposition')
    plt.plot(dataset, label='Original', color='skyblue')
    plt.legend(loc='upper left')

    plt.subplot(412)
    plt.plot(trend, label='Trend', color='skyblue')
    plt.legend(loc='upper left')

    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality', color='skyblue')
    plt.legend(loc='upper left')

    plt.subplot(414)
    plt.plot(residual, label='Residuals', color='skyblue')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f"plots/decomposition_{model}", bbox_inches='tight')
    plt.show()



'''streams_by_day(song, 'song')
streams_by_date(song, 'song')'''

streams_by_day(coldplay, 'coldplay')
streams_by_date(coldplay, 'coldplay')


decompose(coldplay,'multiplicative')
decompose(coldplay,'additive')
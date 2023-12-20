import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator

file = 'dataset/coldplay_grouped_by.csv'
coldplay = pd.read_csv(file)

coldplay['Date'] = pd.to_datetime(coldplay['Date'])
coldplay['Day of week'] = coldplay['Date'].dt.dayofweek
coldplay['Day of week'] = coldplay['Day of week'].map({
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
})


def streams_by_date():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot_date(coldplay['Date'], coldplay['Streams'], fmt='g--')  # x = array of dates, y = array of numbers
    fig.autofmt_xdate()
    # For tickmarks and ticklabels every week
    ax.xaxis.set_major_locator(MonthLocator())

    # For tickmarks and ticklabels every other week
    # ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MO, interval=2))

    plt.xlabel("Date")
    plt.ylabel("Streams")
    plt.title("Number of Streams by Date")

    plt.grid(True)
    plt.savefig('plots/streams_by_date.png', bbox_inches='tight')
    plt.show()


def streams_by_day():
    day = coldplay['Day of week'].unique()
    plt.figure(figsize=(10, 5))
    plt.bar(day, coldplay.groupby('Day of week')['Streams'].sum(), width=0.5, color='green')
    plt.xlabel("Day of Week")
    plt.ylabel("Total Stream")
    plt.title("Number of Streams by day of week")
    plt.savefig('plots/streams_for_day.png', bbox_inches='tight')
    plt.show()

streams_by_day()
streams_by_date()
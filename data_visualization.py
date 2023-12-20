import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import WeekdayLocator, MonthLocator
from matplotlib.dates import MO

file = 'dataset/coldplay_grouped_by.csv'
coldplay = pd.read_csv(file)

fig, ax = plt.subplots(figsize=(12,6))
ax.plot_date(coldplay['Date'], coldplay['Streams'], fmt='g--') # x = array of dates, y = array of numbers
fig.autofmt_xdate()
# For tickmarks and ticklabels every week
ax.xaxis.set_major_locator(MonthLocator())

# For tickmarks and ticklabels every other week
#ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MO, interval=2))

plt.xlabel("Date")
plt.ylabel("Streams")
plt.title("Number of Streams by Date")

plt.grid(True)
plt.savefig('plots/Andamento_Streams.png', bbox_inches='tight')
plt.show()



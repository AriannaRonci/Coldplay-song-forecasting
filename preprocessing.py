import pandas as pd

file = 'dataset/coldplay.csv'
coldplay = pd.read_csv(file)
coldplay = coldplay.groupby(['Date'])['Streams'].sum().reset_index()



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

coldplay.to_csv('dataset/coldplay_grouped_by.csv', header=True)


song = pd.read_csv(file)
song = song.loc[song['Track Name'] == 'Fix You']
song = song.groupby(['Date'])['Streams'].sum().reset_index()

song['Date'] = pd.to_datetime(song['Date'])
song['Day of week'] = song['Date'].dt.dayofweek
song['Day of week'] = song['Day of week'].map({
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
})

song.to_csv('dataset/song_grouped_by.csv', header=True)



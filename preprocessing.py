import pandas as pd

file = 'dataset/coldplay.csv'
coldplay = pd.read_csv(file)

coldplay = coldplay.groupby(['Date'])['Streams'].sum().reset_index()

coldplay.to_csv('dataset/coldplay_grouped_by.csv', header=True)
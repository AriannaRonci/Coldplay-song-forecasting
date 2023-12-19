import pandas as pd

file = 'dataset/coldplay.csv'
coldplay = pd.read_csv(file)

coldplay = coldplay.groupby(['Date'])['Streams'].sum().reset_index()

print('ciao')

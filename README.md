# Forecasting con ARIMA e SARIMAX

## Introduzione
Questa repository è reltiva all'implementazione dei modelli ARIMAX e SARIMAX per l'analisi e la previsione di serie temporali. 

## Dataset
Il dataset utilizzato contiene dati relartivi alla classifica giornaliera delle 200 canzoni più ascoltate in 53 Paesi dagli utenti di Spotify nel periodo 2017-2018.
Tale dataset comprende più di 2 milioni di righe che includono 6629 artisti e 18598 canzoni, per un totale di 105 miliardi di streams.
E' scaricabile al seguente [link](https://www.kaggle.com/datasets/edumucelli/spotifys-worldwide-daily-song-ranking).

## Requisiti
- Python 3.6 o versioni successive
- Dipendenze Python: pandas, numpy, statsmodels, matplotlib, sklearn


## Analisi e modelli addestrati
Decomposizione della serie temporale nelle componenti:
- trend
- stagionalità
- residuo

Analisi della stazionarietà e dell'autocorrelazione 
- test statistico Augmented Dickey-Fuller (ADF)
- test statistico Partial AutoCorrelation Function (PACF)

Modelli
I modelli addestrati e i parametri scelti sono i seguenti:
- ARIMA (1, 0, 2)
- SARIMAX (1, 0, 2)x(1, 0, 1)7

Metriche di valutazione dei modelli
Dopo aver addestrato i modelli si sono valutati i rusltati ottenuti da ciascuno di essi utilizzando le seguenti metriche:
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Mean Squared Error (MSE)
- Coefficient of determination (R2)



## Autori
Progetto realizzato da:
- [Chiara Gobbi](https://github.com/chiaragii)
- [Alice Moretti](https://github.com/AliceMoretti00)
- [Federica Parlapiano](https://github.com/FedericaParlapiano)
- [Arianna Ronci](https://github.com/AriannaRonci)


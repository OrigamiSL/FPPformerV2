import pandas as pd
import numpy as np

# Air
air = pd.read_excel('./Air/AirQualityUCI.xlsx', header=0)
air['Date'] = air['Date'].astype('str')
air['Time'] = air['Time'].astype('str')
air['Date'] = air['Date'] + " " + air['Time']
cols = list(air.columns)
cols.remove('Time')
cols.remove('NMHC(GT)')
air = air[cols]
# data -200
cols.remove('Date')
air_values = air[cols].values
mean_list = []
for i in range(air_values.shape[1]):
    values = air_values[:, i]
    mean_list.append(values[values > -200].mean())
mean_list = np.array(mean_list)
mean_list = np.expand_dims(mean_list, axis=0)
mean_list = mean_list.repeat(air_values.shape[0], axis=0)
air_values = np.where(air_values > -200, air_values, mean_list)
df_air = pd.DataFrame(data=air_values, columns=[cols])
df_air.insert(loc=0, column='date', value=air['Date'])
df_air.to_csv('./Air/Air.csv', mode='w', header=True, index=False)


# # River
river = pd.read_csv('./River/RF2.csv')
river = river.iloc[:, :9]
river.rename(columns={"Unnamed: 0": "date"}, inplace=True)
river.to_csv('./River/River.csv', mode='w', header=True, index=False)

# BTC
date_range = pd.date_range(start='2018-05-15 06:00:00', end='2022-03-01 00:00:00', freq='h')
data = pd.DataFrame(np.random.randn(date_range.shape[0], 1), columns=list('a'))
data["date"] = date_range.astype('str')

btc = pd.read_csv('./BTC/BTC-Hourly.csv')
btc.rename(columns={"Unnamed: 0": "date"}, inplace=True)
btc.to_csv('./BTC/BTC.csv', mode='w', header=True, index=False)
res = btc.sort_values(by='date', ascending=True)
cols = list(btc.columns)
cols.remove('date')
cols.remove('unix')
cols.remove('symbol')
res = res[['date'] + cols]
res = res.reset_index(drop=True)

data = pd.merge(data, res, on='date', how = 'left')
cols = list(data.columns)
cols.remove('a')
cols.remove('date')
data = data[cols]
df_eth = pd.DataFrame(data=data.values, columns=[cols])
df_eth.insert(loc=0, column='date', value=date_range)
df_eth.to_csv('./BTC/BTC.csv', mode='w', header=True, index=False)


# # ETH
date_range = pd.date_range(start='2017-08-17 04:00:00', end='2023-10-19 23:00:00', freq='h')
data = pd.DataFrame(np.random.randn(date_range.shape[0], 1), columns=list('a'))
data["Date"] = date_range.astype('str')

file_root = './ETH/'
name = 'Binance_ETHUSDT_1h (1).csv'
names = [p_l for p_l in name.replace('_', ',').split(',')]
data_load = pd.read_csv(file_root + name)
cols = list(data_load.columns)
cols.remove('Date')
cols.remove('tradecount')
cols.remove('Symbol')

res = data_load.sort_values(by='Date', ascending=True)
res = res[['Date'] + cols]
res['Date'] = res['Date'].astype('str')
res = res.reset_index(drop=True)

data = pd.merge(data, res, on='Date', how = 'left')
cols = list(data.columns)
cols.remove('a')
cols.remove('Date')
data = data[cols]

crypto_values = data.values

mean_list = []
for i in range(crypto_values.shape[1]):
    values = crypto_values[:, i]
    values[np.isnan(values)] = np.nanmean(values)
    crypto_values[:, i] = values

df_eth = pd.DataFrame(data=crypto_values, columns=[cols])
df_eth.insert(loc=0, column='date', value=date_range)
df_eth.to_csv('./ETH/ETH.csv', mode='w', header=True, index=False)

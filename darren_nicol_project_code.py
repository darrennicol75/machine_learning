# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Darren Nicol Management Science Project #
# MSc in Financial Technology # 

# Import required libraires and CSV file # 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


dataraw = pd.read_csv("~/Desktop/darren_nicol.csv")

# Set the date as the index

#set date as the index

dataraw['date'] = pd.to_datetime(dataraw['date'], format='%m/%d/%y')
data=dataraw[dataraw['date'] >= '2013-04-28']


# generate returns
data['BTC pct change'] = -1*(1-(data['BTC Price'] / data['BTC Price'].shift(1)))

# Divide data set by year 

dataYear1 =data[(data['date'] >= '2013-04-28') & (data['date'] <= '2013-12-31')]
dataYear2 =data[(data['date'] >= '2014-01-01') & (data['date'] <= '2014-12-31')]
dataYear3 =data[(data['date'] >= '2015-01-01') & (data['date'] <= '2015-12-31')]
dataYear4 =data[(data['date'] >= '2016-01-01') & (data['date'] <= '2016-12-31')]
dataYear5 =data[(data['date'] >= '2017-01-01') & (data['date'] <= '2017-12-31')]
dataYear6 =data[(data['date'] >= '2018-01-01') & (data['date'] <= '2018-12-31')]
dataYear7 =data[(data['date'] >= '2019-01-01') & (data['date'] <= '2019-05-02')]
# create crypto set
dataAllCrypto = data[(data['date'] >= '2017-10-01')]

# continue set date as index
data.set_index('date', inplace=True)
# arrange oldest to newest
data=data.sort_index() 

dataAllCrypto.set_index('date', inplace=True)
dataAllCrypto=dataAllCrypto.sort_index()

# check for missing dates (actual days) in data

pd.date_range(start='2013-04-28', end='2019-05-01').difference(data.index)

# convert to float 

data['Google trends'] = data['Google trends'].astype(float) 

# Fill missing values - NASDAQ, Gold and DJI price on Friday remains same on Sat/Sun 
cols = ['Nasdaq composite index', 'DJI', 'Gold in USD']
data.loc[:,cols] = data.loc[:,cols].ffill()

# Remove Null values

data=data.dropna(subset = ['BTC network hashrate', 'Average BTC block size', 'NUAU - BTC', 'Number TX - BTC', 'Difficulty - BTC', 'TX fees - BTC', 'Estimated TX Volume USD - BTC', 'Gold in USD'])

# Make sure they  have been dropped from data, returns false as all nulls been dropped

data['BTC network hashrate'].isnull().values.any()
data['Average BTC block size'].isnull().values.any()
data['NUAU - BTC'].isnull().values.any()
data['Number TX - BTC'].isnull().values.any()
data['Difficulty - BTC'].isnull().values.any()
data['TX fees - BTC'].isnull().values.any()
data['Estimated TX Volume USD - BTC'].isnull().values.any()
data['Gold in USD'].isnull().values.any()

# Clean the dataAllCrypto data set 

# Remove null values

dataAllCrypto=dataAllCrypto.dropna(subset = ['BTC network hashrate', 'Average BTC block size', 'NUAU - BTC', 'Number TX - BTC', 'Difficulty - BTC', 'TX fees - BTC', 'Estimated TX Volume USD - BTC', 'Gold in USD'])

# Make sure they  have been dropped from data, returns false as all nulls been dropped

dataAllCrypto['BTC network hashrate'].isnull().values.any()
dataAllCrypto['Average BTC block size'].isnull().values.any()
dataAllCrypto['NUAU - BTC'].isnull().values.any()
dataAllCrypto['Number TX - BTC'].isnull().values.any()
dataAllCrypto['Difficulty - BTC'].isnull().values.any()
dataAllCrypto['TX fees - BTC'].isnull().values.any()
dataAllCrypto['Estimated TX Volume USD - BTC'].isnull().values.any()
dataAllCrypto['Gold in USD'].isnull().values.any()



#Fill Google trends incase needed
data['Google_trends_filled'] = data['Google trends']
cols2= ['Google_trends_filled']
data.loc[:,cols2] = data.loc[:,cols2].bfill()
data.loc[:,cols2] = data.loc[:,cols2].ffill()
data['Google_trends_filled'] = data['Google_trends_filled'].astype(float)

dataAllCrypto.loc[:,cols] = dataAllCrypto.loc[:,cols].ffill()
dataAllCrypto['Google_trends_filled'] = dataAllCrypto['Google trends']
dataAllCrypto.loc[:,cols2]= dataAllCrypto.loc[:,cols2].bfill()
dataAllCrypto.loc[:,cols2] = dataAllCrypto.loc[:,cols2].ffill()
dataAllCrypto['Google_trends_filled'] = dataAllCrypto['Google_trends_filled'].astype(float)


# Transfer trends filled data into google trends 

data = data.drop(columns="Google trends") 
dataAllCrypto = dataAllCrypto.drop(columns="Google trends")

# Rename column Google trends

data = data.rename(columns={"Google_trends_filled": "Google trends"})
dataAllCrypto = dataAllCrypto.rename(columns={"Google_trends_filled": "Google trends"})

# Export to excel  

data.to_excel("data021119.xlsx")
dataAllCrypto.to_excel("dataAllCrypto021119.xlsx")

###### Data manipulation complete #######

###### Exploratory Analysis ####### 

# Correlation Analysis Data

corr=data.corr()

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True 
with sns.axes_style("white"):
    p2 = sns.heatmap(corr, mask=mask, square=True)

# Correlation Analysis DataAllCrypto

corrCrypto=dataAllCrypto.corr() 

mask = np.zeros_like(corrCrypto)
mask[np.triu_indices_from(mask)] = True 
with sns.axes_style("white"):
    p2 = sns.heatmap(corrCrypto, mask=mask, square=True) 
    
 
# Line graph  of BTC Price over time 
    
sns.set(rc={'figure.figsize':(11,4)})
alltimebtcprice=data['BTC Price'].plot(linewidth=0.5) 
alltimebtcprice.set_ylabel('BTC Price (USD)') 
alltimebtcprice.set_xlabel('Year') 
alltimebtcprice.set_title('Bitcoin Price (USD) over time')



# Hash rate and difficulty 

x=data['BTC network hashrate'] 
y=data['Difficulty - BTC'] 
plt.scatter(x,y, color='red', alpha=0.5)
plt.title('Bitcoin network hash rate vs. difficulty')
plt.xlabel('Hashrate') 
plt.ylabel('Difficulty')

# BTC Price and Litecoin Price

x=data['BTC Price'] 
y=data['Litecoin Price'] 
plt.scatter(x,y, color='blue', alpha=0.5)
plt.title('Bitcoin Price vs. Litecoin Price')
plt.xlabel('BTC Price') 
plt.ylabel('Litecoin Price')

# Nasdaq and  Gold

x=dataAllCrypto['Nasdaq composite index'] 
y=dataAllCrypto['Gold in USD'] 
plt.scatter(x,y, color='purple', alpha=0.5)
plt.title('Nasdaq v Gold')
plt.xlabel('Nasdaq') 
plt.ylabel('Gold')


# Difficulty vs. Transaction fees 

x=dataAllCrypto['Difficulty - BTC'] 
y=dataAllCrypto['TX fees - BTC'] 
plt.scatter(x,y, color='orange', alpha=0.5)
plt.title('Difficulty vs. TX fees')
plt.xlabel('Difficulty') 
plt.ylabel('TX fees')


# Estimated Transaction Volume (USD) 
y=data['BTC Price'] 
x=data['Estimated TX Volume USD - BTC'] 
plt.scatter(x,y, alpha=0.5) 
plt.title('Bitcoin price and estimated transaction volume (USD)')
plt.xlabel('Estimated transaction volume of Bitcoin (USD)') 
plt.ylabel('Bitcoing price (USD)')

# Line graph of each crypto currencies 

dataAllCrypto.plot(y=['BTC Price', 'Litecoin Price', 'Ethereum Price', 'Cardano Price', 'Bitcoin Cash Price'], figsize=(11,5), grid=True)
plt.title('Cryptocurrency Prices (USD) 2017 - 2019')
plt.xlabel('Date')
plt.ylabel('Price (USD)')

dataAllCrypto.plot(y=['BTC Price'], color='blue', figsize=(11,5), grid=True)
plt.title('Cryptocurrency Prices (USD) 2017 - 2019')
plt.xlabel('Date')
plt.ylabel('Price (USD)') 

dataAllCrypto.plot(y=['Litecoin Price'], color='orange', figsize=(11,5), grid=True)
plt.title('Cryptocurrency Prices (USD) 2017 - 2019')
plt.xlabel('Date')
plt.ylabel('Price (USD)') 

dataAllCrypto.plot(y=['Ethereum Price'], color='green', figsize=(11,5), grid=True)
plt.title('Cryptocurrency Prices (USD) 2017 - 2019')
plt.xlabel('Date')
plt.ylabel('Price (USD)') 

dataAllCrypto.plot(y=['Cardano Price'], color='red', figsize=(11,5), grid=True)
plt.title('Cryptocurrency Prices (USD) 2017 - 2019')
plt.xlabel('Date')
plt.ylabel('Price (USD)') 

dataAllCrypto.plot(y=['BTC Price', 'Google trends'], figsize=(11,5), grid=True)
plt.title('Cryptocurrency Prices (USD) 2017 - 2019')
plt.xlabel('Date')
plt.ylabel('Price (USD)')

dataAllCrypto.plot(y=['Bitcoin Cash Price'], color='purple', figsize=(11,5), grid=True)
plt.title('Cryptocurrency Prices (USD) 2017 - 2019')
plt.xlabel('Date')
plt.ylabel('Price (USD)')

# All data BTC Price

data.plot(y=['BTC Price'], color='blue', figsize=(11,5), grid=True)
plt.title('Cryptocurrency Prices (USD) 2017 - 2019')
plt.xlabel('Date')
plt.ylabel('Price (USD)')


# 2017 - 2019 (dataAllCrypto)

dataAllCrypto.plot(y=['BTC Price', 'Litecoin Price', 'Ethereum Price', 'Cardano Price'], figsize=(11,5), grid=True)
plt.title('Cryptocurrency Prices (USD) 2017 - 2019')
plt.xlabel('Date')
plt.ylabel('Price (USD)') 

# dataAllCrypto.plot(y=['Google trends'])

# Line Graph of bitcoin proce compared to other exchanges 

dataAllCrypto.plot(y=['BTC Price', 'Gold in USD', 'Nasdaq composite index', 'DJI'], figsize=(11,5), grid=True)
plt.title('Bitcoin price compared with other classes')
plt.xlabel('Date') 
plt.ylabel('Price') 

data.plot(y=['BTC Price', 'Bitcoin Cash Price', 'Gold in USD', 'Nasdaq composite index', 'DJI'], figsize=(11,5), grid=True)
plt.title('Bitcoin price compared with other classes')
plt.xlabel('Date') 
plt.ylabel('Price') 

# Line Graph of returns 

data.plot(y=['BTC pct change'], figsize=(11,5), grid=True)
plt.title('Percentage change in BTC over time')
plt.xlabel('Date') 
plt.ylabel('Percentage change in BTC price') 

# Individual Years 

# Examine the variation in price for each year 

annualprice=pd.concat([dataYear7['BTC Price'],dataYear6['BTC Price'],dataYear5['BTC Price'],dataYear4['BTC Price'],dataYear3['BTC Price'],dataYear2['BTC Price'],dataYear1['BTC Price']], axis=1, keys=['7','6','5','4','3','2','1'])

# Plot box plot

annualprice.plot.box(grid='True')

plt.title('Bitcoin Price per Year') 
plt.xlabel('Year')
plt.ylabel('Bitcoin  Price(USD)')

# Create statistics

dataYear7.describe()
dataYear6.describe()
dataYear5.describe()
dataYear4.describe()
dataYear3.describe()
dataYear2.describe()
dataYear1.describe()
 
datatable = data.describe()
dataAllCryptotable = dataAllCrypto.describe()

annualpctchange=pd.concat([dataYear7['BTC pct change'],dataYear6['BTC pct change'],dataYear5['BTC pct change'],dataYear4['BTC pct change'],dataYear3['BTC pct change'],dataYear2['BTC pct change'],dataYear1['BTC pct change']],axis=1, keys=['7','6','5','4','3','2','1'])

#plt.title('Percentage Change in BTC Price (USD) per Year')
#plt.xlabel('Year')
#plt.ylabel('Percentage Change in BTC Price (USD)')

annualpctchange.plot.box(grid='True')

# Sentiment Analysis 

import datetime 
y=data['Google trends']
sns.set(rc={'figure.figsize':(11,4)})
plt.plot(y,'o',color='blue')
plt.ylabel('Number of Google searches (scaled by Google)')
plt.xlabel('Time')
plt.title('Number of worldwide Google searches for Bitcoin from 2013 to 2019')

y=dataAllCrypto['Google trends']
sns.set(rc={'figure.figsize':(11,4)})
plt.plot(y,'o',color='yellow')
plt.ylabel('Number of Google searches (scaled by Google)')
plt.xlabel('Time')
plt.title('Number of worldwide Google searches for Bitcoin from 2013 to 2019')

##########################

# LSTM

conda install keras==2.1.2
conda install tensorflow==1.4.0

import tensorflow 

import keras 

import requests

from keras.models import Sequential 
from keras.layers import Activation, Dense, Dropout, LSTM 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import seaborn as sns 
from sklearn.metrics import mean_absolute_error

from keras.callbacks import CSVLogger 
from keras.layers import LSTM, Dropout, Dense 

import os

from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 

# Splitting data into training and testing data 

dataset=data[["BTC Price"]].values 
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset) 
df_train, df_test = train_test_split(dataset, train_size=0.8, test_size=0.2, shuffle=False)
print("Train and Test Size", len(df_train), len(df_test))

# Train and Test Size 1724 and 431 

train_size = int(len(dataset) * 0.80) 
test_size = len(dataset)
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

######

# Convert an array into dataset  matrix 

def create_dataset(dataset, look_back=1): 
    dataX, dataY = [], []
    for i in range (len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a) 
        dataY.append(dataset[i + look_back, 0])
        
    return np.array(dataX), np.array(dataY)

# Set the look back to achieve the lowest error (between 1-10)

look_back = 7 

trainX, trainY = create_dataset(train, look_back) 

testX, testY = create_dataset(test, look_back)  

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))  


model = Sequential() 

model.add(LSTM(4, input_shape=(1, look_back)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=500, batch_size=1, verbose=2)

trainPredict = model.predict(trainX) 

testPredict = model.predict(testX)         
        

# Invert predictions

trainPredict = scaler.inverse_transform(trainPredict) 

trainY = scaler.inverse_transform([trainY]) 

testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])

# Calculate root mean squared error

trainScore = (mean_squared_error(trainY[0], trainPredict[:,0]))

print('Train Score: %.2f MSE' % (trainScore))

testScore = (mean_squared_error(testY[0], testPredict[:,0]))

print('Test Score: %.3f MSE' % (testScore))

# shift train predictions for plottting 

trainPredictPlot = np.empty_like(dataset)

trainPredictPlot[:,:] = np.nan 

trainPredictPlot[look_back:len(trainPredict)+look_back,:] = trainPredict

# shift test predictions for plotting 

testPredictPlot = np.empty_like(dataset)

testPredictPlot[:,:] = np.nan 

testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1,:] =testPredict 

# plot baseline and predictions 

plt.figure(figsize= (19,10)) 

plt.plot(scaler.inverse_transform(dataset), label = "Bitcoin Price")

plt.plot(testPredictPlot, label = "Test Prediction") 

plt.plot(trainPredictPlot, label = "Train Prediction") 

#plt.plot('BTC Price', label = "Bitcoin Price") 

plt.grid(True) 

plt.legend(loc='upper  right')

plt.title("Bitcoin Price Prediction Model- LSTM")

plt.xlabel("Number of Days") 

plt.ylabel("Price (USD)") 

#PLT LEGEND() ADD LEGEND!! #####################

plt.show()

# Choose regression 

from sklearn.model_selection import cross_val_score 

from sklearn.linear_model import LinearRegression 

independent_variables = ['BTC network hashrate', 'Average BTC block size', 'NUAU - BTC', 'Difficulty - BTC', 'Number TX - BTC', 'Estimated TX Volume USD - BTC', 'Gold in USD', 'Ethereum Price', 'Litecoin Price', 'Bitcoin Cash Price', 'Cardano Price', 'Nasdaq composite index', 'DJI']

Xs= dataAllCrypto[independent_variables] 

Xs.isnull().values.any() 

y = dataAllCrypto['BTC Price'].values.reshape(-1,1) 

# Normal Regression 

lin_reg = LinearRegression() 

MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=5)
mean_MSE = np.mean(MSEs)
print(mean_MSE) 

# Ridge Regression

from sklearn.model_selection import GridSearchCV 

from sklearn.linear_model import Ridge 

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2,1,5,10,20]}

ridge_regressor=GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(Xs,y)

print(ridge_regressor.best_params_)

print(ridge_regressor.best_score_) 

# pick one with lwest score and lowest MSE 

# Lasso Regression 

from sklearn.linear_model import Lasso 

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2,1,5,10,20]}

lasso_regressor=GridSearchCV(lasso, parameters,scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(Xs,y)

print(lasso_regressor.best_params_)

print(lasso_regressor.best_score_) 

########    END OF CODE   ########  

# Darren Nicol- Management Science Project for  MSc in Financial Technology #
        
        
        
        



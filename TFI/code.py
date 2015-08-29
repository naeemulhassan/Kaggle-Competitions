import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from datetime import datetime

data = pd.read_csv('/home/naeemul/Dropbox/Kaggle/TFI/train.csv')


Period = np.zeros((len(data), 1))
d1 = datetime.strptime('04/15/2015', "%m/%d/%Y")
for i in range(0,len(data)):
    d2 = datetime.strptime(data['Open Date'][i], "%m/%d/%Y")
    Period[i] = (d1 - d2).days

data['Period'] = Period
data = pd.concat([data, data.Type.str.get_dummies()], axis = 1)
#data = pd.concat([data, data.City.str.get_dummies()], axis = 1)
data = pd.concat([data, data['City Group'].str.get_dummies()], axis = 1)
data['MB'] = np.zeros((len(data), 1))

features = data.columns[5:42]|data.columns[43:len(data.columns)]

mask = np.random.rand(len(data)) < 0.8
train, test = data[mask], data[~mask]

clf = RandomForestRegressor(n_estimators = 100)
clf.fit(train[features], train.revenue)
rfr_prediction = clf.predict(test[features])
rfr_mse = np.sqrt(sum((rfr_prediction - test.revenue) ** 2))

del data['MB']
features = data.columns[5:42]|data.columns[43:len(data.columns)]
data_norm = (data[features] - data[features].mean()) / (data[features].max() - data[features].min())
data['MB'] = np.zeros((len(data), 1))

train, test = data[mask], data[~mask]

clf = linear_model.LinearRegression()
clf.fit(train[features], train.revenue)
lr_prediction = clf.predict(test[features])
lr_mse = np.sqrt(sum((lr_prediction - test.revenue) ** 2))

rfr_lr_prediction = (rfr_prediction+lr_prediction)/2
rfr_lr_mse = np.sqrt(sum((rfr_lr_prediction - test.revenue) ** 2))

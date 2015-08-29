import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from datetime import datetime

def postprocess(data):
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
	return data

train = pd.read_csv('/home/naeemul/Dropbox/Kaggle/TFI/train.csv')
test = pd.read_csv('/home/naeemul/Dropbox/Kaggle/TFI/test.csv')

train = postprocess(train)
test = postprocess(test)
features = train.columns[5:42]|train.columns[43:len(train.columns)]

clf = RandomForestRegressor(n_estimators = 100)
clf.fit(train[features], train.revenue)
rfr_prediction = clf.predict(test[features])

file = open('RF_2.csv','w')
file.write('Id,Prediction\n')
for i in range(0,len(rfr_prediction)):
	file.write(str(i)+','+str(rfr_prediction[i])+'\n')
	
file.close()

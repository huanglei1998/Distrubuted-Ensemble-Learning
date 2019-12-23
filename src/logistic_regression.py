
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

dataframe = pd.read_csv("phoneme.dat",header=None,skiprows=[0,1],sep='\s+')
data=dataframe.values
random.shuffle(data)

for i in range(data.shape[0]):
    if(data[i,-1]==0):
        data[i,-1]=-1

train_data=data[0:5000,:]
test_data=data[5000:,:]
train_x=train_data[:,0:-1]
train_y=train_data[:,-1]
test_x=test_data[:,0:-1]
test_y=test_data[:,-1]


# train_index_list=[]
# test_index_list=[]
# kf=KFold(n_splits=5,shuffle=False)
# for train_index,test_index in kf.split(train_x):
#     train_index_list.append(train_index)
#     test_index_list.append(test_index)
#
# print(test_index_list[3])
#
# train_data=train_x[test_index_list[3],:]

w = np.ones(train_y.shape)
weight = w / train_y.shape[0]

estimator = AdaBoostClassifier()
estimator.fit(train_x,train_y,sample_weight=weight)

predict_y=estimator.predict(test_x)
score=estimator.score(test_x,test_y)
print(score)





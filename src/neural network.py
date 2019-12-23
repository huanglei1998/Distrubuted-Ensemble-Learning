from sklearn.neural_network import MLPClassifier

import pandas as pd
import numpy as np
import random

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

w = np.ones(train_y.shape)
weight = w / train_y.shape[0]

def select(weight):
    num_=[i for i in range(train_x.shape[0])]
    #概率列表
    sum_ = 0
    ran = random.random()
    for num, r in zip(num_, weight):
        sum_ += r
        if ran < sum_ :break
    return num



train_x_weight = []
train_y_weight = []
for a in range(train_x.shape[0]):
    num = select(weight)
    train_x_weight.append(train_x[num, :])
    train_y_weight.append(train_y[num])
train_x_weight = np.array(train_x_weight)
train_y_weight = np.array(train_y_weight)

estimator = MLPClassifier()
estimator.fit(train_x_weight, train_y_weight)


mlp = MLPClassifier()
mlp.fit(train_x,train_y)

predict_y=mlp.predict(test_x)
score=mlp.score(test_x,test_y)
print(score)






# -*- coding: utf-8 -*-
'''
  The chapter 3 Linear Models for Regression
  The code of Linear Basis Function Models 
'''

# -*- coding: utf-8 -*-
'''
The code is to more deeply understand the PRML Chapter 3 about LinearRegression
The data is diabetes patient set from sklearn
Part 1 is normal kernel linear regression. 
Part 2 is regual kernel linear regression
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,cross_validation

def load_data():
    #data_train,data_predict,train_lab,predict_lab four data sets
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data,diabetes.target,test_size=0.20,random_state=0)


def LinearRegression(dt,dp,tl,pl,kernelfunction):
    design_matrix = []
    if kernelfunction == 'gauss': #supposing standard gauss distribution
        for data in dt:
            tem = []
            for i in range(len(data)):
                if i == 0 :
                    tem.append(np.exp(-(data[i] - 0.0294638316759)**2 / (2*0.11170534249)))
                if i == 1 :
                    tem.append(np.exp(-(data[i] - 0.0636844215484)**2 / (2*0.107140579897)))
                if i == 2:
                    tem.append(np.exp(-(data[i] + 0.0245251016585) ** 2 / (2*0.116465058849)))
                if i == 3:
                    tem.append(np.exp(-(data[i] + 0.023323761252) ** 2 / (2*0.1101685293)))
                if i == 4:
                    tem.append(np.exp(-(data[i] + 0.0178207686246) ** 2 / (2*0.110054133791)))
                if i == 5:
                    tem.append(np.exp(-(data[i] + 0.0311765153256) ** 2 / (2*0.110768000707)))
                if i == 6:
                    tem.append(np.exp(-(data[i] + 0.0076677187343) ** 2 / (2*0.109714460455)))
                if i == 7:
                    tem.append(np.exp(-(data[i] - 0.014989786312) ** 2 / (2*0.111275377915)))
                if i == 8:
                    tem.append(np.exp(-(data[i] - 0.00106815786274)**2 / (2*0.115277205067)))
                if i == 9:
                    tem.append(np.exp(-(data[i] + 0.00656270921372)** 2 / (2*0.116974959819)))

            design_matrix.append(tem)
        design_matrix = np.mat(design_matrix)
        t = np.mat(tl[:,np.newaxis])
        w = (design_matrix.T*design_matrix).I * design_matrix.T * t
        return w

    if kernelfunction == 'logic':
        pass

def stastic(dt):
    count = 0.0
    for data in dt:
        for i in range(len(data)):
            if i == 9:
                count += (data[i]- 0.0294638316759)**2 #0:0.0294638316759 1:0.0636844215484 2:-0.0245251016585 3:-0.023323761252 4:-0.0178207686246 5:-0.0311765153256 6:-0.0076677187343 7:0.014989786312 8:0.00106815786274 9:-0.00656270921372

    print count/len(data)


if __name__=='__main__':
    dt,dp,tl,pl=load_data()
    LinearRegression(dt,dp,tl,pl,'gauss')
    #stastic(dt)
    #print np.mat(tl[:,np.newaxis]).shape

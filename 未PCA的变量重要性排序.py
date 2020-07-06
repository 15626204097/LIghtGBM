# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:21:16 2018

@author: Administrator
"""

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc    

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.datasets import  make_classification
from sklearn.decomposition import PCA
from sklearn.cross_validation import PredefinedSplit
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
# 加载数据
row_data = pd.read_csv('D:/Pycharm/P2P/test280014.csv',header=None)
row_data.columns = ['id','target','loan_money','year_rate','qua_time','gender',
                    'age','education','learn_means','success_loan','success_payback','success_allpay',
                    'accumulate_loan','need_pay','per_max','history_max_debat']
new = pd.get_dummies(row_data,prefix=['gender_oht','educaiton_oht','learn_oht'],columns=['gender','education','learn_means'])#离散变量处理
new = new.drop('id',axis=1)#去掉第二列编号
feature_data = new.drop('target',axis=1)#去掉第2列分类

def standard(column):#归一化量表 浮点型数据处理
    feature_data[column] = feature_data.apply(lambda x:(x-x.mean())/x.std())[column]
float_variable = ['loan_money','year_rate','qua_time','age','success_loan','success_payback','success_allpay','accumulate_loan','need_pay','per_max','history_max_debat']
for i in float_variable:
    standard(i)

x_train,x_test,y_train,y_test = train_test_split(feature_data,new['target'],test_size=0.25)#测试集和训练集1-3


X_train = x_train
X_test = x_test

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {}
#'metric': 'binary_logloss',    #'num_iterations':500,

#缺分类精度和F1
clf4 = lgb.LGBMClassifier( )#xgboost添加参数优化
model4 = clf4.fit(x_train,y_train)
y_hat4 = model4.predict(x_test)
sum(y_hat4 == y_test)/y_test.count()
c = confusion_matrix(y_hat4,y_test)
acc1 = c[0,0]/sum(c[0,:])
acc2 = c[1,1]/sum(c[1,:])
print('4-1:%.2f%%'%(acc1*100))
print('4-2:%.2f%%'%(acc2*100))
r4 = clf4.score(x_test,y_test) #评估模型准确率  
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

'''eval_metric="RMSE" eval_metric="logloss"

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
'''
eval_set = [(X_test, y_test)]
model4.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="Rlogloss", eval_set=eval_set, verbose=True)

#Stopping. Best iteration:
from xgboost import plot_importance
from matplotlib import pyplot

#model.fit(X, y)

plot_importance(model)
pyplot.show()
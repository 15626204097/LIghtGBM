from __future__ import division
# import tensorflow as tf
import math
import csv
from sklearn import metrics
import numpy as np
from pylab import*
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.datasets import  make_classification
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
'''
B本粒子是用auc来寻找变量重要性排序的
'''
# 加载数据
row_data = pd.read_csv('test280014new.csv',header=None)
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



#原生形式使用lightgbm
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict

params = {}#字典调参

                 
#'metric': 'binary_logloss',    #'num_iterations':500,
clf4 = lgb.LGBMClassifier( )#xgboost添加参数优化
model4 = clf4.fit(X_train,y_train)
y_hat4 = model4.predict(X_test)

clf4.fit(X_train, y_train)

 #变量排序不对   
X, y = X_train, y_train   
df =feature_data
df.columns = ['loan_money','year_rate', 'qua_time', 'age', 'success_loan', 'success_payback',
       'success_allpay', 'accumulate_loan', 'need_pay', 'per_max',
       'history_max_debat', 'gender_oht_1', 'gender_oht_2', 'educaiton_oht_1',
       'educaiton_oht_2', 'educaiton_oht_3', 'educaiton_oht_4',
       'educaiton_oht_5', 'learn_oht_1', 'learn_oht_2', 'learn_oht_3',
       'learn_oht_4', 'learn_oht_5', 'learn_oht_6']

feat_labels = df.columns[0:]#懂第一列开始
importances = clf4.feature_importances_
indices = np.argsort(importances)[::-1]#从大到小排序
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))
    


#输出图形分析表


plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],#垂直柱状图barh
       color="r",  align="center")
plt.xticks(range(X_train.shape[1]),feat_labels,rotation=90)
plt.xlim=([-1,X_train.shape[1]])
plt.tight_layout()
#plt.plot(datax,data0, 'k', label='真实值数据点')
#plt.xlim([-1, X.shape[1]])
plt.show()

#优化前的LGB-RMSE越小越好 线性回归表示拟合能力
# 测试机预测              
gbm = lgb.LGBMRegressor()
gbm.fit(X_train, y_train,eval_set=[(X_test, y_test)],eval_metric='l1',early_stopping_rounds=100)
# 模型评估
print('Start predicting...')
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)


                                         









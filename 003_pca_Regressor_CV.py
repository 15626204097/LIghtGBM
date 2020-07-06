# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 15:18:34 2018

@author: Administrator
"""
#主城分新分析+10折交叉验证+生成所需要的数据
#PCA
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
from sklearn.decomposition import PCA
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
#主成分分析
pca = PCA(n_components=15)
pca_data = pca.fit_transform(feature_data)#重新采用PCA后的特征值预测
sum(pca.explained_variance_ratio_)#15特征值代表的总权值是0.99
#分割数据
x_train,x_test,y_train,y_test = train_test_split(pca_data,new['target'],test_size=0.25)

X_train = x_train
X_test = x_test

#lightGBM原生形式使用lightgbm

lgb_train = lgb.Dataset(x_train,y_train)
lgb_eval = lgb.Dataset(x_test,y_test,reference=lgb_train)

#需要交叉验证？？？？？？？？？？？？
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=10)#10折交叉验证
#y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
#print(y_pred)



   
clf4 = lgb.LGBMClassifier( )#xgboost添加参数优化

model4 = clf4.fit(x_train,y_train)
y_hat4 = model4.predict(x_test)
sum(y_hat4 == y_test)/y_test.count()
c = confusion_matrix(y_hat4,y_test)
acc1 = c[0,0]/sum(c[0,:])
acc2 = c[1,1]/sum(c[1,:])
print('4-1:%.2f%%'%(acc1*100))
print('4-2:%.2f%%'%(acc2*100))
print('confuse_matrix')
print(c)

 ### 保存模型
from sklearn.externals import joblib
joblib.dump(gbm,'gbm.pkl')
#线性回归+
gbm = lgb.LGBMRegressor()
gbm.fit(X_train, y_train,eval_set=[(X_test, y_test)],eval_metric='l1',early_stopping_rounds=10)
# 测试机预测
print('Start predicting...')
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
print(y_pred)
# 模型评估_优化后的RMSE
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# 网格搜索，参数优化  线性回归做估计优化
#estimator = lgb.LGBMRegressor(num_leaves=31)
# cv调节50是最优值 
param_grid = {  
        'max_depth': range(5,15,2),  
        'num_leaves': range(10,40,5),  
    } 
estimator = lgb.LGBMRegressor(  num_leaves = 50,   max_depth = 13,   learning_rate =0.1,    n_estimators = 1000,    objective = 'regression',     min_child_weight = 1,   subsample = 0.8,   colsample_bytree=0.8,   nthread = 7,   )  
    
#estimator = LGBMRegressor(  num_leaves = 50,   max_depth = 13,   learning_rate =0.1,    n_estimators = 1000,    objective = 'regression',     min_child_weight = 1,   subsample = 0.8,   colsample_bytree=0.8,   nthread = 7,   )  
    

gbm = GridSearchCV(estimator, param_grid)#,scoring='roc_auc', cv=10 10折交叉

gbm.fit(X_train, y_train)

print('Best parameters found by grid search are:', gbm.best_params_)
print(gbm.grid_scores_)
#print(gbm.best_scores_)    










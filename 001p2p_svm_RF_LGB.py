# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:50:59 2018

@author: Administrator
"""

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
#from sklearn.cross_validation import PredefinedSplit
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
# 加载数据
row_data = pd.read_csv('test280014new.csv',header=None)
row_data.columns = ['id','target','loan_money','year_rate','qua_time','gender',
                    'age','education','learn_means','success_loan','success_payback','success_allpay',
                    'accumulate_loan','need_pay','per_max','history_max_debat']
new = pd.get_dummies(row_data,prefix=['gender_oht','educaiton_oht','learn_oht'],columns=['gender','education','learn_means'])#离散变量处理
print("new")
print(new)

new = new.drop('id',axis=1)#去掉第二列编号
print("new1")
print(new)
feature_data = new.drop('target',axis=1)#去掉第2列分类

def standard(column):#归一化量表 浮点型数据处理
    feature_data[column] = feature_data.apply(lambda x:(x-x.mean())/x.std())[column]
float_variable = ['loan_money','year_rate','qua_time','age','success_loan','success_payback','success_allpay','accumulate_loan','need_pay','per_max','history_max_debat']
for i in float_variable:
    standard(i)

x_train,x_test,y_train,y_test = train_test_split(feature_data,new['target'],test_size=0.25)#测试集和训练集1-3

#缺一个F1的检验 或者查全率 查准率
                                                
                                                
                                                
clf = SVC() # SVM
model = clf.fit(x_train,y_train)
y_hat = model.predict(x_test)
sum(y_hat == y_test)/y_test.count()
a = confusion_matrix(y_hat,y_test)

print("a")
print(a)

acc1 = a[0,0]/sum(a[0,:])
acc2 = a[1,1]/sum(a[1,:])


#acc3= sum(a[0,0]+a[1,1])/(sum(a[0,]+sum(a[1,:))
print('SVM...')
print('1-1:%.2f%%'%(acc1*100))
print('1-2:%.2f%%'%(acc2*100))
r1 = clf.score(x_test,y_test) #评估模型准确率  
print('分类精度:%.2f%%'%(r1*100))#

#print('confuse_matrix')
#print(a)输出混淆矩阵
y_true=y_test
y_pred=y_hat
f1=metrics.f1_score(y_test, y_hat, average='weighted')  
print('F1:%.2f%%'%(f1*100))



clf2 = RandomForestClassifier() # Randomforest
# clf2 = RandomForestClassifier(n_estimators=50, criterion='gini', max_features='sqrt', max_depth=5, min_samples_split=2,
#                             bootstrap=True, n_jobs=1, random_state=1)
model2 = clf2.fit(x_train, y_train)
y_hat2 = model2.predict(x_test)
print("y_hat2:::::")
print(y_hat2)
sum(y_hat2 == y_test)/y_test.count()
b = confusion_matrix(y_hat2, y_test)
acc1 = b[0, 0]/sum(b[0, :])
acc2 = b[1, 1]/sum(b[1, :])

print('RF...')
print('2-1:%.2f%%'%(acc1*100))#%%表示文字格式 %+输出的数据
print('2-2:%.2f%%'%(acc2*100))
r2 = clf2.score(x_test,y_test) #评估模型准确率  
print('分类精度:%.2f%%'%(r2*100))#
#print('confuse_matrix')
#print(b)输出混淆矩阵
f1=metrics.f1_score(y_test, y_hat2, average='weighted')  
print('F1:%.2f%%'%(f1*100))
      

clf3 = XGBClassifier() #xgboost
model3 = clf3.fit(x_train,y_train)
y_hat3 = model3.predict(x_test)
sum(y_hat3 == y_test)/y_test.count()
c = confusion_matrix(y_hat3,y_test)
acc1 = c[0,0]/sum(c[0,:])
acc2 = c[1,1]/sum(c[1,:])

print('XGB...')
print('3-1:%.2f%%'%(acc1*100))
print('3-2:%.2f%%'%(acc2*100))
r3 = clf3.score(x_test,y_test) #评估模型准确率  

print('分类精度:%.2f%%'%(r3*100))# 
#print('confuse_matrix')
#print(c)输出混淆矩阵
f1=metrics.f1_score(y_test, y_hat3, average='weighted')  
print('F1:%.2f%%'%(f1*100))




# create dataset for lightgbm
X_train = x_train
X_test = x_test


lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {}
#'metric': 'binary_logloss',    #'num_iterations':500,

#缺分类精度和F1
clf4 = lgb.LGBMClassifier( )#xgboost添加参数优化
model4 = clf4.fit(x_train, y_train)
y_hat4 = model4.predict(x_test)
sum(y_hat4 == y_test)/y_test.count()
c = confusion_matrix(y_hat4, y_test)
acc1 = c[0, 0]/sum(c[0, :])
acc2 = c[1, 1]/sum(c[1, :])
#print('4-1:%.2f%%'%(acc1*100))
#print('4-2:%.2f%%'%(acc2*100))
r4 = clf4.score(x_test, y_test) #评估模型准确率


#print('confuse_matrix')
#print(c)输出混淆矩阵
validation_x = x_test
validation_y = y_test
r = clf4.score(validation_x,validation_y) #评估模型准确率   
#print('分类精度r4:%.2f%%'%(r4*100))#

f1 = metrics.f1_score(y_test, y_hat4, average='weighted')
#print('F1:%.2f%%'%(f1*100))    


gbm = lgb.LGBMRegressor()
gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=10)
# 模型评估
print('Start predicting...')
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)#迭代寻优了
#print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

 
     
train_x = x_train
train_y = y_train
validation_x = x_test
validation_y = y_test
#未做优化生成的auc图
clf4 = lgb.LGBMClassifier()#xgboost添加参数优化 
s = clf4.fit(train_x, train_y) # 训练模型
r = clf4.score(validation_x,validation_y) #评估模型准确率  

predict_y_validation = clf4.predict(validation_x)#直接给出预测结果，每个点在所有label的概率和为1，内部还是调用predict——proba()  
# print(predict_y_validation)  
prob_predict_y_validation = clf4.predict_proba(validation_x)#给出带有概率值的结果，每个点所有label的概率和为1  
predictions_validation = prob_predict_y_validation[:, 1]  
fpr, tpr, _ = roc_curve(validation_y, predictions_validation,pos_label=2)  ###计算真正率和假正率 
roc_auc = auc(fpr, tpr)  #计算auc的值
print('LGBM...')
print('4-1:%.2f%%'%(acc1*100))
print('4-2:%.2f%%'%(acc2*100))
print('分类精度:%.2f%%'%(r4*100))# 
f1=metrics.f1_score(y_test, y_hat4, average='weighted')  
print('F1:%.2f%%'%(f1*100))
#print('RMSE-yhat:', mean_squared_error(y_test, y_hat4) ** 0.5)             
print('RMSE-最优:', mean_squared_error(y_test, y_pred) ** 0.5)   #迭代寻优后REMS更小                                               
print('AUC:%.2f%%'%(roc_auc * 100))#

plt.figure()   
lw = 2  
plt.figure(figsize=(10,10))  
plt.plot(fpr, tpr, color='darkorange',  
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线  
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Receiver operating characteristic example')  
plt.legend(loc="lower right")  
plt.show()  

     
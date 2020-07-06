import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import roc_curve, auc  
import pickle 
from scipy import interp  
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
from sklearn.cross_validation import PredefinedSplit
from sklearn.metrics import mean_absolute_error
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

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'auc'},
    'num_leaves': 60,
    'learning_rate': 0.05, 'scale_pos_weight':1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0 ,'min_sum_hessian_in_leaf':100
}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=10)#10折交叉验证
#y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
#print(y_pred)

# specify your configurations as a dict

#'metric': 'binary_logloss',    #'num_iterations':500,
clf4 = lgb.LGBMClassifier()#xgboost添加参数优化

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

train_x = x_train
train_y = y_train
validation_x = x_test
validation_y = y_test



clf = lgb.LGBMClassifier()#xgboost添加参数优化 
s = clf.fit(train_x, train_y) # 训练模型  
r = clf.score(validation_x,validation_y) #评估模型准确率  
print ('分类精度:%.2f%%'%r)  
predict_y_validation = clf.predict(validation_x)#直接给出预测结果，每个点在所有label的概率和为1，内部还是调用predict——proba()  
# print(predict_y_validation)  
prob_predict_y_validation = clf.predict_proba(validation_x)#给出带有概率值的结果，每个点所有label的概率和为1  
predictions_validation = prob_predict_y_validation[:, 1]  
fpr, tpr, _ = roc_curve(validation_y, predictions_validation)  ###计算真正率和假正率 
    #  
roc_auc = auc(fpr, tpr)  #计算auc的值
print(roc_auc)
#画图roc
from sklearn.tree import DecisionTreeClassifier  
from matplotlib.pyplot import *  
from sklearn.cross_validation import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.externals.joblib import Parallel, delayed  
from sklearn.tree import export_graphviz 
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
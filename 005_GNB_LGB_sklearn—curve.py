#高斯贝叶斯分类和LGB分类测试

from sklearn.naive_bayes import GaussianNB
import numpy as np
# from sklearn.learning_curve import learning_curve #c查看是否过拟合
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import roc_curve, auc  
import pickle 
from scipy import interp  
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.datasets import  make_classification
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
# 加载数据
row_data = pd.read_csv('/Users/a/Desktop/P2P/test280014.csv',header=None)
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

def plot_learning_curve(estimator, title, X, y, ylim=None,  n_jobs=1, train_sizes=[100,300,600,1000,1886],cv=10, verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"train_sample")
        plt.ylabel(u"score")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"train_score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"cross_validation_score")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()
        plt.savefig("learn_curve.jpg") 

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff
if __name__=='__main__':

    
    X=x_train  #c对训练集进行拟合判断
    y=y_train

    clf4 = lgb.LGBMClassifier()#xgboost添加参数优化
    Gmodel=GaussianNB()
    train_sizes, train_scores, test_scores=learning_curve(model4,X,y,train_sizes=[100,300,600,1000,1886],cv=10)
    plot_learning_curve(model4, u"learning curve", X, y)
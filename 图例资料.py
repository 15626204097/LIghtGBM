#画图roc
plt.figure()  
lw = 4 
plt.figure(figsize=(10,10))  
plt.plot(fpr, tpr, color='darkorange',  
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线  
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('False Positive Rate',fontsize=20)  
plt.ylabel('True Positive Rate',fontsize=)  
plt.title('ROC Space',fontsize=20)  
plt.legend(loc="lower right")  
# 设置刻度字体大小
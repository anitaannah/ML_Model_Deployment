#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN
import warnings
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
warnings.filterwarnings("ignore")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[3]:


df=pd.read_csv(r'C:\Users\HP\Downloads\final_dataset.csv')


# In[4]:


df.drop(['Unnamed: 0','Instrumentalness'],axis=1,inplace=True)


# In[5]:


df.head()


# In[6]:


from scipy import stats
df['Liveness']=np.log(df['Liveness'])
df['Speechiness']=stats.boxcox(df['Speechiness'])[0]
df['Acousticness']=stats.boxcox(df['Acousticness'])[0]


# In[7]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[8]:


counter=Counter(y)
print('before',counter)
smt=SMOTETomek()
balanced_x,balanced_y=smt.fit_resample(x,y)
counter=Counter(balanced_y)
print('after',counter)


# In[9]:


x_train,x_test,y_train,y_test=train_test_split(balanced_x,balanced_y,test_size=0.2)


# In[10]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
  
scaler=StandardScaler()
for i in ['Danceability','Energy','Loudness','Speechiness','Acousticness','Liveness','Valence','Tempo','Genre']:
    scaler.fit(x_train[[i]])
    x_train[i]=scaler.transform(x_train[[i]])
    x_test[i]=scaler.transform(x_test[[i]])


# In[11]:


scaled_x_train=scaler.fit_transform(x_train)
scaled_x_test=scaler.transform(x_test)


# In[12]:


counter=Counter(y_train)
print('before',counter)
smt=SMOTETomek()
balanced_x_train,balanced_y_train=smt.fit_resample(scaled_x_train,y_train)
counter=Counter(balanced_y_train)
print('after',counter)


# In[13]:


from sklearn.tree import DecisionTreeClassifier


# In[14]:


dt = DecisionTreeClassifier(max_depth=3)
dt.fit(x_train, y_train)


# In[15]:


from sklearn import tree
import matplotlib.pyplot as plt


# In[16]:


Model=DecisionTreeClassifier()
Model.fit(scaled_x_train,y_train)
pred=Model.predict(scaled_x_test)
acc=accuracy_score(y_test,pred)
print(acc)


# In[17]:


## Hyper parameter tuning using randomized search cross validation
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


# In[18]:


param_dist={"criterion":["gini","entropy"],"max_depth":[1,2,3,4,5,6,7,8,9,10,11]}
grid=RandomizedSearchCV(DecisionTreeClassifier(),param_distributions=param_dist,cv=5,n_jobs=-1)
grid.fit(scaled_x_train,y_train)


# In[19]:


grid.best_estimator_


# In[20]:


grid.best_score_


# In[21]:


final_model=grid.best_estimator_


# In[22]:


y_pred=final_model.predict(scaled_x_test)


# In[23]:


print('confusion matrix')
print(confusion_matrix(y_test,y_pred))


# In[24]:


202/(106+202)


# In[25]:


print('classification report')
print(classification_report(y_test,y_pred))


# In[26]:


acc=accuracy_score(y_test,y_pred)
print(acc)


# In[27]:


#plotting ROC curve

from sklearn import metrics
metrics.plot_roc_curve(final_model,scaled_x_test,y_test)
plt.show()


# In[28]:



import joblib


# In[29]:


from sklearn.ensemble import AdaBoostClassifier
boost=AdaBoostClassifier(base_estimator=final_model)
boost.fit(scaled_x_train,y_train)
y_predict=boost.predict(scaled_x_test)


# In[30]:


#plotting ROC curve
print('score: ', boost.score)
print()
from sklearn import metrics
metrics.plot_roc_curve(boost,scaled_x_test,y_test)
plt.show()


# In[31]:


print('classification report')
print(classification_report(y_test,y_predict))


# In[32]:


print(confusion_matrix(y_test,y_predict))


# In[ ]:


param_dist={'n_estimators':[40,50,60,70,80], 'learning_rate':[0.04,0.03,0.02,0.1],'algorithm':['SAMME', 'SAMME.R']}
grid_1=RandomizedSearchCV(boost,param_distributions=param_dist,cv=5,n_jobs=-1)
grid_1.fit(scaled_x_train,y_train)


# In[ ]:


#plotting ROC curve
print('score: ', grid_1.best_score_)
print()
print('ROC-AUC curve')
from sklearn import metrics
metrics.plot_roc_curve(grid_1,scaled_x_test,y_test)
plt.show()


# In[ ]:


predict=grid_1.predict(scaled_x_test)


# In[ ]:


accuracy_score(y_test, predict)


# In[ ]:


print('classification report')
print(classification_report(y_test,predict))


# In[ ]:


boosted_model=grid_1.best_estimator_


# In[ ]:


grid_1.best_params_


# In[ ]:


print('confusion_matrix')
print(confusion_matrix(y_test,predict))


# In[ ]:


import pickle
Pkl_Filename = "Boosted_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(boosted_model, file)


# In[ ]:


import joblib
joblib.dump(boosted_model, 'decision_tree_boosted.pkl')


# In[ ]:


import joblib
model = joblib.load('decision_tree_boosted.pkl') 
print(model)


# In[ ]:


lda=LinearDiscriminantAnalysis()
lda.fit(scaled_x_train,y_train)
y_pred=lda.predict(scaled_x_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:


parameters={'solver':['svd', 'lsqr', 'eigen'],'shrinkage':['auto','None']}
grid=RandomizedSearchCV(LinearDiscriminantAnalysis(store_covariance=True),param_distributions=parameters,cv=5,n_jobs=-1)
grid.fit(scaled_x_train,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


lda_model=grid.best_estimator_


# In[ ]:


y_pred=lda.predict(scaled_x_test)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


#Plotting ROC curve for LDA

metrics.plot_roc_curve(lda_model,scaled_x_test,y_test)
plt.show()


# In[ ]:


import joblib
joblib.dump(lda_model, 'LDA.pkl')


# In[53]:


pip install streamlit


# In[54]:


import streamlit as st


# In[ ]:





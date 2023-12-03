#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[65]:


data = pd.read_csv("Copper_Set.xlsx - Result 1.csv")
data


# In[66]:


data.describe()


# In[67]:


data.info()


# In[68]:


data.dtypes


# In[69]:


data.value_counts()


# In[70]:


data['item_date'] = pd.to_datetime(data['item_date'], format='%Y%m%d', errors='coerce').dt.date
data['delivery date'] = pd.to_datetime(data['delivery date'], format='%Y%m%d', errors='coerce').dt.date


# In[71]:



data['quantity tons'] = pd.to_numeric(data['quantity tons'], errors='coerce')
data['customer'] = pd.to_numeric(data['customer'], errors='coerce')
data['country'] = pd.to_numeric(data['country'], errors='coerce')
data['application'] = pd.to_numeric(data['application'], errors='coerce')
data['thickness'] = pd.to_numeric(data['thickness'], errors='coerce')
data['width'] = pd.to_numeric(data['width'], errors='coerce')
data['material_ref'] = data['material_ref'].str.lstrip('0')
data['product_ref'] = pd.to_numeric(data['product_ref'], errors='coerce')
data['selling_price'] = pd.to_numeric(data['selling_price'], errors='coerce')


# In[72]:


data


# In[73]:


data.isnull().sum()


# In[74]:


#to get % of null values 
(data["material_ref"].isnull().sum()/data.shape[0]*100)


# In[75]:


null_col = []
for col in data.columns:
    if data[col].isnull().sum()>0:
        null_col.append(col)


# In[76]:


null_col


# In[77]:


# percentage of null values in null col
for col in null_col:
    percentage = round((data[col].isnull().sum()/data.shape[0])*100,2)
    print(f"{col} has {percentage} of null values")


# In[ ]:





# In[78]:


data["material_ref"].fillna("unknown",inplace=True)


# In[79]:


data = data.dropna()


# In[80]:


data1 = data.copy()


# In[81]:


a = data1['selling_price'] <= 0
print(a.sum())
data1.loc[a, 'selling_price'] = np.nan

a = data1['quantity tons'] <= 0
print(a.sum())
data1.loc[a, 'quantity tons'] = np.nan

a = data1['thickness'] <= 0
print(a.sum())


# In[82]:


sns.distplot(data1['quantity tons'])
plt.show()
sns.distplot(data1['country'])
plt.show()
sns.distplot(data1['application'])
plt.show()
sns.distplot(data1['thickness'])
plt.show()
sns.distplot(data1['width'])
plt.show()
sns.distplot(data1['selling_price'])
plt.show()


# In[83]:


data1['selling_price_log'] = np.log(data1['selling_price'])
sns.distplot(data1['selling_price_log'])
plt.show()

data1['quantity tons_log'] = np.log(data1['quantity tons'])
sns.distplot(data1['quantity tons_log'])
plt.show()

data1['thickness_log'] = np.log(data1['thickness'])
sns.distplot(data1['thickness_log'])


# In[84]:


#converting catagorical into numerical
from sklearn.preprocessing import OrdinalEncoder
OE = OrdinalEncoder()
data1.status = OE.fit_transform(data1[['status']])
data1['item type'] = OE.fit_transform(data1[['item type']])


# In[85]:


data1 = data1.dropna()


# In[86]:


X=data1[['quantity tons_log','status','item type','application','thickness_log','width','country','customer','product_ref']]
y=data1['selling_price_log']


# In[87]:


from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS.fit_transform(X)


# In[88]:


#spliting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[89]:


#Regression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[90]:


lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))


# In[91]:


rf = RandomForestRegressor(n_estimators = 20, max_depth = 4, random_state = 42)
rf.fit(X_train, y_train)
print(rf.score(X_train, y_train))
print(rf.score(X_test,y_test))


# In[92]:


gbr = GradientBoostingRegressor(n_estimators = 10, learning_rate = 0.1, random_state =42)
gbr.fit(X_train,y_train)
print(gbr.score(X_train,y_train))
print(gbr.score(X_test,y_test))


# In[99]:


dtr = DecisionTreeRegressor()
# hyperparameters
param_grid = {'max_depth': [2, 5, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'max_features': ['auto', 'sqrt', 'log2']}
# gridsearchcv
grid_search = GridSearchCV(estimator=dtr, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)


# In[95]:


from sklearn import metrics


# In[96]:


#MSE
metrics.mean_squared_error(y_test,y_pred)


# In[97]:


MAE
metrics.mean_absolute_error(y_test, y_pred)


# In[98]:


R2
print(metrics.r2_score(y_test,y_pred))


# In[38]:


dfc = df.copy()


# In[39]:


dfc = df[df['status'].isin(['Won', 'Lost'])]
len(dfc)


# In[40]:


dfc.status.value_counts()


# In[41]:


#convering catagorical into numerical
OE = OrdinalEncoder()
dfc.status = OE.fit_transform(dfc[['status']])
dfc['item type'] = OE.fit_transform(dfc[['item type']])


# In[32]:


#split data into X, y
X = dfc[['quantity tons','selling_price','item type','application','thickness','width','country','customer','product_ref']]
y = dfc['status']


# In[36]:


SS.fit_transform(X)


# In[37]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[38]:


#Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# In[127]:


rfc = RandomForestClassifier(n_estimators = 20, max_depth =6,random_state = 35)
rfc.fit(X_train, y_train)
print(rfc.score(X_train, y_train))
print(rfc.score(X_test, y_test))


# In[39]:


dtc = DecisionTreeClassifier(max_depth = 5, random_state = 1)
dtc.fit(X_train, y_train)
train_score = dtc.score(X_train, y_train)
test_score = dtc.score(X_test, y_test)
print(train_score)
print(test_score)


# In[ ]:


gbc = GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.1,random_state = 28)
gbc.fit(X_train, y_train)
print(gbc.score(X_train, y_train))
print(gbc.score(X_test, y_test))


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train, y_train)
print(knn.score(X_train, y_train))
print(knn.score(X_test, y_test))


# In[103]:


from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
import xgboost as xgb
xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)
xgb_model.fit(X_train, y_train)
xgb_model.score(X_train, y_train)


# In[104]:


y_predict = xgb_model.predict(X_test)


# In[105]:


mean_squared_error(y_test, y_predict)


# In[106]:


xgb_model


# In[107]:


from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=20, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))


# In[108]:


knn.predict(X_test)


# In[109]:


y_pred= knn.predict(X_test)


# In[117]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[100]:


data


# In[120]:


from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix,classification_report, roc_curve, auc


# In[121]:


accuracy_score(y_test, y_pred)


# In[122]:


f1_score(y_test,y_pred, average = 'macro')


# In[123]:


plot_confusion_matrix(knn, X_test, y_test);


# In[124]:


# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'm-')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.2])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[125]:


print(classification_report(y_test, y_pred))


# In[102]:





# In[ ]:





# In[ ]:





# In[ ]:





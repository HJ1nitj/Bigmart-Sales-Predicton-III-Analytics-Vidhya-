# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:43:53 2018

@author: DELL
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import seaborn as sns
sns.set()

#loading the packages
X=pd.read_csv('train.csv')
y=pd.read_csv('test.csv')

#basic fetching
print X.shape
print X.info()
print y.shape
print y.info()
print X.isnull().sum()

##############################################################
#------------------------CLEANING-----------------------------
##############################################################

#**********************Item weight*********************
X['Item_Weight'].fillna(X.groupby('Item_Type')['Item_Weight'].transform('mean'), inplace=True)
y['Item_Weight'].fillna(y.groupby('Item_Type')['Item_Weight'].transform('mean'), inplace=True)




#*****************Outlet_Size**************************
print X['Outlet_Size'].value_counts()
train_test_data=[X, y]
#mapping of Outlet_Size
outlet_size_mapping={'Medium':0, 'Small':1, 'High':2}
for dataset in train_test_data:
    dataset['Outlet_Size']=dataset['Outlet_Size'].map(outlet_size_mapping)

#filling the missing value
X['Outlet_Size'].fillna(X.groupby('Outlet_Type')['Outlet_Size'].transform('median'), inplace=True)
y['Outlet_Size'].fillna(y.groupby('Outlet_Type')['Outlet_Size'].transform('median'), inplace=True)

#Visualising the outlet size
plt.scatter(X['Outlet_Size'], X['Item_Outlet_Sales'], marker='o', c='green')
plt.show()

#One Hot Encoder
#(TRAIN)
dummies_X=pd.get_dummies(X['Outlet_Size'])
dummies_X.columns=['Medium','Small','High']
X=pd.concat([X, dummies_X], axis=1)
#(TEST)
dummies_y=pd.get_dummies(y['Outlet_Size'])
dummies_y.columns=['Medium','Small','High']
y=pd.concat([y, dummies_y], axis=1)

#*********************Fat Content*********************
print X.info()
print X['Item_Fat_Content'].value_counts()
train_test_data=[X, y]
#mapping of fat_content
item_fat_content_mapping={'Low Fat':0, 'Regular':1, 'LF':0, 'reg':1, 'low fat':0}
for dataset in train_test_data:
    dataset['Item_Fat_Content']=dataset['Item_Fat_Content'].map(item_fat_content_mapping)

#dummies
#(Trian)
dummies_X=pd.get_dummies(X['Item_Fat_Content'])
dummies_X.columns=['Low Fat', 'Regular']
X=pd.concat([X, dummies_X], axis=1)
#(TEST)
dummies_y=pd.get_dummies(y['Item_Fat_Content'])
dummies_y.columns=['Low Fat', 'Regular']
y=pd.concat([y, dummies_y], axis=1)


#*********************Item Type********************
print X['Item_Type'].value_counts()
#mapping of the Item_type
train_test_data=[X, y]
item_type_mapping={'Fruits and Vegetables':0, 'Snack Foods':1, 'Household':2, 'Frozen Foods':3, 'Dairy':4 ,'Canned':5, 'Baking Goods':6, 'Health and Hygiene':7, 'Soft Drinks':8, 'Meat':9, 'Breads':10, 'Hard Drinks':11,  'Starchy Foods':12, 'Breakfast':13, 'Seafood':14, 'Others':15}
for dataset in train_test_data:
    dataset['Item_Type']=dataset['Item_Type'].map(item_type_mapping)
    
'''  
#Dummies for Item type
#(Trian)
dummies_X=pd.get_dummies(X['Item_Type'])
dummies_X.columns=['Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods', 'Dairy' ,'Canned', 'Baking Goods', 'Health and Hygiene', 'Soft Drinks', 'Meat', 'Breads', 'Hard Drinks',  'Starchy Foods', 'Breakfast', 'Seafood', 'Others']
X=pd.concat([X, dummies_X], axis=1)
#(TEST)
dummies_y=pd.get_dummies(y['Item_Type'])
dummies_y.columns=['Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods', 'Dairy' ,'Canned', 'Baking Goods', 'Health and Hygiene', 'Soft Drinks', 'Meat', 'Breads', 'Hard Drinks',  'Starchy Foods', 'Breakfast', 'Seafood', 'Others']
y=pd.concat([y, dummies_y], axis=1)
'''
X=X.drop(['Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods', 'Dairy' ,'Canned', 'Baking Goods', 'Health and Hygiene', 'Soft Drinks', 'Meat', 'Breads', 'Hard Drinks',  'Starchy Foods', 'Breakfast', 'Seafood', 'Others'], axis=1)  
y=y.drop(['Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods', 'Dairy' ,'Canned', 'Baking Goods', 'Health and Hygiene', 'Soft Drinks', 'Meat', 'Breads', 'Hard Drinks',  'Starchy Foods', 'Breakfast', 'Seafood', 'Others'], axis=1)


#*****************Oulet_Location_Type******************
print X['Outlet_Location_Type'].value_counts()

#mapping
outlet_location_type_mapping={'Tier 1':1, 'Tier 2':2, 'Tier 3':3}
for dataset in train_test_data:
    dataset['Outlet_Location_Type']=dataset['Outlet_Location_Type'].map(outlet_location_type_mapping)

#dummies
#(Trian)
dummies_X=pd.get_dummies(X['Outlet_Location_Type'])
dummies_X.columns=['Tier 1', 'Tier 2', 'Tier 3']
X=pd.concat([X, dummies_X], axis=1)
#(TEST)
dummies_y=pd.get_dummies(y['Outlet_Location_Type'])
dummies_y.columns=['Tier 1', 'Tier 2', 'Tier 3']
y=pd.concat([y, dummies_y], axis=1)

#*******************Outlet_Establishment Year****************8
train_test_data=[X, y]
for dataset in train_test_data:
    dataset['Outlet_Establishment_Year']=2013-dataset['Outlet_Establishment_Year']



#**********************Item Visiblity*****************
X['Item_Visibility'] = X['Item_Visibility'].replace(0,np.mean(X['Item_Visibility']))
y['Item_Visibility'] = y['Item_Visibility'].replace(0,np.mean(y['Item_Visibility']))


#****************Outlet_Type*******************
#mapping
Outlet_type_mapping={'Supermarket Type1':1, 'Supermarket Type2':2, 'Supermarket Type3':3, 'Grocery Store':4}
for dataset in train_test_data:
    dataset['Outlet_Type']=dataset['Outlet_Type'].map(Outlet_type_mapping)
    
#Dummies
#(Trian)
dummies_X=pd.get_dummies(X['Outlet_Type'])
dummies_X.columns=['Supermarket Type 1', 'Supermarket Type 2', 'Supermarket Type 3', 'Grocery Store']
X=pd.concat([X, dummies_X], axis=1)
#(TEST)
dummies_y=pd.get_dummies(y['Outlet_Type'])
dummies_y.columns=['Supermarket Type 1', 'Supermarket Type 2', 'Supermarket Type 3', 'Grocery Store']
y=pd.concat([y, dummies_y], axis=1)


######################################################
#-----Visualising the numeric feture with the target variable-----
######################################################

#Item Weight
plt.scatter(X['Item_Weight'], X['Item_Outlet_Sales'], marker='o', c='red')
plt.show()

#Item Visibility
plt.scatter(X['Item_Visibility'], X['Item_Outlet_Sales'], marker='o', c='green')
plt.show()


#Item MRP
plt.scatter(X['Item_MRP'], X['Item_Outlet_Sales'], marker='o', c='blue')
plt.show()


#Outlet_Establishment
plt.scatter(X['Outlet_Establishment_Year'], X['Item_Outlet_Sales'], marker='o', c='red')
plt.show()

plt.scatter(X['Item_MRP'], X['Item_Outlet_Sales'], marker='o', color='red')
plt.scatter(X['Item_Visiblity'], X['Item_Outlet_Sale'], color='green')
plt.xlim(xmin=5, xmax=300)
plt.show()

#box plot
X['Item_Visibility'].plot.box()

#####################################################
#----------------------linear_Regression-------------- 
#####################################################
from sklearn.preprocessing import PolynomialFeatures
X=X.drop(['Item_Identifier', 'Outlet_Identifier', 'Outlet_Size', 'Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Type'], axis=1)
y=y.drop(['Item_Identifier', 'Outlet_Identifier', 'Outlet_Size', 'Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Type'], axis=1)

print X.shape
print y.shape

X_predictors=X.iloc[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
X_target=X.iloc[:, 5:6]

print X_predictors.shape
#spliting into the train and test data
X_train, X_test, y_train, y_test=train_test_split(X_predictors, X_target, test_size=0.3, random_state=123)

print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape

len_reg=LinearRegression()
len_reg.fit(X_train, y_train)
y_pred=len_reg.predict(X_test)

###############################################
#-------Checking the Hetroskedacity---------
###############################################
plt.scatter(y_pred, (y_pred-y_test), marker='o', c='green')
plt.xlabel('Fitted Values')
plt.ylabel('Residual')
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max())
plt.show


#the above graph shows that the variation in error is non-constant
#Converting the target into log(Y)


#***********Now checking the OLS funtion******
import statsmodels.formula.api as sm
b0=np.ones((8523, 1))
ones_df=pd.DataFrame({'b0':b0[:, 0]})
X_predictor_new=pd.concat([ones_df['b0'].astype(int), X_predictors], axis=1)
import math
X_target['Item_Outlet_Sales']=np.sqrt(X_target['Item_Outlet_Sales'])

#OLS
X_opt=X_predictor_new.iloc[:, [0, 1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
regresser_OLS=sm.OLS(endog=X_target, exog=X_opt).fit()
regresser_OLS.summary()


X_opt=X_predictor_new.iloc[:, [0,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
regresser_OLS=sm.OLS(endog=X_target, exog=X_opt).fit()
regresser_OLS.summary()


X_opt=X_predictor_new.iloc[:, [0,2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
regresser_OLS=sm.OLS(endog=X_target, exog=X_opt).fit()
regresser_OLS.summary()

X_opt=X_predictor_new.iloc[:, [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
regresser_OLS=sm.OLS(endog=X_target, exog=X_opt).fit()
regresser_OLS.summary()
###########################################################
#-------------------Train and Test Data-----------------
###########################################################
Xtrain=X_predictor_new.iloc[:, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
Xtarget=X_target['Item_Outlet_Sales']

ytest=y.drop([ 'Item_Weight', 'Item_Type', 'Item_Visibility'], axis=1)

print Xtrain.shape
print ytest.shape

####################################################
#----------------Linear Regression------------------
####################################################
lreg=LinearRegression()
lreg.fit(Xtrain, Xtarget)

y_prediction=lreg.predict(ytest)
print y_prediction[0]

pred_df=pd.DataFrame({'Item_Outlet_Sales':y_prediction})
pred_df['Item_Outlet_Sales']=np.square(pred_df['Item_Outlet_Sales'])

print pred_df.head()

######################################
#-----------Submission Block-------
######################################
pred_df.to_csv('19th _Submission.csv', index=False)


#########################################
#--------Polynomial Regression-----------
#########################################
import statsmodels.formula.api as sm
from sklearn.preprocessing import PolynomialFeatures
Xpoly=Xtrain.iloc[:, 2:3]
ypoly=ytest.iloc[:, 2:3]
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(Xpoly)
y_poly=poly_reg.fit_transform(ypoly)

#OLS

df=pd.DataFrame(X_poly)
X_poly=df.iloc[:, 1:3]
X_opt=sm.OLS(endog=X_target, exog=X_poly).fit()
X_opt.summary()

print X_poly.iloc[:, 0:2]
##############################################
#------Adding the polynomial terms in data--------
##############################################
Xpoly_df=pd.DataFrame({'MRP':X_poly.iloc[:, 0], 'MRP^2':X_poly.iloc[:, 1]})
ypoly_df=pd.DataFrame({'MRP':y_poly[:, 1], 'MRP^2':y_poly[:, 2]})

Xtrain_poly=pd.concat([Xpoly_df, Xtrain], axis=1)
ytest_poly=pd.concat([ypoly_df, ytest], axis=1)

#drop elements
Xtrain=Xtrain_poly.drop('Item_MRP', axis=1)
ytest=ytest_poly.drop('Item_MRP', axis=1)

print Xtrain.shape
print ytest.shape

########################################
#--------Checking the OLS Function-----
########################################
X_opt=Xtrain.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
regressor_OLS=sm.OLS(endog=X_target, exog=X_opt).fit()
regressor_OLS.summary()

Xtrain_new=Xtrain.drop(['Grocery Store'], axis=1)
ytest_new=ytest.drop(['Grocery Store'], axis=1)

##########################################
#---------Submission Block-------------
##########################################
lreg=LinearRegression()
lreg.fit(Xtrain_new, X_target)
predictions=lreg.predict(ytest_new)

pred=[]
for i in predictions:
    pred.append(i[0])

pred[0]
pred_df=pd.DataFrame({'Item_Outlet_Sales':pred})
pred_df['Item_Outlet_Sales']=np.square(pred_df['Item_Outlet_Sales'])

#csv
pred_df.to_csv('13th.csv',index=False)

################################################
#---------Decision Tree Classifier--------
################################################
from sklearn.tree import DecisionTreeRegressor
decision_tree=DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
decision_tree.fit(Xtrain_new, X_target)
decision_pred=decision_tree.predict(ytest_new)

decision_pred[0]
result=np.square(decision_pred)

df=pd.DataFrame({'Item_Outlate_Sales':result})
df.to_csv('14th.csv', index=False)

###########################################
#----------Random Forest--------------
############################################
from sklearn.ensemble import RandomForestRegressor
random_regressor=RandomForestRegressor(n_estimators=1000, max_depth=7, min_samples_leaf=100,n_jobs=4)
random_regressor.fit(Xtrain_new, X_target)

random_pred=random_regressor.predict(ytest_new)

random_df=pd.DataFrame({'Item_Outlet_Sales':random_pred})
random_df['Item_Outlet_Sales']=np.square(random_df['Item_Outlet_Sales'])
random_df.to_csv('24th.csv', index=False)


####################################################
#---------------XGBoost-------------------
####################################################
from xgboost.sklearn import XGBRegressor
dmatrix=xgb.DMatrix(data=Xtrain_new, label=X_target)
params={'booster':'gblinear', 'objective':'reg:linear'}
xg_reg=xgb.train(params=params, dtrain=dmatrix, num_boost_round=10)
dmtrix_test=xgb
xg_pred=xg_reg.predict(ytest_new)

xg_reg=XGBRegressor(n_estimators=500, max_depth=6, objective='reg:linear', seed=123)
xg_reg.fit(Xtrain_new, X_target)
xg_pred=xg_reg.predict(ytest_new)

xg_df=pd.DataFrame({'Item_Outlet_Sales':xg_pred})
xg_df['Item_Outlet_Sales']=np.square(xg_df['Item_Outlet_Sales'])
random_df.to_csv('18th.csv', index=False)

#############################################
#-------------SVR-----------------------
##############################################
from sklearn.svm import SVR
X_target_new=np.square(X_target['Item_Outlet_Sales'])

svm_reg=SVR(C=3, epsilon=0.2)
svm_reg.fit(Xtrain_new, X_target_new)

svm_pred=svm_reg.predict(ytest_new)
svm_df=pd.DataFrame({'Item_Outlet_Sales':svm_pred})
svm_df['Item_Outlet_Sales']=np.square(svm_df['Item_Outlet_Sales'])
svm_df.to_csv('26th SVM.csv', index=False)



train.pivot_table(values='Item_Outlet_Sales', index='Outlet_Type')

train.groupby('Outlet_Type')['Item_Outlet_Sales']

predictors=[x for x in Xtrain_new.columns]
coef=pd.Series(lreg.coef_[0], predictors).sort_values()
coef.plot(kind='bar', title='Model Coefficient')
print lreg.coef_


#*********************Ensembling****************
from sklearn.ensemble import AdaBoostClassifier















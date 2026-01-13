from sklearn.datasets import fetch_california_housing
import pandas as pd 
# import house dataset from sklearn
df= pd.DataFrame(fetch_california_housing().data, columns=fetch_california_housing().feature_names)
df['price'] = fetch_california_housing().target

#iporting and viewing the dataset
# first 5 rows
print(df.head())
# shape of the datset
print(df.shape)
# columns with datatypes and non null values
print(df.info())
# handle missing values
# check for null values
# print(df.isnull().sum())
#few missing values-just drop them-dropna()
#many missing values-fill them with mean/mode/median
# df['price'].fillna(df['price'].mean(),inplace=True)
#categorical missing data-fill with mode

#handling duplicates
print("duplicate value:",df.duplicated().sum())
#if there are duplicte value,we are going to drop them-dro_duplicats()

#detection of outliers
df.describe() #use mean,max,min,% values to spot odd values
#graphical representation
from matplotlib import pyplot as plt
df.boxplot(column=['price'])
#5 keys-- min, 25th percentile,meadian,75th percentile, max
plt.show()

#using z-score to remove outliers
import numpy as np
from scipy import stats
#select only numeric columns
numeric_cols=df.select_dtypes(include=[np.number]).columns
#compute z score and filter all rows below 3
z_scores=np.abs(stats.zscore(df[numeric_cols]))
df=df[(z_scores<3).all(axis=1)]
#remove extra spaces
# fix typo: columns property
df.columns = df.columns.str.strip()
#after removing outliers
print("after removing outliers")
print(df.describe())
print("remaining rows and numericcolumns:",len(df),list(numeric_cols))
print(df.info())

#handling categorical data
#label encoding and one hot encoding-classification algos
#skip label encoding since we have all the data in numeric format

#feature scaling-normlaisation and standrdization
#models like knn,svm,linear regreesion get biased toward large valued features
#we use scaling to bring all features to same scle-ensure large valued columns dont sominate smaller
# ones
#standardization we convert values bw 0-1(Z-SCORE SCALING)
#normalization-we convert mean-0and sd-1(MIN-MAX SCALING)


#feature engineering
# 
#splitting the dataset
from sklearn.model_selection import train_test_split
# define features and target
X = df.drop('price', axis=1)
y = df['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
rf=RandomForestRegressor()
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print("R2 SCORE:",r2_score(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))
#save the trained model sing joblib
# import jolib
# jolib.dump(rf,'california_house_price_model.pkl')


#overfitting-prform well on train dataset not so on test dataset
#underfiiting
#regularisation techniques
# lasso
# ridge 





#https://www.geeksforgeeks.org/box-office-revenue-prediction-using-linear-regression-in-ml/?ref=lbp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
#select interpreter through clicking file directly

import os
#print("Current Working Directory:", os.getcwd())



#Properties of the data 

df = pd.read_csv('boxoffice.csv', encoding='latin-1')

#print(df.head())
#print(df.shape)
#print(df.info())
#print(df.describe().T)

to_remove = ['world_revenue', 'opening_revenue']
df.drop(to_remove, axis=1, inplace=True)
#axis is the column, inplace will directly change data frame 
#print(df.head())
#print((df.isnull().sum()*100)/df.shape[0])
#prints percentage of nulls in each column 
df.drop('budget', axis=1, inplace=True)




#cleaning data

for col in ['MPAA', 'genres']:
    df[col] = df[col].fillna(df[col].mode()[0])
    #replaces null value with the most common value in the column (mode)

df.dropna(inplace=True)
#removes rows that have a missing value
df.isnull().sum().sum()

df['domestic_revenue'] = df['domestic_revenue'].str[1:]

for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
    df[col] = df[col].str.replace(',', '')
    #removes , for numbers
    temp = (~df[col].isnull())
    # ~ is negation operator for pandas
    df[temp][col] = df[temp][col].convert_dtypes(float)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    #changes the column datatype to integers instead of object
    
    
    
#plotting data 


'''plt.figure(figsize=(10, 5))
sb.countplot(x=df['MPAA'], data=df)
plt.show()'''
#to change axis ^

#average revenue for each rating category 

#df.groupby('MPAA')['domestic_revenue'].mean()
#groups each rating by their average revenue

#features = ['domestic_revenue', 'opening_theaters', 'release_days']

'''plt.subplots(figsize=(15, 5))
for i, col in enumerate(features):
    plt.subplot(1, 3, i+1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()'''
#plot of distribution of values 
#distplot no axis specification needed


'''plt.subplots(figsize=(15, 5))
for i, col in enumerate(features):
    plt.subplot(1, 3, i+1)
    sb.boxplot(x=df[col])
plt.tight_layout()
plt.show() '''
#plot of distribution in box plot form
#boxplot axis specification needed

'''for col in features:
    df[col] = df[col].apply(lambda x: np.log10(x))'''
    #uses values in column and finds log base 10 of them (scales the values down)
    #lambda: small anonymous (no name) function that uses parameter x and returns value based on one defined expression
    
'''plt.subplots(figsize=(15, 5))
for i, col in enumerate(features):
    plt.subplot(1, 3, i+1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()'''



#applying data

vectorizer = CountVectorizer()
#vectorizer will convert a data frame column into a numerical array (or other formats)
vectorizer.fit(df['genres'])
features = vectorizer.transform(df['genres']).toarray()
genres = vectorizer.get_feature_names_out()
for i, name in enumerate(genres):
    df[name] = features[:, i]
#colon = all the rows
 
df.drop('genres', axis=1, inplace=True)

removed = 0
for col in df.loc[:, 'action':'western'].columns:
    if (df[col]==0).mean() > .95:
        removed +=1
        df.drop(col, axis=1, inplace=True)
        #remove columns with more than 95% zeroes 

for col in ['distributor', 'MPAA']:
    le = LabelEncoder()
    #converts label of each column to a numerical value
    df[col] = le.fit_transform(df[col])
    
plt.figure(figsize=(8, 8))
sb.heatmap(df.corr(numeric_only=True) > .8, annot=True, cbar=False)
#plt.show()
#plot used to see the sci-fi error in which fi and sci are different genres but in actuality are the same data 




#creating a prediction model

features = df.drop(['title', 'domestic_revenue', 'fi'], axis=1)
#print(features)
target = df['domestic_revenue'].values
 
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1, random_state=22)
#input dataframe, splits data in a set which the model is trained on (90%) and a set which the model is tested on (10%)
print(X_train.shape, X_val.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

from sklearn.metrics import mean_absolute_error as mae
model = XGBRegressor()
#creates regression model
model.fit(X_train, Y_train)
print(X_train)

train_predict = model.predict(X_train)
print('Training Error: ', mae(Y_train, train_predict))
val_predict = model.predict(X_val)
print('Validation Error: ', mae(Y_val, val_predict))
print()
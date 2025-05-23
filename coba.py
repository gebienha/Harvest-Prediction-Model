#Importing pandas library which will be used for loading the dataset, Data Analysis, and Data Manipulation
import pandas as pd

#Imported warnings to ignore certain warnings that might arise
import warnings
warnings.filterwarnings('ignore')

#Loading the dataset
df = pd.read_csv("Crop_Data.csv")
df = df[~df.index.duplicated(keep = 'first')]
df.drop(columns=["ID"], axis=1, inplace=True)

#Replacing missing values with the mean
df['Number_Weeks_Used'].fillna(df['Number_Weeks_Used'].mean(),inplace = True)

df.drop(columns=["Unnamed: 0", "Unnamed: 11"], axis=1, inplace=True, errors='ignore')

#Function to return plots for the feature
import scipy.stats as stats
import pylab

def normality(data,feature):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.kdeplot(data[feature])
    plt.subplot(1,2,2)
    stats.probplot(data[feature],plot=pylab)
    plt.show()
    
#Converting Estimated Insects Count feature to Normal Distribution using Box-Cox transform
#Plotting to check the transformation
df['Estimated_Insects_Counts'], parameters = stats.boxcox(df['Estimated_Insects_Count'])
normality(df,'Estimated_Insects_Counts')

import numpy as np
df.loc[df['Number_Weeks_Used']>55,'Number_Weeks_Used'] = np.mean(df["Number_Weeks_Used"])
df.loc[df['Estimated_Insects_Count']>3500,'Estimated_Insects_Count'] = np.mean(df["Estimated_Insects_Count"])
df.loc[df['Number_Weeks_Quit']>40,'Number_Weeks_Quit'] = np.mean(df["Number_Weeks_Quit"])
df.loc[df['Number_Doses_Week']>80,'Number_Doses_Week'] = np.mean(df["Number_Doses_Week"])
df.drop(columns = ["Estimated_Insects_Count"], axis = 1, inplace = True)

#Creating predictors and Target
y = df['Crop_Damage']
X = df.drop(columns = ['Crop_Damage'])

#Performing Train Test split using sklearn library
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.65, random_state = 0)

#We do encoding for nominal data so I used get_dummies method
X_train = pd.get_dummies(data = X_train, columns = ["Season","Pesticide_Use_Category","Soil_Type","Crop_Type"])
np.save('model_columns.npy', X_train.columns)
X_train.head(10)

import joblib

#Let us normalize values for features(Number_Doses_Week,	Number_Weeks_Used,	Number_Weeks_Quit,	Estimated_Insects_Counts)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

joblib.dump(scaler, 'scaler.save')

#Checking normalized values by creating a dataframe
from pandas import DataFrame
X_train_df = DataFrame(X_train)
X_train_df.head(10)

#Performed feature encoding to the X_test feature using get_dummies and then transformed
X_test = pd.get_dummies(data = X_test, columns=["Season","Pesticide_Use_Category","Soil_Type","Crop_Type"])
X_test = scaler.transform(X_test)

#Creating a dataframe from normalized values of test dataset
X_test_df = DataFrame(X_test)
X_test_df.head(10)
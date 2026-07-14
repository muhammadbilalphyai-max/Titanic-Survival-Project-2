#Import important Libraries
import numpy as np
import pandas as pd
#Load dataset
df=pd.read_csv("titanic_train.csv")
print("Dataset loaded successfully")
#Data preprocessing
print(df.columns)
df = df.rename(columns={
    'PassengerId': 'Passenger_ID',
    'Survived': 'Survived',
    'Pclass': 'Passenger_Class',
    'Name': 'Passenger_Name',
    'Sex': 'Gender',
    'Age': 'Age',
    'SibSp': 'Siblings_Spouses',
    'Parch': 'Parents_Children',
    'Ticket': 'Ticket_Number',
    'Fare': 'Ticket_Fare',
    'Cabin': 'Cabin_Number',
    'Embarked': 'Embarkation_Port'
})

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarkation_Port']=df['Embarkation_Port'].fillna(df['Embarkation_Port'].mode()[0],inplace=True)
print(df['Embarkation_Port'].isnull().sum())
#Dropping Unneccessary Columns
X=df.drop(columns=["Passenger_ID",'Ticket_Number','Cabin_Number','Survived'])
y=df['Survived']
#Train test split
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=43)




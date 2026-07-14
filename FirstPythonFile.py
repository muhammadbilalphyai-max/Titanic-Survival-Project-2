#Import important Libraries
import numpy as np
import pandas as pd
#Load dataset
df=pd.read_csv("titanic_train.csv")
print("Dataset loaded successfully")
#Data preprocessing
#print(df.columns)
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
df['Embarkation_Port']=df['Embarkation_Port'].fillna(df['Embarkation_Port'].mode()[0])
print(df['Embarkation_Port'].isnull().sum())
#Label Encoding
df['Gender']=df['Gender'].map(lambda x:1 if x=="male" else 0)
df['Embarkation_Port']=df['Embarkation_Port'].map(lambda x: 0 if x == 'S' else (1 if x == 'C' else 2))
#Dropping Unneccessary Columns
X=df.drop(columns=["Passenger_ID","Passenger_Name",'Ticket_Number','Cabin_Number','Survived'])
y=df['Survived']
#print(df['Survived'].value_counts())

#Train test split
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=43)
#Training model
from sklearn.linear_model import LogisticRegression
Logistic_model=LogisticRegression(max_iter=1000)
Logistic_model.fit(train_x,train_y)
print("Model trained Successfully")
#Prediction
pred_y=Logistic_model.predict((test_x))

#Evaluation
from sklearn.metrics import classification_report,confusion_matrix

# Confusion Matrix Graph
cm = confusion_matrix(test_y, pred_y)

#classification report
print(classification_report(test_y, pred_y))


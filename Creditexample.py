#importing the packages required
import os
import pandas as pd
import numpy as np

#importing the CSV files 
os.chdir('C:\\Users\\hi\\Desktop\\Data Science\\Python\\Loan')
train=pd.read_csv('train_Loan.csv')
test=pd.read_csv('test_Loan.csv')

#describing the data - to find out the nature of data
summary=train.describe()
train.info()

#imputing values

#impute gender
train['Gender'].value_counts()
newgender=np.where(train['Gender'].isnull(),'Male',train['Gender'])
train['Gender']=newgender
#impute married
train['Married'].value_counts()
newmarried=np.where(train['Married'].isnull(),'Yes',train['Married'])
train['Married']=newmarried
#handling the inconsistencies
train['Dependents'].dtype
train['Dependents'].replace('3+','3',inplace=True)
train['Dependents'].value_counts()
newdependents=np.where(train['Dependents'].isnull(),'0',train['Dependents'])
train['Dependents']=newdependents
train['Dependents'].value_counts()
#impute self employed
train['Self_Employed'].value_counts()
newselfemployed=np.where(train['Self_Employed'].isnull(),'No',train['Self_Employed'])
train['Self_Employed']=newselfemployed
#impute loan amount
imval=train['LoanAmount'].median()
train['LoanAmount'].mean()
newloanamount=np.where(train['LoanAmount'].isnull(),imval,train['LoanAmount'])
train['LoanAmount']=newloanamount
#imputeloan amount term
amtvale=train['Loan_Amount_Term'].median()
newloanamtterm=np.where(train['Loan_Amount_Term'].isnull(),amtvale,train['Loan_Amount_Term'])
train['Loan_Amount_Term']=newloanamtterm
#impute credit history
train['Credit_History'].value_counts()
newcredithistory=np.where(train['Credit_History'].isnull(),1,train['Credit_History'])
train['Credit_History']=newcredithistory

#converting the signals-categorical and ordinal
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
train['Gender']=LE.fit_transform(train['Gender'])
train['Gender'].value_counts()
train['Married']=LE.fit_transform(train['Married'])
train['Married'].value_counts()
train['Education']=LE.fit_transform(train['Education'])
train['Education'].value_counts()
train['Self_Employed']=LE.fit_transform(train['Self_Employed'])
train['Self_Employed'].value_counts()
train['Property_Area']=LE.fit_transform(train['Property_Area'])
train['Property_Area'].value_counts()
train['Loan_Status']=LE.fit_transform(train['Loan_Status'])
train['Loan_Status'].value_counts()

#Using different models to predict  - Linear regression, Logistic regression, Random Forest, Support Vector Machine, Clusttering means)
train.info()
Y=train['Loan_Status']
X=train[['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']]

#linear regression

from sklearn import linear_model
lm=linear_model.LinearRegression()
model=lm.fit(X,Y)
preds_LR=model.predict(X)

from sklearn.metrics import mean_squared_error
rmse_LR=np.sqrt(mean_squared_error(Y,preds_LR))

print(rmse_LR)

#Logistic Regression 
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X,Y)
preds_lr=clf.predict(X)

from sklearn.metrics import confusion_matrix
cm_logisticR=confusion_matrix(Y,preds_lr)

from sklearn.metrics import accuracy_score
print("accuracy:",accuracy_score(Y,preds_lr))

#randomforestclassifier 
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=500)
model_rf=rf.fit(X,Y)
preds_rf=model_rf.predict(X)

from sklearn.metrics import confusion_matrix
rfc=confusion_matrix(Y,preds_rf)
from sklearn.metrics import accuracy_score
print("accuracy:",accuracy_score(Y,preds_rf))

#support vector machine 
from sklearn.svm import SVC
svc_c=SVC(kernel='rbf')
model_svc=svc_c.fit(X,Y)
preds_svc=model_svc.predict(X)
from sklearn.metrics import confusion_matrix
svc_c=confusion_matrix(Y,preds_svc)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y,preds_svc))

############### clusterring K means ###################
from sklearn.metrics import confusion_matrix
cm_logisticR= confusion_matrix(Y,preds_lr)

X_continous=X[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']]

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2,random_state=2)
kmeans.fit(X_continous)
pred_clusters=kmeans.predict(X_continous)



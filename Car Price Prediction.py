import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics


data=pd.read_csv("car data.csv")
print(data.shape)
print(data.head())

data.info()

print(data.isnull().sum())


#checking the distribution of categorical data
print(data.Fuel_Type.value_counts())
print(data.Transmission.value_counts())
print(data.Seller_Type.value_counts())


#encodingn the catergorical data into numerical data

data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
data.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
data.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

print(data.head())

#traintest split

X=data.drop(['Car_Name','Selling_Price'],axis=1)
Y=data['Selling_Price']

print(X)
print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.1, random_state=2)

#Linear Regression

lin_Rig_Model=LinearRegression()
lin_Rig_Model.fit(X_train,Y_train)

#Model Evaluation
predict_linear_reg =lin_Rig_Model.predict(X_train)


#Error prediction
error_score = metrics.r2_score(Y_train,predict_linear_reg)
print("Error Score:",error_score)

#visualize actual
plt.scatter(Y_train,predict_linear_reg)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()


#Model Evaluation
test_data_prediction =lin_Rig_Model.predict(X_test)


#Error prediction
error_score = metrics.r2_score(Y_test,test_data_prediction)
print("Error Score:",error_score)

#visualize actual
plt.scatter(Y_test,test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()

#prediction with new data
prediction=lin_Rig_Model.predict((np.array([[2015,10.38,26000,0,0,0,0]])))
print("Prediction:",prediction)





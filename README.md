# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset using pandas.
2. Select input features and target variable (price).
3. Split the data into training and testing sets.
4. Train the Linear Regression model with scaling.
5. Train the Polynomial Regression model (degree 2).
6. Predict prices using both models.
7. Evaluate performance using MSE, MAE, and R² score and compare results.


## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: Anisha A
RegisterNumber: 212225220009
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
df=pd.read_csv('encoded_car_data (1).csv')
print(df.head())
X=df[['enginesize','horsepower','citympg','highwaympg']]
Y=df['price']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#1.linear regression(with scaling)
lr=Pipeline([
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
lr.fit(X_train,Y_train)
Y_pred_linear=lr.predict(X_test)
#2.polynomial regression(degree=2)
poly_model=Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
poly_model.fit(X_train,Y_train)
Y_pred_poly=poly_model.predict(X_test)
#evaluate models
print('Name: Anisha A')
print('reg. No.: 212225220009')
print("Linear Regression:")
print("MSE=",mean_squared_error(Y_test,Y_pred_linear))
print('MAE=',mean_absolute_error(Y_test,Y_pred_linear))
print("R2 Score=",r2_score(Y_test,Y_pred_linear))
print("\nPolynomial Regression:")
print("MSE=",mean_squared_error(Y_test,Y_pred_poly))
print('MAE=',mean_absolute_error(Y_test,Y_pred_poly))
print(f"R2 Score= {r2_score(Y_test,Y_pred_poly):.2f}")
#plot actual vs predict
plt.figure(figsize=(10,5))
plt.scatter(Y_test,Y_pred_poly,label='linear',alpha=0.6)
plt.scatter(Y_test,Y_pred_poly,label='Polynomial (degree=2)',alpha=0.6)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
```

## Output:
<img width="700" height="150" alt="Screenshot 2026-02-12 082026" src="https://github.com/user-attachments/assets/b18e40cd-ccb9-4f46-b3b2-68940cb22bac" />
<img width="640" height="122" alt="Screenshot 2026-02-12 082247" src="https://github.com/user-attachments/assets/a2281512-59d7-431a-96a8-8ba8b4f8974f" />
<img width="1094" height="539" alt="Screenshot 2026-02-12 082301" src="https://github.com/user-attachments/assets/2f098481-afdb-4a3d-b414-3ec169089ef1" />




## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.

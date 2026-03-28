# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Start
2.Import libraries (pandas, numpy, sklearn matplotlib, seaborn, statsmodels)
3.Load dataset
4.Read CarPrice_Assignment.csv into a dataframe df.
5.Select features and target
6.Set X = [enginesize, horsepower, citympg, highwaympg] Set Y = price
7.Split dataset
8.Divide into training and testing sets using train_test_split (80% train, 20% test).
9.Scale the features
10.Create StandardScaler
11.Fit scaler on X_train and transform it → X_train_scaled
12.Transform X_test using same scaler → X_test_scaled
13.Train the model
14.Create LinearRegression model and fit training data
15.Predict prices for X_test_scaled → Y_pred
16.Evaluate performance Compute and print: MSE, MAE, RMSE, R² 17.Check linearity
18.Plot Actual (Y_test) vs Predicted (Y_pred) with reference line
19.Residual analysis Compute residuals = Y_test - Y_pred 
20.Calculate Durbin-Watson statistic for autocorrelation Check homoscedasticity
Plot residuals vs predicted using sns.residplot

```

## Program:
```
/*
Program to implement linear regression model for predicting car prices and test assumptions.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df=pd.read_csv('CarPrice_Assignment.csv')
df.head()
X=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)
print('Name: S R NIVEDHITHA')
print('Reg No:212225240102')
print("MODEL COEFFICIENTS:")
for feature,coef in zip(X.columns,model.coef_):
    print(f"{feature:}:{coef:}")
print(f"{'Intercept':}:{model.intercept_:}")
print("\nMODEL PERFORMANCE:")
print(f"{'MSE':}:{mean_squared_error(y_test,y_pred):}")
print(f"{'RMSE':}:{np.sqrt(mean_squared_error(y_test,y_pred)):}")
print(f"{'MAE':}:{mean_absolute_error(y_test,y_pred):}")
print(f"{'R-squared':}:{r2_score(y_test,y_pred):}")
plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()
residuals=y_test-y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistics: {dw_test:.2f}",
     "\n(Values close to 2 indicate no autocorrelation)")
plt.figure(figsize=(10,5))
sns.residplot(x=y_pred,y=residuals,lowess=True,line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()
fig, (ax1,ax2)= plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals,kde=True,ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals,line='45',fit=True,ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()

Developed by: S R NIVEDHITHA
RegisterNumber: 25000724 
*/
```

## Output:
![alt text](<Screenshot 2026-02-15 133128.png>)
![alt text](<Screenshot 2026-02-15 133140.png>)
![alt text](<Screenshot 2026-02-15 133158-1.png>)
<img width="1113" height="591" alt="1 e" src="https://github.com/user-attachments/assets/b0a434a6-3f51-44c3-85d0-9ed7698d03c9" />
<img width="1125" height="582" alt="2 e" src="https://github.com/user-attachments/assets/cfd92107-cf1b-4697-a195-56907fbf53bd" />
<img width="1229" height="508" alt="3 e" src="https://github.com/user-attachments/assets/1601b8b2-acd3-4b03-b153-6209a8b6bbf4" />


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.

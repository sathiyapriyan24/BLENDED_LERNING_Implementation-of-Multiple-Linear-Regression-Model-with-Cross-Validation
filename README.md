# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Bring in the necessary libraries.
2. Load the Dataset: Load the dataset into your environment.
3. Data Preprocessing: Handle any missing data and encode categorical variables as needed.
4. Define Features and Target: Split the dataset into features (X) and the target variable (y).
5. Split Data: Divide the dataset into training and testing sets.
6. Build Multiple Linear Regression Model: Initialize and create a multiple linear regression model.
7. Train the Model: Fit the model to the training data.
8. Evaluate Performance: Assess the model's performance using cross-validation.
9. Display Model Parameters: Output the model’s coefficients and intercept.
10. Make Predictions & Compare: Predict outcomes and compare them to the actual values.
 

## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: SATHIYA PRIYAN G
RegisterNumber:  212225100048
*/
```
```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

data = pd.read_csv('CarPrice_Assignment.csv')
data.head()

data = data.drop(['car_ID', 'CarName'],axis=1)
data = pd.get_dummies(data, drop_first = True)
data.head()

x = data.drop('price',axis=1)
y = data['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

print("Name: DEEPAK B")
print("Reg No: 25018314")
print("\n===Cross Validadion===")
cv_scores = cross_val_score(model, x, y, cv=5)
print("Fold R2 scores:",{f"{score:.4f}"for score in cv_scores})
print(f"Average R2: {cv_scores.mean():.4f}")

y_pred=model.predict(x_test)
print("\n===Test Set Performance===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()

```

## Output:
<img width="633" height="239" alt="Screenshot 2026-03-27 120308" src="https://github.com/user-attachments/assets/012a4db5-ec88-468d-8b05-54dfae900bc2" />
<img width="950" height="678" alt="Screenshot 2026-03-27 120316" src="https://github.com/user-attachments/assets/e61901cc-4908-4a85-8ec6-39bc96503913" />


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.

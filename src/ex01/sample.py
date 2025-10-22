import math
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

if __name__ == '__main__':

    # Load the Boston Housing dataset
    df = pd.read_csv("../../resources/HousingData.csv")
    print(df.head())

    # Define the feature (independent variable) and target (dependent variable)
    X = df[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','LSTAT']] # Number of rooms
    y = df['MEDV'] # Median value of owner-occupied homes in $1000’s ~ie. House prices
    # Check for missing values
    print(df.isnull().sum())

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    model1 = LinearRegression()
    model2 = BayesianRidge()

    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # Print the intercept and coefficient
    print(f"Intercept: {model1.intercept_}")
    print(f"Coefficient: {model1.coef_}")

    # MEDV = -36.25+9.35*6.575 = 25.23
    print(f"R2 train = {model1.score(X_train, y_train)}")
    print(f"R2 train = {model2.score(X_train, y_train)}")


    # Predict house prices for the test set
    y_pred = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    #
    predictions = pd.DataFrame({'Actual': y_test, 'Predicted_lr': y_pred,
                                'Predicted_brr': y_pred2})
    predictions.to_csv("../../resources/predictions.csv",index=False)
    # bài tập: tính RMSE, MAE,[MAPE]
    print(predictions.head())
    x = r2_score(y_test, y_pred)
    print(f"R2 test score: {x}")

    mae_lr = mean_squared_error(y_test, y_pred)
    mae_br = mean_squared_error(y_test, y_pred2)

    print(f"Mean Squared Error: {mae_lr} - {mae_br}")
    rmse_lr = math.sqrt(mean_squared_error(y_test, y_pred))
    rmse_br = math.sqrt(mean_squared_error(y_test, y_pred2))

    print(f"Root Mean Squared Error: {rmse_lr} - {rmse_br}")
    print("DONE")

    pickle.dump(model1, open("model1.pkl", "wb"))
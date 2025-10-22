import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score


def v1():
    df = pd.read_csv("HousingData_1.csv")
    df_2 = pd.read_csv("HousingData_2.csv")
    print(df.head())

    X = df.iloc[:, :-1]  # all columns except target
    y = df.iloc[:, -1]  # target

    X_2 = df_2.iloc[:, :-1]  # all columns except target
    y_2 = df_2.iloc[:, -1]  # target

    alg = RandomForestRegressor(n_estimators=100, random_state=1, max_depth=10)
    alg.fit(X, y) #train tren X


    rmse = root_mean_squared_error(y, alg.predict(X))
    print(f"rmse= {rmse}")
    mean_val=np.abs(y).mean()
    print(f"mean_val= {mean_val}")
    print(f"Mean error= {rmse/mean_val:.2%} ")

    rmse_2 = root_mean_squared_error(y_2, alg.predict(X_2))
    print(f"rmse_2= {rmse_2}")
    mean_val_2 = np.abs(y_2).mean()
    print(f"mean_val= {mean_val}")
    print(f"Mean error= {rmse_2 / mean_val_2:.2%} ")
def v2():
    df = pd.read_csv("../../resources/HousingData.csv")
    X = df.iloc[:, :-1]  # all columns except target
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
    print(f"Training set: {X_train.shape}, {y_train.shape}, {X_valid.shape}, {y_valid.shape}")

    train_errors, valid_errors = [], []
    param_range =[1,2,3,4,5,6,7,8,9,10]
    for param in param_range:
        random_forest = RandomForestRegressor(n_estimators=100, random_state=1, max_depth=param)
        random_forest.fit(X_train, y_train)

        train_errors.append(random_forest.score(X_train, y_train))
        valid_errors.append(random_forest.score(X_valid, y_valid))

    plt.xlabel("param")
    plt.plot(param_range, train_errors, label="train")
    plt.plot(param_range, valid_errors, label="valid")
    plt.legend()
    plt.show()
def v3():
    df = pd.read_csv("HousingData_1.csv")
    X = df.iloc[:, :-1]  # all columns except target
    y = df.iloc[:, -1]  # target
    alg = RandomForestRegressor(n_estimators=100, random_state=1, max_depth=1)
    cross_val_scores = cross_val_score(alg, X, y, cv=5, scoring="neg_mean_squared_error")
    cross_val_scores = np.sqrt(np.abs(cross_val_scores))
    print(f"cross_val_scores= {cross_val_scores}")
    print(f"mean_val= {np.mean(cross_val_scores)}")

if __name__ == '__main__':
    v3()
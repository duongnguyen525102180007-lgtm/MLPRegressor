import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    df = pd.read_csv("HousingData_1.csv")
    X = df.iloc[:, :-1]  # all columns except target
    y = df.iloc[:, -1]  # target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = MLPRegressor(max_iter=10000, random_state=42)
    # Tuning hyperparameter MLPClassifier
    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'alpha': np.logspace(-4, -1, 4),
        'learning_rate': ['constant', 'adaptive']
    }
    # RandomizedSearchCV để tìm tham số tốt nhất
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=2
    )
    # Huấn luyện
    random_search.fit(X_train, y_train)

    # Model tốt nhất
    best_model = random_search.best_estimator_

    # Dự đoán và đánh giá
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # In kết quả
    print("\n========== KẾT QUẢ TUNING ==========")
    print("Best Parameters: ", random_search.best_params_)
    print("Best CV R2 Score: ", random_search.best_score_)
    print("Test R2 Score: ", r2)
    print("Test MSE: ", mse)

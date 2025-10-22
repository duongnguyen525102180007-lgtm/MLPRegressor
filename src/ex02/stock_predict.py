import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

if __name__ == '__main__':
    df = pd.read_csv("../../resources/Google.csv")
    df['Next_Close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df[['Open', 'High', 'Low', 'Close']]
    y = np.where(df['Next_Close'] > df['Close'], 1,
                 np.where(df['Next_Close'] < df['Close'], -1, 0))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train_scaled, y_train)
    y_pred_log = model_lr.predict(X_test_scaled)

    model_rf = RandomForestClassifier(n_estimators=200, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)

    # --- 9. Đánh giá ---
    acc_log = accuracy_score(y_test, y_pred_log)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    print("Logistic Regression Accuracy:", round(acc_log * 100, 2), "%")
    print("Random Forest Accuracy:", round(acc_rf * 100, 2), "%")

    print("\n--- Logistic Regression Report ---")
    print(classification_report(y_test, y_pred_log, zero_division=0))

    print("\n--- Random Forest Report ---")
    print(classification_report(y_test, y_pred_rf, zero_division=0))


    pickle.dump(model_rf, open("log_reg.pkl", "wb"))
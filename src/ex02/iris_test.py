import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':

    # Load the Boston Housing dataset
    df = pd.read_csv("../../resources/Iris.csv")
    print(df.head())
    print(df.describe())

    X =df.iloc[:,:-1] # all columns except target
    y =df.iloc[:,-1] # target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    cls = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    cls_lr = LogisticRegression(random_state=42)
    cls.fit(X_train, y_train)

    y_pred = cls.predict(X_test)
    y_pred_lr = cls.predict(X_test)

    cf = confusion_matrix(y_test, y_pred)
    print(cf)
    print(f"accuracy: {metrics.accuracy_score(y_test, y_pred)}")

    cf_lr = confusion_matrix(y_test, y_pred_lr)
    print(cf_lr)
    print(f"accuracy lr: {metrics.accuracy_score(y_test, y_pred_lr)}")
    print(classification_report(y_test, y_pred_lr))

    #draw tree
    # plt.figure(figsize=(12, 8))
    # plot_tree(cls, feature_names=X.columns, filled=True, rounded=True)
    # plt.show()

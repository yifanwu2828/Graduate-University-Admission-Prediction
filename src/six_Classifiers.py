import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataLoader import load_data
from evaluation import evaluate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

if __name__ == '__main__':

    root = os.path.dirname(os.getcwd())
    file_name = os.path.join(root, 'Data/clean_data.csv')
    x_train, x_test, y_train, y_test = load_data(file_name)

    # ========================= Training ==========================
    # Linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    # Logistic regression model
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)

    # random forest
    RF = RandomForestClassifier(n_estimators=100, max_features=9, max_depth=10, random_state=1)
    RF.fit(x_train, y_train)

    # Gradient boosted decision tree
    GBDT = GradientBoostingClassifier(n_estimators=100, learning_rate=0.07, max_features=9, max_depth=10, random_state=1)
    GBDT.fit(x_train, y_train)

    # support vector machine
    SVM = svm.SVC()
    SVM.fit(x_train, y_train)

    # XGBoost
    XGB = XGBClassifier(n_estimators=100, learning_rate=0.07, max_depth=10, use_label_encoder=False)
    XGB.fit(x_train, y_train)
    
    
    # ================= Evaluate and print the results ===================
    print('======== Linear regression ========')
    y_pred = lin_reg.predict(x_test) > 0.5
    evaluate(y_pred, y_test)

    print('======== Logistic regression ========')
    y_pred = log_reg.predict(x_test) > 0.5
    evaluate(y_pred, y_test)

    print('======== Random forest ========')
    y_pred = RF.predict(x_test)
    evaluate(y_pred, y_test)

    print('======== Gradient boosted decision tree ========')
    y_pred = GBDT.predict(x_test)
    evaluate(y_pred, y_test)

    print('======== Support vector machine ========')
    y_pred = SVM.predict(x_test)
    evaluate(y_pred, y_test)

    print('======== XGBoost ========')
    y_pred = XGB.predict(x_test)
    evaluate(y_pred, y_test)

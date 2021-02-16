# ECE143_Group6
ECE 143  Group6 Winter 2021
# Countdown: 10 day left!!
# Presentation Date: Feb 26th


## Graduate University Admission Prediction

Goal: Predict the probability a student will be admitted to the specific graduate university
Dataset :https://www.kaggle.com/nitishabharathi/university-recommendation

Done:Exploratory data analysis (EDA) analyzes and investigates data sets and summarizes their main characteristics, often employing data visualization methods.
Feb 16th

TODO: Feature Engineering -- Chris Yifan
TODO: Modelling -- David, Kimi

### Modelling

Currently, we have implemented six binary classification models to predict whether a student would be admitted by his target school. The models includes:

+ Linear regression,
+ Logistic regression,
+ Random forest,
+ Gradient boosted decision tree (GBDT),
+ Support vector machine (SVM),
+ XGBoost.

#### Implementation

Run all blocks in the **Classifier.ipynd** file.

#### Results

|                     | Accuracy | Precision | Recall |
| ------------------- | -------- | --------- | ------ |
| Linear regression   | 0.604    | 0.628     | 0.569  |
| Logistic regression | 0.604    | 0.628     | 0.573  |
| Random forest       | 0.678    | 0.705     | 0.647  |
| GBDT                | 0.677    | 0.694     | 0.670  |
| SVM                 | 0.609    | 0.628     | 0.593  |
| XGBoost             | 0.682    | 0.697     | 0.679  |


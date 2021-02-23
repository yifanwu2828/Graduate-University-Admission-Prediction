# ECE143_Group6
ECE 143  Group6 Winter 2021
# Project website: http://123.57.56.85:7791/
# Countdown: 4 days left!!
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

Run all blocks in the **Classifiers.ipynb** file.

#### Results

|                     | Accuracy | Precision | Recall |
| ------------------- | -------- | --------- | ------ |
| Linear regression   | 0.610    | 0.628     | 0.600  |
| Logistic regression | 0.610    | 0.627     | 0.604  |
| Random forest       | 0.684    | 0.711     | 0.652  |
| GBDT                | 0.690    | 0.705     | 0.688  |
| SVM                 | 0.617    | 0.637     | 0.600  |
| XGBoost             | 0.694    | 0.713     | 0.682  |


# ECE143_Group6
ECE 143  Group6 Winter 2021
# Project website: http://123.57.56.85:7791/


## Graduate University Admission Prediction

Goal: Predict the probability a student will be admitted to the specific graduate university
Dataset :https://www.kaggle.com/nitishabharathi/university-recommendation

Done:Exploratory data analysis (EDA) analyzes and investigates data sets and summarizes their main characteristics, often employing data visualization methods.
Feb 16th

Done: Feature Engineering -- Yifan Feb 23th
Dobe: Modelling -- David, Kimi Feb 23th

### Modelling

Currently, we have implemented six binary classification models to predict whether a student would be admitted by his target school. The models includes:

+ Linear regression,
+ Logistic regression,
+ Random forest,
+ Gradient boosted decision tree (GBDT),
+ Support vector machine (SVM),
+ XGBoost.
+ Least Squares Binary Classifier
+ Multilayer Perceptron

#### Implementation

+ Run all blocks in the **Classifiers.ipynb** file for the first 6 models.
+ Run all blocks in **Least_Squares_Binary_Classifiers.ipynb** for the Least Squares Binary model.
+ Run all blocks in **Multilayer_proceptron_model.ipynb** for the MLP model. And the functions used in this model are:

#### Results

|                     | Accuracy | Precision | Recall |
| ------------------- | -------- | --------- | ------ |
| Linear regression   | 0.610    | 0.628     | 0.600  |
| Logistic regression | 0.610    | 0.627     | 0.604  |
| Random forest       | 0.684    | 0.711     | 0.652  |
| GBDT                | 0.690    | 0.705     | 0.688  |
| SVM                 | 0.617    | 0.637     | 0.600  |
| XGBoost             | 0.694    | 0.713     | 0.682  |
| Least Squares Binary| 0.633    | 0.638     | 0.622  |
| MLP                 | 0.624    | 0.634     | 0.614  |

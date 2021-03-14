# Graduate University Admission Prediction
### ECE 143, Group6 Winter 2021

<img src="plot_Result/stonybrook/Stony Brook_Corr.png" height="400">

Goal: Predict the probability a student will be admitted to the specific graduate university <br>
Dataset: https://www.kaggle.com/nitishabharathi/university-recommendation <br>
Project website: http://123.57.56.85:7791/ <br>


## Modelling

Currently, we have implemented eight binary classification models to predict whether a student would be admitted to his or her target school. The models includes:

+ Linear regression,
+ Logistic regression,
+ Random forest,
+ Gradient boosted decision tree (GBDT),
+ Support vector machine (SVM),
+ XGBoost.
+ Least Squares Binary Classifier
+ Multilayer Perceptron


## File Structure
```
.
├── Data                            # Source data, cleansed data, and conversion tables
├── ECE143_Web                      # Website source
├── plot_Result                     # Analysis figures
├── src         
│    ├── EDA.py                     # Code to cleanse data
│    ├── Data_Visualization.ipynb   # Presentation visualizations
│    └── ...                        # Prediction models
└── ...                             # Slides, test cases, and 3rd party modules
```

## How to use this repository
### Update cleansed dataset:
+ **src/EDA.py**: Optionally update cleansed dataset in `/data`
### First six models:
+ **src/six_Classifiers.py**: train and evaluate the first six models.
### Least Squares Binary Classifier
+ Run **src/Least_Squares_Binary_Classifiers.py** for the Least Squares Binary model.
### MLP
+ Run **src/Multilayer_proceptron_model.py** for the MLP model. And the layers used in this model are:
+ Linear Layer
+ ReLU Layer
+ Softmax Layer
+ Loss function Layer
### CNN
+ Run **src/torch model.py** for the CNN model. (reshaped 9 features to 3 * 3 image)
### Utils:
+ **src/DataLoader.py**: Read the data.
+ **src/evaluation.py**: Compute the accuracy, precision, and recall of a model.


## Results
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
| pytorch CNN         | 0.606    | 0.603     | 0.624  |

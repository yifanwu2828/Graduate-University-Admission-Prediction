import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(file_name):
    '''
    Read the data.
    :param file_name: file name.
    :type file_name: str
    :return: (numpy.ndarray)
        x_train: training features.
        x_test:  testing features.
        y_train: training labels.  
        y_test:  testing labels.
    '''
    assert os.path.splitext(file_name)[-1] == ".csv"

    # load the file
    data = pd.read_csv(file_name)   

    # Drop unecessary columns and get the labels
    features = data.drop(columns=['univName','year'])  
    labels = features.pop('admit')

    # Split the dataset into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

    #scale features between 0 and 1
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # convert labels into numpy array
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test

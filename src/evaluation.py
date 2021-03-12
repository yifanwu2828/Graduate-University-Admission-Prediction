import numpy as np

def evaluate(y_pred, y_test):
    '''
    compute the accuracy, precision, and recall of a model.
    :y_pred: predicted labels (np.ndarray)
    :y_test: test labels (np.ndarray)
    '''
    assert type(y_pred) == np.ndarray and type(y_test) == np.ndarray
    accuracy = sum(y_pred == y_test)/len(y_test)
    precision = sum(y_pred+y_test==2)/sum(y_pred==1)
    recall = sum(y_pred+y_test==2)/sum(y_test==1)
    print('accuracy: {}'.format(accuracy))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    return

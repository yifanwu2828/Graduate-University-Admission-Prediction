# Least_Squares_Binary_Classifier
import numpy as np
import pandas as pd


def check_you_gonna_get_in(features):
    """
    input 9 features
    output the confidence
    """
    assert type(features).__module__ == np.__name__
    assert features.shape[0] == 1
    assert features.shape[1] == 9
    return np.matmul(features, beta) + alpha


if __name__ == '__main__':
    VERBOSE = False
    # Load data
    dataset = pd.read_csv('../Data/clean_data.csv')
    dataset = dataset.to_numpy()
    dataset = np.delete(dataset, 8, 1)
    dataset = np.delete(dataset, 8, 1)
    dataset = dataset.astype('float64')
    np.random.shuffle(dataset)
    class_label = dataset[:, -1]  # for last column
    dataset = dataset[:, :-1]  # for all but last column

    # Separate data into training data and testing data

    x_train = dataset[0:49000, :]
    y_train = class_label[0:49000]
    x_test = dataset[49000:, :]
    y_test = class_label[49000:]

    y_train.shape = (49000, 1)
    y_test.shape = (2598, 1)

    if VERBOSE:
        print(dataset.shape)
        print(class_label.shape)
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)

    '''
    Least squares calculation, weights saved in alpha and beta
    
    alpha: constant bias
    
    beta: weights
    '''

    columnOfOnes = np.ones((49000, 1))  # append 1 for adding constant bias term
    A = np.append(x_train, columnOfOnes, 1)
    assert np.linalg.matrix_rank(np.matmul(A.T, A)) == 10
    beta_alpha = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, y_train))
    # print(beta_alpha.shape)

    alpha = beta_alpha[9][0]
    beta = beta_alpha[0:9, :]
    # print(alpha.shape)
    # print(beta.shape)

    '''
    Checking for training data
    '''

    prediction_train = np.matmul(x_train, beta) + alpha
    # print(prediction_train.shape)

    for i in range(prediction_train.shape[0]):
        if prediction_train[i][0] > 0.5:
            prediction_train[i][0] = 1
        else:
            prediction_train[i][0] = 0

    correct_count_train = 0
    tp_train = 0
    fp_train = 0
    fn_train = 0
    for i in range(prediction_train.shape[0]):
        if prediction_train[i][0] == y_train[i][0]:
            correct_count_train += 1
        if y_train[i][0] == 1:
            if prediction_train[i][0] == 1:
                tp_train += 1
            else:
                fn_train += 1
        if y_train[i][0] == 0:
            if prediction_train[i][0] == 1:
                fp_train += 1

    '''
    Checking for testing set
    '''

    prediction_test = np.matmul(x_test, beta) + alpha
    # print(prediction_test.shape)

    for i in range(prediction_test.shape[0]):
        if prediction_test[i][0] > 0.5:
            prediction_test[i][0] = 1
        else:
            prediction_test[i][0] = 0

    correct_count_test = 0
    tp_test = 0
    fp_test = 0
    fn_test = 0
    for i in range(prediction_test.shape[0]):
        if prediction_test[i][0] == y_test[i][0]:
            correct_count_test += 1
        if y_test[i][0] == 1:
            if prediction_test[i][0] == 1:
                tp_test += 1
            else:
                fn_test += 1
        if y_test[i][0] == 0:
            if prediction_test[i][0] == 1:
                fp_test += 1

    '''
    Output result
    '''

    print("training accuracy: ", correct_count_train / y_train.shape[0])
    print("tp_train: ", tp_train)
    print("fp_train: ", fp_train)
    print("fn_train: ", fn_train)
    print("precision_train: ", tp_train / (tp_train + fp_train))
    print("recall_train: ", tp_train / (tp_train + fn_train))
    print("testing accuracy: ", correct_count_test / y_test.shape[0])
    print("tp_test: ", tp_test)
    print("fp_test: ", fp_test)
    print("fn_test: ", fn_test)
    print("precision_test: ", tp_test / (tp_test + fp_test))
    print("recall_test: ", tp_test / (tp_test + fn_test))

    '''
    Conclusion:
    
    Least squares method presents some under fitting
    
    Predicting function
    '''
    check_you_gonna_get_in(x_test[0].reshape(1, 9))



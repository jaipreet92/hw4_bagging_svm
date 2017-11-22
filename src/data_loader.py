import numpy as np


def load_diabetes_train_from_file():
    raw_data = np.loadtxt('../data/diabetes_train.csv', delimiter=',')
    print(raw_data.shape)
    return raw_data


def load_diabetes_test_from_file():
    raw_data = np.loadtxt('../data/diabetes_test.csv', delimiter=',')
    print(raw_data.shape)
    return raw_data

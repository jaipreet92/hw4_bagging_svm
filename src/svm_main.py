import numpy as np
import scipy.stats as stats

from svm_port.svmutil import *

# These are the precomputed max values of the training and test data so that we can scale the values between [0,1]
# There are no negative values so we can divide by the max
train_scaling_factors = [None, 17.0, 199.0, 122.0, 99.0, 846.0, 67.1, 2.329, 72.0]
test_scaling_factors = [None, 15., 197., 102., 63., 545., 59.4, 2.42, 81.]


def load_svm_data():
    y_test, x_test = svm_read_problem('../data/diabetes_libsvmformat_test.txt')
    y_train, x_train = svm_read_problem('../data/diabetes_libsvmformat_train.txt')
    return y_test, x_test, y_train, x_train


def scale_data(x_train, x_test):

    for x in x_train:
        for k, v in x.items():
            scaled_val = v / train_scaling_factors[k]
            x[k] = scaled_val

    for x in x_test:
        for k, v in x.items():
            scaled_val = v / test_scaling_factors[k]
            x[k] = scaled_val


def compute_bias(x_train, y_train, num_bootstrap_samples, kernel):
    predictions = np.full((num_bootstrap_samples, len(y_train)), fill_value=-1)

    # Generate bootstrap sample, train, and store predictions
    for i in range(num_bootstrap_samples):
        print('Training Bootstrap sample {}'.format(i))
        x_train_sample, y_train_sample = generate_bootstrap_sample(x_train, y_train)
        m = svm_train(y_train_sample, x_train_sample, '-t {} -q'.format(kernel))
        y_predicted, p_acc, p_val = svm_predict(y_train_sample, x_train_sample, m, '-q')
        predictions[i] = y_predicted

    # Get most common(main) prediction across the bootstrap samples for each training example
    ybar_vals = stats.mode(predictions, axis=0).mode[0]

    # Bias = 1 where main_prediction != actual_prediction for each training example
    bias_vals = np.zeros(len(y_train))
    bias_vals[np.where(ybar_vals != y_train)] = 1.0

    # Variance = Prob(actual_prediction != main_prediction) for each training example
    variance_vals = np.zeros(len(y_train))
    for i in range(len(y_train)):
        ybar_val = ybar_vals[i]
        predictions_for_example = predictions[:, i]
        assert len(predictions_for_example) == num_bootstrap_samples

        variance_vals[i] = np.count_nonzero(predictions_for_example[:] != ybar_val) / num_bootstrap_samples

    return bias_vals, variance_vals


def generate_bootstrap_sample(x_train, y_train):
    assert len(x_train) == len(y_train)
    idxs = np.random.choice(len(x_train), len(x_train), replace=True)
    return [x_train[i] for i in idxs], [y_train[j] for j in idxs]


def log_results(bias_vals, variance_vals, accuracy, kernel):
    print('Kernel: {} , Accuracy: {} , Total Bias: {}, Total Variance: {}'.format(kernel, accuracy,
                                                                                  np.mean(bias_vals),
                                                                                  np.mean(variance_vals)))


if __name__ == '__main__':
    # Load Diabetes data
    y_test, x_test, y_train, x_train = load_svm_data()

    # We scale the data between 0 and 1 as per the recommendation in https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
    scale_data(x_train, x_test)

    # Train and Test, for kernels 0,1,2,3, get overall accuracy, bias, and variance
    for kernel in range(4):
        m = svm_train(y_train, x_train, '-q -t {}'.format(kernel))
        p_label, p_acc, p_val = svm_predict(y_test, x_test, m, '-q')
        biases, variances = compute_bias(x_train, y_train, num_bootstrap_samples=30, kernel=kernel)
        log_results(biases, variances, p_acc[0], kernel=kernel)

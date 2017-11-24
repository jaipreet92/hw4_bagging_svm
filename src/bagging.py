from dtree import DTree
import scipy.stats as stats
import numpy as np


class Bagging(object):
    def __init__(self, num_samples=1, max_depth=None):
        self._num_samples = num_samples
        self._learners = []
        self._max_depth = max_depth

    def train_ensemble(self, training_data):
        for i in range(self._num_samples):
            curr_tree = DTree(sample=True, max_depth=self._max_depth)
            curr_tree.train(training_data)
            self._learners.append(curr_tree)

    def test(self, testing_data):
        all_predicted_vals = np.zeros((self._num_samples, testing_data.shape[0]))
        for idx, learner in enumerate(self._learners):
            all_predicted_vals[idx, :] = learner.test(testing_data)

        # Find most common val amongst all training examples.
        majority_predicted_vals = stats.mode(all_predicted_vals, axis=0).mode[0]
        actual_vals = testing_data[:, -1]

        error_rate = np.sum(np.absolute(majority_predicted_vals - actual_vals)) / testing_data.shape[0]
        return error_rate, majority_predicted_vals


def calculate_bias(training_data, max_depth=None, num_samples=3):
    num_bootstrap_samples = 30

    bootstrap_samples = []
    for i in range(num_bootstrap_samples):
        idxs = np.random.choice(training_data.shape[0], training_data.shape[0], replace=True)
        bootstrap_samples.append(training_data[idxs, :])

    learners = []
    for bootstrap_sample in bootstrap_samples:
        learner = Bagging(max_depth=max_depth, num_samples=num_samples)
        learner.train_ensemble(bootstrap_sample)
        learners.append(learner)

    predictions = np.full((num_bootstrap_samples, training_data.shape[0]), fill_value=-1)
    for idx, learner in enumerate(learners):
        er, predictions[idx] = learner.test(training_data)

    assert np.any(predictions[:, :] == -1) == False

    ybar_vals = stats.mode(predictions, axis=0).mode[0]
    t_vals = training_data[:, -1]

    # Bias for all test/train examples
    bias_vals = np.absolute(t_vals - ybar_vals)

    # Variance for all test/train examples
    variance_vals = np.zeros(predictions.shape[1])
    row, column = predictions.shape
    for idx in range(column):
        variance_vals[idx] = np.count_nonzero(predictions[:, idx] != ybar_vals[idx]) / predictions.shape[0]

    print('Num Samples: {} Depth: {} Total Bias: {} Variance: {}'.format(num_samples, max_depth, np.sum(bias_vals),
                                                                         np.sum(variance_vals)))
    return bias_vals, variance_vals

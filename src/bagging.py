from dtree import DTree
import scipy.stats as stats
import numpy as np


class Bagging(object):
    def __init__(self, num_samples=1):
        self._num_samples = num_samples
        self._learners = []

    def train_ensemble(self, training_data):
        for i in range(self._num_samples):
            curr_tree = DTree(sample=True)
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
        print('Error rate for sample size {} is {}'.format(self._num_samples, error_rate))
        return error_rate

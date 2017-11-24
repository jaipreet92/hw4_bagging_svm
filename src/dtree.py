from sklearn.tree import DecisionTreeClassifier
import numpy as np
import scipy.stats as stats


class Learner(object):
    def train(self, training_data):
        pass

    def test(self, testing_data):
        pass


class DTree(Learner):
    def __init__(self, max_depth=None, sample=False):
        super().__init__()
        self._tree = None
        self._max_depth = max_depth
        self._sample = sample

    def train(self, training_data):
        training_data_sample = training_data
        if self._sample:
            idxs = np.random.choice(training_data.shape[0], training_data.shape[0], replace=True)
            training_data_sample = training_data[idxs, :]

        tree = DecisionTreeClassifier(criterion='entropy', max_depth=self._max_depth)
        self._tree = tree.fit(training_data_sample[:, 0:7], training_data_sample[:, 8])

    def test(self, testing_data):
        predicted_vals = self._tree.predict(testing_data[:, 0:7])
        actual_vals = testing_data[:, -1]
        self.last_error_rate = np.sum(np.absolute(predicted_vals - actual_vals)) / testing_data.shape[0]
        return predicted_vals


def calculate_bias(training_data, testing_data, max_depth=None):
    num_bootstrap_samples = 30

    bootstrap_samples = []
    for i in range(num_bootstrap_samples):
        idxs = np.random.choice(training_data.shape[0], training_data.shape[0], replace=True)
        bootstrap_samples.append(training_data[idxs, :])

    trees = []
    for bootstrap_sample in bootstrap_samples:
        tree = DTree(sample=False, max_depth=max_depth)
        tree.train(bootstrap_sample)
        trees.append(tree)

    predictions = np.full((num_bootstrap_samples, training_data.shape[0]), fill_value=-1)
    for idx, tree in enumerate(trees):
        predictions[idx] = tree.test(training_data)

    assert np.any(predictions[:, :] == -1) == False
    print(predictions.shape)

    ybar_vals = stats.mode(predictions, axis=0).mode[0]
    t_vals = training_data[:, -1]

    # Bias for all test/train examples
    bias_vals = np.absolute(t_vals - ybar_vals)

    # Variance for all test/train examples
    variance_vals = np.zeros(predictions.shape[1])
    row, column = predictions.shape
    for idx in range(column):
        variance_vals[idx] = np.count_nonzero(predictions[:, idx] != ybar_vals[idx]) / predictions.shape[0]

    print('Depth: {} Total Bias: {} Variance: {}'.format(max_depth, np.sum(bias_vals), np.sum(variance_vals)))
    return bias_vals, variance_vals

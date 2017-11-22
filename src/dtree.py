from sklearn.tree import DecisionTreeClassifier
import numpy as np

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
        training_data_sample  = training_data
        if self._sample:
            idxs = np.random.choice(training_data.shape[0], training_data.shape[0], replace=True)
            training_data_sample = training_data[idxs, :]

        tree = DecisionTreeClassifier(criterion='entropy', max_depth=self._max_depth)
        self._tree = tree.fit(training_data_sample[:, 0:7], training_data_sample[:, 8])

    def test(self, testing_data):
        predicted_vals = self._tree.predict(testing_data[:, 0:7])
        actual_vals = testing_data[:, 8]
        print('Errors: {}'.format(predicted_vals - actual_vals))

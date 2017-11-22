from sklearn.tree import DecisionTreeClassifier


class Learner(object):
    def train(self, training_data):
        pass

    def test(self, testing_data):
        pass


class DTree(Learner):
    def __init__(self, max_depth=None):
        super().__init__()
        self._tree = None
        self._max_depth = max_depth

    def train(self, training_data):
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=self._max_depth)
        self._tree = tree.fit(training_data[:, 0:7], training_data[:, 8])

    def test(self, testing_data):
        predicted_vals = self._tree.predict(testing_data[:, 0:7])
        actual_vals = testing_data[:, 8]
        print('Errors: {}'.format(predicted_vals - actual_vals))

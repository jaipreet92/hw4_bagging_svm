import data_loader
import dtree
from dtree import DTree
from bagging import Bagging

if __name__ == '__main__':
    training_data = data_loader.load_diabetes_train_from_file()
    testing_data = data_loader.load_diabetes_test_from_file()

    # Testing 1 DTree
    tree = DTree(sample=False)
    tree.train(training_data)
    tree.test(testing_data)
    print('Error rate for single learner is {}'.format(tree.last_error_rate))

    # Testing ensemble
    error_rates = []
    for num_sample in [1, 3, 5, 10, 20]:
        bagger = Bagging(num_samples=num_sample)
        bagger.train_ensemble(training_data)
        error_rates.append(bagger.test(testing_data))

    # Calculate bias and variance in 1 DTree
    for max_depth in [1,2,3,6,7,9]:
        dtree.calculate_bias(training_data, testing_data, max_depth=max_depth)
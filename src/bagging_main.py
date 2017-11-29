import data_loader
import dtree
import bagging
from dtree import DTree
from bagging import Bagging
import matplotlib.pyplot as plt


def plot_results(error_rates, base_accuracy, num_samples=[1, 3, 5, 10, 20]):
    fig, ax = plt.subplots()
    error_rates  = [100 * (1 - error_rates[i]) for i in range(len(error_rates))]
    ax.plot(num_samples, error_rates, label='Accuracy')
    ax.set(xlabel='Number of Samples', ylabel='Accuracy %',
           title='Accuracy with bagging(Base={})'.format(100 * (1 - base_accuracy)))
    ax.legend()
    for xy in zip(num_samples, error_rates):
        ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    fig.savefig('../data/bagging_accuracy_result.png')


if __name__ == '__main__':
    training_data = data_loader.load_diabetes_train_from_file()
    testing_data = data_loader.load_diabetes_test_from_file()

    # Testing 1 DTree
    tree = DTree(sample=False)
    tree.train(training_data)
    tree.test(testing_data)
    print('Error rate for single learner is {}'.format(100 * (1 - tree.last_error_rate)))

    # Testing ensemble
    error_rates = []
    for num_sample in [1, 3, 5, 10, 20]:
        for max_depth in [None, 1, 2, 3, 6]:
            bagger = Bagging(num_samples=num_sample, max_depth=max_depth)
            bagger.train_ensemble(training_data)
            error_rate = bagger.test(testing_data)[0]
            if max_depth is None:
                error_rates.append(error_rate)
            print('Num Samples: {} Depth: {} Accuracy: {}'.format(num_sample, max_depth,
                                                                  100 * (1 - error_rate)))
    plot_results(error_rates, tree.last_error_rate)

    # Calculate bias and variance in 1 DTree
    for max_depth in [1, 2, 3, 6, 7, 9]:
        dtree.calculate_bias(training_data, max_depth=max_depth)

    # Calculate bias and variance in Ensemble
    for ensemble_size in [1, 3, 5, 10, 20]:
        for max_depth in [1, 2, 3, 6]:
            bagging.calculate_bias(training_data, max_depth=max_depth, num_samples=ensemble_size)

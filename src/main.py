import data_loader
from dtree import DTree

if __name__ == '__main__':
    training_data = data_loader.load_diabetes_train_from_file()
    testing_data = data_loader.load_diabetes_test_from_file()

    tree = DTree()
    tree.train(training_data)
    tree.test(testing_data)

import data_loader
import mlp_port.sigmoid_backprop as sigmoid
import mlp_port.relu_backprop as relu
from mlp_port.parameters import HyperParameters
import numpy as np

if __name__ == '__main__':
    training_data = data_loader.load_diabetes_train_from_file()
    testing_data = data_loader.load_diabetes_test_from_file()

    training_data_labels = training_data[:, -1]
    testing_data_labels = testing_data[:, -1]

    sigmoid.do_train(training_data_features=training_data[:, :7],
                     training_data_labels=np.array(
                         [[training_data_labels[i]] for i in range(len(training_data_labels))]),
                     testing_data_features=testing_data[:, :7],
                     testing_data_labels=np.array([[testing_data_labels[i]] for i in range(len(testing_data_labels))]),
                     parameters=HyperParameters(num_hidden_units=50,
                                                num_epochs=50,
                                                num_input_units=8,
                                                num_output_units=1,
                                                mini_batch_size=1,
                                                learning_rate=0.1,
                                                momentum=0.1))

    relu.do_train(training_data_features=training_data[:, :7],
                     training_data_labels=np.array(
                         [[training_data_labels[i]] for i in range(len(training_data_labels))]),
                     testing_data_features=testing_data[:, :7],
                     testing_data_labels=np.array([[testing_data_labels[i]] for i in range(len(testing_data_labels))]),
                     parameters=HyperParameters(num_hidden_units=100,
                                                num_epochs=50,
                                                num_input_units=8,
                                                num_output_units=1,
                                                mini_batch_size=1,
                                                learning_rate=0.1,
                                                momentum=0.1))





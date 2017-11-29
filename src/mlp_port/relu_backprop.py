import numpy as np

from mlp_port.parameters import HyperParameters


def back_propagate_errors(training_example_label,
                          hidden_layer_weights,
                          hidden_unit_values,
                          output_unit_values):
    """

    :param training_example_label: 10 x 1 ndarray for actual output layer or label for current training example
    :param hidden_layer_weights: num_hidden_units x 10 ndarray representing weights between output units and hidden units
    :param hidden_unit_values: num_hidden_unit x 1 ndarray representing the hidden unit values obtained after feed forward
    :param output_unit_values: 10 x 1 ndarray representing the output unit values obtained after feed forward
    """
    assert training_example_label.shape[0] == output_unit_values.shape[0]

    output_unit_error_terms = np.zeros(output_unit_values.shape)
    output_unit_error_terms[output_unit_values > 0.0] = 1.0
    output_unit_error_terms = (training_example_label - output_unit_values) * output_unit_error_terms

    hidden_unit_error_terms = np.zeros(hidden_unit_values.shape)
    hidden_unit_error_terms[hidden_unit_values > 0.0] = 1.0
    hidden_unit_error_terms = hidden_unit_error_terms * np.dot(hidden_layer_weights, output_unit_error_terms)

    return hidden_unit_error_terms, output_unit_error_terms


def get_weight_delta(hidden_unit_error_terms,
                     output_unit_error_terms,
                     parameter_factory,
                     training_example,
                     hidden_layer_values,
                     previous_input_weights_delta,
                     previous_hidden_unit_weights_delta):
    """

    :param previous_hidden_unit_weights_delta:
    :param previous_input_weights_delta:
    :param hidden_layer_values:  num_hidden_units x 1 ndarray representing the values calculated at the hidden layer
    :param training_example: 51 x 1 ndarray representing the input layer values of the training example
    :param hidden_unit_error_terms: num_hidden_units x 1 ndarray representing error terms of the hidden layer // LIST
    :param output_unit_error_terms: 10 x 1 ndarray representing error terms of the output layer
    :param parameter_factory:
    """
    input_unit_weights_delta = (np.outer(training_example,
                                         hidden_unit_error_terms) * parameter_factory.learning_rate()) + (
                                   parameter_factory.momentum() * previous_input_weights_delta)
    hidden_unit_weights_delta = (np.outer(hidden_layer_values,
                                          output_unit_error_terms) * parameter_factory.learning_rate()) + (
                                    parameter_factory.momentum() * previous_hidden_unit_weights_delta)
    return input_unit_weights_delta, hidden_unit_weights_delta


def do_train(training_data_features,
             training_data_labels,
             testing_data_features,
             testing_data_labels,
             parameters=HyperParameters(num_hidden_units=100,
                                        num_epochs=15,
                                        num_input_units=51,
                                        num_output_units=10,
                                        mini_batch_size=1,
                                        learning_rate=0.1,
                                        momentum=0.1)):
    """

    :param training_data_features: 60000 x 50 matrix
    :param training_data_labels:  60000 x 1 matrix
    """
    assert training_data_features.shape[0] == training_data_labels.shape[0]

    # Replace nan values in the training data
    replace_nan_values(training_data_features, training_data_labels)

    # Initialize weights
    input_unit_weights, hidden_unit_weights = parameters.initialize_weights()

    # Normalize the training data
    training_data_features = training_data_features / 1000.0
    testing_data_features = testing_data_features / 1000.0

    # Add bias units to the features.
    training_data_features = np.insert(training_data_features, 0,
                                       np.full((training_data_features.shape[0],), 1.0), axis=1)
    testing_data_features = np.insert(testing_data_features, 0,
                                      np.full((testing_data_features.shape[0],), 1.0), axis=1)

    input_unit_weights_delta = np.zeros(input_unit_weights.shape)
    hidden_unit_weights_delta = np.zeros(hidden_unit_weights.shape)

    previous_input_unit_weights_delta = np.zeros(input_unit_weights.shape)
    previous_hidden_unit_weights_delta = np.zeros(hidden_unit_weights.shape)

    # Values for plotting
    epoch_nums = np.arange(0.0, parameters.num_epochs(), 0.5)
    epoch_mse_vals_test = []
    epoch_mse_vals_train = []
    zero_one_errors_test = []
    zero_one_errors_train = []

    for n in range(parameters.num_epochs()):
        for idx, training_example in enumerate(training_data_features):
            if idx % parameters.mini_batch_size() == 0:
                input_unit_weights = input_unit_weights + input_unit_weights_delta
                hidden_unit_weights = hidden_unit_weights + hidden_unit_weights_delta
                input_unit_weights_delta = np.zeros(input_unit_weights.shape)
                hidden_unit_weights_delta = np.zeros(hidden_unit_weights.shape)

            hidden_layer_values, output_layer_values = feed_forward(training_example,
                                                                    input_unit_weights,
                                                                    hidden_unit_weights)
            hidden_unit_error_terms, output_unit_error_terms = back_propagate_errors(training_data_labels[idx],
                                                                                     hidden_unit_weights,
                                                                                     hidden_layer_values,
                                                                                     output_layer_values)
            curr_input_unit_weights_delta, curr_hidden_unit_weights_delta = \
                get_weight_delta(hidden_unit_error_terms,
                                 output_unit_error_terms,
                                 parameters,
                                 training_example,
                                 hidden_layer_values,
                                 previous_input_unit_weights_delta,
                                 previous_hidden_unit_weights_delta)
            input_unit_weights_delta += curr_input_unit_weights_delta
            hidden_unit_weights_delta += curr_hidden_unit_weights_delta
            previous_input_unit_weights_delta = curr_input_unit_weights_delta
            previous_hidden_unit_weights_delta = curr_hidden_unit_weights_delta

            if (idx + 1) % 518 == 0:
                total_square_error_test, total_square_error_train, test_correct_predictions, test_incorrect_predictions, train_error_rate = \
                    get_squared_error(
                        testing_data_features,
                        testing_data_labels,
                        input_unit_weights,
                        hidden_unit_weights,
                        training_data_features,
                        training_data_labels)
                print(
                    'n {} : Test SE: {} Train SE: {} Test Accuracy: {}, Train Accuracy: {}'.format(
                        n, total_square_error_test,
                        total_square_error_train,
                        100.0 - 100.0 * (test_incorrect_predictions / (test_correct_predictions + test_incorrect_predictions)),
                        100.0 - 100.0 * train_error_rate))
                epoch_mse_vals_test.append(total_square_error_test)
                epoch_mse_vals_train.append(total_square_error_train)
                zero_one_errors_test.append(
                    (test_incorrect_predictions / (test_correct_predictions + test_incorrect_predictions)) * 100.9)
                zero_one_errors_train.append(100.0 * train_error_rate)
    print('Done!')


def feed_forward(training_example,
                 input_layer_weights,
                 hidden_layer_weights):
    """
    Feed forward part of the the BackPropagation algorithmn
    :rtype: object
    :param training_example: 51 x 1 ndarray
    :param input_layer_weights: 51 x num_hidden_units ndarray
    :param hidden_layer_weights: num_hidden_units x 10 ndarray
    """
    # calculate hidden layer values
    hidden_layer_values = np.maximum(np.dot(training_example, input_layer_weights), 0.0)
    # calculate output layer values
    output_layer_values = np.maximum(np.dot(hidden_layer_values, hidden_layer_weights), 0.0)
    return hidden_layer_values, output_layer_values


def get_squared_error(testing_data_features,
                      testing_data_labels,
                      input_unit_weights,
                      hidden_unit_weights,
                      training_data_features,
                      training_data_labels):
    total_square_error_test = 0.0
    total_square_error_train = 0.0
    test_correct_predictions = 0.0
    test_incorrect_predictions = 0.0

    train_correct_predictions = 0.0
    train_incorrect_predictions = 0.0
    # Test data
    for idx, test_example in enumerate(testing_data_features):
        predicted_values = feed_forward(test_example, input_unit_weights, hidden_unit_weights)[1]
        actual_values = testing_data_labels[idx]
        total_square_error_test += np.sum(np.square(predicted_values - actual_values))

        if testing_data_labels[idx] == np.around(predicted_values):
            test_correct_predictions += 1.0
        else:
            test_incorrect_predictions += 1.0
    if total_square_error_test == 0.0:
        raise ValueError('0 error ?!')

    # Training data
    for idx, train_example in enumerate(training_data_features):
        predicted_values = feed_forward(train_example, input_unit_weights, hidden_unit_weights)[1]
        actual_values = training_data_labels[idx]
        total_square_error_train += np.sum(np.square(predicted_values - actual_values))

        if training_data_labels[idx] == np.around(predicted_values):
            train_correct_predictions += 1.0
        else:
            train_incorrect_predictions += 1.0

    return total_square_error_test / (2.0 * testing_data_labels.shape[0]), total_square_error_train / (
        2.0 * training_data_labels.shape[
            0]), test_correct_predictions, test_incorrect_predictions, train_incorrect_predictions / (
               train_correct_predictions + train_incorrect_predictions)


def replace_nan_values(training_data, training_data_labels):
    """
    Replaces 'nan' values in the training data with 0.0, as these cause problems down the line when using these
    values
    :param training_data:
    :param training_data_labels:
    """
    if np.any(np.isnan(training_data)):
        np.nan_to_num(training_data, copy=False)
    if np.any(np.isnan(training_data_labels)):
        np.nan_to_num(training_data_labels, copy=False)

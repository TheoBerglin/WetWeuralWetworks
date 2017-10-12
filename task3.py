import numpy as np
import matplotlib.pyplot as plt
import random

def activation_function(chosen_pattern, weights):
    # patterns are row, weights each j a row
    difference = chosen_pattern - weights
    distance = np.linalg.norm(difference, axis=1)**2
    exp_matrix = np.exp(-distance/2.0)
    exp_sum = np.sum(exp_matrix, 0)
    output = exp_matrix / exp_sum
    return output

def winning_neuron(activation_vector):
    winning_index = np.argmax(activation_vector)
    return winning_index

def update_weight(chosen_pattern, weights, winning_index):
    eta = 0.02
    weight_change = eta*(chosen_pattern - weights[winning_index, :])
    new_weight = weights[winning_index, :] + weight_change
    weights[winning_index, :] = new_weight
    return weights

def initialize_weights(n_rows, n_cols):
    lower = -1
    upper = 1
    return np.random.uniform(lower, upper, [n_rows, n_cols])

def unsupervised_learning(inputs, n_neurons, n_inputs):
    weights = initialize_weights(n_neurons, n_inputs)
    n_iterations=  10**5

    n_patterns = inputs.shape[0]
    
    for i in range(n_iterations):
        r = random.randint(0, n_patterns-1)
        selected_pattern = inputs[r]

        activation_vector = activation_function(selected_pattern, weights)
        winning_index = winning_neuron(activation_vector)
        weights = update_weight(selected_pattern, weights, winning_index)

    return weights

def supervised_activation_function(beta, mean_field):
    activation = np.tanh(beta * mean_field)
    return activation

def mean_field(chosen_pattern, weights):
    mean_field = weights * chosen_pattern
    return mean_field
    


def supervised_learning(perceptron_inputs, outputs, n_neurons):
    eta = 0.01
    beta = 1/2.0
    n_iteartions = 3000
    n_patterns = perceptron_inputs.shape[0]

    # Bias trick
    perceptron_weights = initialize_weights(1, n_neurons + 1)

    for i in range(n_iterations):
        r = random.randint(0, n_patterns-1)
        selected_pattern = inputs[r]
        # Bias trick
        input_pattern = np.add(selected_pattern, 1)

        mean_field = mean_field(input_pattern, perceptron_weights)
        calculated_output = supervised_activation_function(beta, mean_field)
        weight_change = eta*(outputs(r) - calculated_output)*input_pattern

        # Should the bias change be positive or negative? IF we get error this
        # may be the reason
        perceptron_weights = weight_change + perceptron_weights

    return perceptron_weights
        

        

    
        

    
if __name__ == '__main__':
    n_neurons = 4
    n_input_variables = 2
    iterations = 20

    pattern = np.array([1, 2])
    weights = np.array([[1.0, 3.0], [4, 6], [7, 8]])
    data = np.loadtxt('data_ex2_task3_2017.txt')

    data_output = data[:, 0]
    data_input = data[:, 1:3]
    print(data_input)

    classification_error = np.zeros([iterations, 1])
    finished_weights = dict() # bias trick
    finished_perceptron_weights = np.zeros([iterations, n_neurons + 1])
    for i in range(iterations):
        # Unsupervised
        weights = unsupervised_learning(data_input, n_neurons, n_input_variables)
        finished_weights[i] = weights
        print(weights)
        n_patterns = data_input.shape[0]
        perceptron_input = np.zeros([n_input_variables, n_neurons])
        for j in range(n_patterns):
            perceptron_input[j] = activation_function(data_input[j], weights)

        # Supervised, using bias trick 
        perceptron_weights = supervised_learning(perceptron_input, data_output, n_neurons)
        n_correct = 0
        n_wrong = 0
        for j in range(n_patterns):
            mean_field = mean_field(perceptron_input[j], perceptron_weights)
            calculated_output = supervised_activation_function(beta, mean_field)
            
            # Should we use some kind of sign function???
            if calculated_output == data_output[j]:
                n_correct += 1
            else:
                n_wrong += 1

        classification_error[i] = n_wrong/(n_wrong + n_correct)
        finished_perceptron_weights[i] = perceptron_weights


    min_index = np.argmin(classification_error)
    weights = finished_weights[min_index]
    plt.subplot(2, 1, 1)
    for i in range(data_input.shape[0]):

        if data_output[i] == 1:
            plt.scatter(data_input[i],color='g')
        else:
            plt.scatter(data_input[i],color='r')
    plt.scatter(weights, color='b')

    plt.subplot(2, 1, 2)

    plt.show

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm

def initialize_weights():
    lower = -1
    upper = 1
    return np.random.uniform(lower, upper, [2,])


def calculate_dw(network_output, eta, weights, selected_pattern):
    return eta*network_output*(selected_pattern-network_output*weights)


def do_oja(input_data):
    n_output = input_data.shape[0]
    weights = initialize_weights()
    iterations = 2*10**4
    eta = 0.001
    weights_time = np.zeros([iterations, 1])
    for t in range(iterations):
        r = random.randint(0, n_output - 1)
        selected_pattern = input_data[r]
        network_output = np.dot(weights.T, selected_pattern)
        weights = weights + calculate_dw(network_output, eta, weights, selected_pattern)
        weights_time[t] = np.linalg.norm(weights)
    return weights, weights_time


if __name__ == '__main__':
    input_data = np.loadtxt('data_ex2_task2_2017.txt')
   # input_data = np.reshape(input_data)
    weights_task_a, weights_time_task_a = do_oja(input_data)
    mean_input = np.mean(input_data, axis=0)
    input_data_mean_zero = input_data-mean_input
    weights_task_b, weights_time_task_b = do_oja(input_data_mean_zero)

    # Plot if the network converges
    
    plt.figure(1)
    n_iterations = weights_time_task_b.shape[0]
    plt.subplot(2, 2, 1)
    plt.plot(range(n_iterations), weights_time_task_a)
    plt.xlabel('Number of iterations')
    plt.ylabel('Norm of weight vector')
    
    plt.subplot(2, 2, 2)
    plt.plot(range(n_iterations), weights_time_task_b)
    plt.xlabel('Number of iterations')
    plt.ylabel('Norm of weight vector')
    plt.subplot(2, 2, 3)
    plt.scatter(input_data[:, 0], input_data[:, 1], color='r', alpha=0.4)
    #vector = np.zeros([2, 1])
    #vector.append(weights)
    plt.quiver(0,0,weights_task_a[0], weights_task_a[1], scale_units='xy', scale=1)

    plt.subplot(2, 2, 4)
    plt.scatter(input_data_mean_zero[:, 0], input_data_mean_zero[:, 1], color='r', alpha=0.4)
    # vector = np.zeros([2, 1])
    # vector.append(weights)
    plt.quiver(0, 0, weights_task_b[0], weights_task_b[1], scale_units='xy', scale=1)
    plt.show()
   # plt.plot(weights_ordering[:, 0], weights_ordering[:, 1], color='b', marker='o')





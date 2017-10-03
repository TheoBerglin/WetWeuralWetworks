import numpy as np
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm


def initialize_weights(n_outputs, n_inputs):
    lower = -1
    upper = 1
    return np.random.uniform(lower, upper, [n_outputs, n_inputs])


def create_random_patterns(n_patterns, number_variables):
    patterns = np.zeros([n_patterns, number_variables])
    for index, pattern in enumerate(patterns):
        x1 = np.random.rand()
        x2 = np.random.rand()
        while x2 < 0.5 and x1 > 0.5:
            x1 = np.random.rand()
            x2 = np.random.rand()
        patterns[index] = [x1, x2]
    return patterns

def sigma_t(t, sigma0, tau0):
    return sigma0*math.exp(-t/tau0)

def eta_t(t, eta0, tau0):
    return eta0*math.exp(-t/tau0)


def get_winning_neuron(weights, selected_pattern):
    i0 = 0
    current_weight = weights[0,:]
    for index, weight in enumerate(weights):
        if np.linalg.norm(current_weight-selected_pattern) > np.linalg.norm(weight-selected_pattern):
            i0 = index
            current_weight = weight
    return i0

def calculate_space_position(i):
    ri = np.zeros(2)
    ri[0] = round(i/10)
    ri[1] = i % 10
    return ri


def calculate_neighbourhood(i, i0, sigma):
    ri = calculate_space_position(i)
    ri0 = calculate_space_position(i0)
    return math.exp(-abs(i-i0)**2/(2*sigma**2))


def update_weights(weights,selected_pattern, eta, sigma, i0):
    for index, weight in enumerate(weights):
        neighbour = calculate_neighbourhood(index, i0, sigma)
        weights[index] = weight + eta*neighbour*(selected_pattern-weight)

    return weights


def ordering_phase(weights, patterns):
    iterations = 10**3
    sigma0 = 5
    eta0 = 0.1
    tau0 = 300
    n_patterns = patterns.shape[0]
    for t in tqdm(iterable=range(iterations), desc="Ordering phase", mininterval=5):
        r = random.randint(0, n_patterns-1)
        selected_pattern = patterns[r]
        i0 = get_winning_neuron(weights, selected_pattern)
        eta = eta_t(t, eta0, tau0)
        sigma = sigma_t(t, sigma0, tau0)
        weights = update_weights(weights, selected_pattern, eta, sigma, i0)

    return weights

def convergence_phase(weights, patterns):
    eta = 0.01
    sigma = 0.9
    iterations = 2*10**4
    n_patterns = patterns.shape[0]
    for t in tqdm(iterable=range(iterations), desc="Convergence phase", mininterval=5):
        r = random.randint(0, n_patterns-1)
        selected_pattern = patterns[r]
        i0 = get_winning_neuron(weights, selected_pattern)
        weights = update_weights(weights, selected_pattern, eta, sigma, i0)
    return weights


if __name__ == '__main__':
    n_output = 100
    n_input = 2
    n_patterns = 1000
    patterns = create_random_patterns(n_patterns, 2)
    weights = initialize_weights(n_output, n_input)
    weights = ordering_phase(weights, patterns)
    np.save('weights_task1b_ordering', weights)
    np.save('patterns_task1b', patterns)
    weights = convergence_phase(weights, patterns)
    np.save('weights_task1b_convergence', weights)








import matplotlib.pyplot as plt
import numpy as np
def plotting_task1():
    weights_ordering = np.load('weights_task1b_ordering.npy')
    weights_convergence = np.load('weights_task1b_convergence.npy')
    patterns = np.load('patterns_task1b.npy')

    plt.figure(1)
    plt.scatter(patterns[:,0], patterns[:,1],color='r', alpha=0.4)
    plt.plot(weights_ordering[:,0], weights_ordering[:,1], color='b', marker='o')
    plt.title('Weights ordering phase')



    plt.figure(2)
    plt.scatter(patterns[:,0], patterns[:,1],color='r', alpha=0.4)
    plt.plot(weights_convergence[:,0], weights_convergence[:,1], color='b', marker='o')
    plt.title('Weights convergence phase')
    plt.show()

plotting_task1()

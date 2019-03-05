import nn
import numpy as np

if __name__ == '__main__':
    X = np.vstack((np.random.normal(0, 1, (50, 5)),
                   np.random.normal(3, 1, (50, 5))))
    y = np.vstack((np.zeros(50).reshape(-1, 1), np.ones(50).reshape(-1, 1)))
    neural_net = nn.NeuralNetwork(X, y, hidden_sizes=[3], activation='relu')
    neural_net.train(X, y, 'SGD', batch_size=1, eta=0.5,
                     momentum={'type': 'nesterov', 'alpha': 0.5})

    print neural_net.optimizer.error_per_epochs[0]
    print neural_net.optimizer.error_per_epochs[-1]
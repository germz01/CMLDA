import nn
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    X = np.vstack((np.random.normal(0, 1, (50, 5)),
                   np.random.normal(3, 1, (50, 5))))
    y = np.vstack((np.zeros(50).reshape(-1, 1), np.ones(50).reshape(-1, 1)))
    neural_net = nn.NeuralNetwork(X, y, hidden_sizes=[10],
                                  activation='sigmoid')

    if raw_input('TRAINING WITH SGD OR CGD?(SGD/CGD) ') == 'SGD':
        neural_net.train(X, y, 'SGD', batch_size=X.shape[0], eta=0.3,
                         epochs=1000,
                         momentum={'type': 'nesterov', 'alpha': 0.5})
    else:
        neural_net.train(X, y, 'CGD', max_epochs=10, error_goal=1e-4,
                         beta_m='hs')

    print '\n'
    print 'INITIAL ERROR: {}'.format(neural_net.optimizer.error_per_epochs[0])
    print 'FINAL ERROR: {}'.format(neural_net.optimizer.error_per_epochs[-1])
    print '\n'

    if raw_input('PLOT LEARNING CURVE?(Y/N) ') == 'Y':
        plt.plot(range(len(neural_net.optimizer.error_per_epochs)),
                 neural_net.optimizer.error_per_epochs)
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE per Epoch')
        plt.show()

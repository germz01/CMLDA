import json
import matplotlib.pyplot as plt
import metrics
import numpy as np
import pandas as pd
import validation as val

from datetime import datetime


class MonksTest(object):
    """
    This class is a wrapper for the test procedure on the three monks datasets

    Attributes
    ----------
    monks: list
        a list in which each element is a list containing the path to the
        training set and the test set for each one of the monks' datasets

    param_ranges: dict
        a dictionary containing the initialization's ranges for each one of the
        hyperparameters

    grid: validation.HyperGrid
        the hypergrid containing the sets of hyperparameters to be tested

    train_set: pandas.DataFrame
        the train set to test

    test_set: pandas.DataFrame
        the test set to test
    """

    def __init__(self, size=20, **kwargs):
        """
        The class' constructor.

        Parameters
        ----------
        size: int
            the hypergrid's size

        kwargs: dict
            a dictionary containing the initialization's ranges for each
            hyperparameter

        Returns
        -------
        """
        self.monks = [['../data/monks/monks-1_train_bin.csv',
                       '../data/monks/monks-1_test_bin.csv'],
                      ['../data/monks/monks-2_train_bin.csv',
                       '../data/monks/monks-2_test_bin.csv'],
                      ['../data/monks/monks-3_train_bin.csv',
                       '../data/monks/monks-3_test_bin.csv']]
        self.param_ranges = kwargs
        self.param_ranges_backup = self.param_ranges.copy()
        self.grid_size = size

    def test(self, dataset, repetitions=1, preliminary_search=False,
             to_fix=[]):
        """
        This function implements the testing procedure for the monks' datasets.
        It permits two kind of search for the best hyperparameters. The first
        kind of search simply performs a search using one HyperGrid, and
        selecting the set of hyperparameters which returns the best result.
        The second kind of search performs a deeper search, by searching first
        the best value for some hyperparameters, fixing them, and searching
        again the values for the remaing ones.

        Parameters
        ----------
        dataset: int or list
            either a single index or a list of indexes, each one representing
            a dataset in self.monks

        repetitions: int
            cross validation's repetitions
            (Default value = 1)

        preliminary_search: bool
            whether or not to execute a preliminary search for the best value
            for some hyperparameters, fix them, and search again for the
            remaining hyperparameters
            (Default value = False)

        to_fix: list
            a list of hyperparameters that must be fixed
            (Default value = [])

        size: int
            the new hypergrid's size for the new search for the best
            hyperparameters in the preliminary_search function
            (Default value = 0)

        Returns
        -------
        """
        if type(dataset) == int:
            assert dataset >= 0 and dataset <= 2
            dataset = [dataset]
        else:
            assert len(dataset) > 0 and len(dataset) <= 3

        for ds in dataset:
            print 'TESTING MONK DATASET {}\n'.format(ds + 1)

            self.train_set = pd.\
                read_csv(self.monks[ds][0], names=['class'] +
                         ['x{}'.format(j) for j in range(17)]).values
            self.test_set = pd.\
                read_csv(self.monks[ds][1], names=['class'] +
                         ['x{}'.format(j) for j in range(17)]).values

            self.grid = val.HyperGrid(self.param_ranges, size=self.grid_size,
                                      seed=datetime.now())
            self.selection = val.ModelSelectionCV(self.grid,
                                                  repetitions=repetitions)

            # PRELIMINARY SEARCH FOR SOME OF THE PARAMETERS ###################

            assert len(to_fix) != 0
            self.preliminary_search(ds, to_fix, repetitions)

            # SEARCHING FOR THE OTHER PARAMETERS ##############################

            self.selection.search(
                self.train_set[:, 1:], self.train_set[:, 0].reshape(-1, 1),
                save_results=True,
                fname='../data/model_selection_results_monk_{}.json'.
                format(ds + 1), par_name='REMAINING PARAMETERS')

            best_model = self.selection.\
                select_best_model(
                    self.train_set[:, 1:],
                    self.train_set[:, 0].reshape(-1, 1),
                    fname='../data/model_selection_results_monk_{}.json'.
                    format(ds + 1))

            y_pred = best_model.predict(self.test_set[:, 1:])
            y_pred = np.apply_along_axis(lambda x: 0 if x < .5 else 1, 1,
                                         y_pred).reshape(-1, 1)
            print '\n\n\n'
            bca = metrics.BinaryClassifierAssessment(self.test_set[:, 0].
                                                     reshape(-1, 1), y_pred)

            self.save_best_result(ds, best_model, bca)
            self.plot_best_result(ds, best_model)

            self.param_ranges = self.param_ranges_backup.copy()

    def preliminary_search(self, dataset, to_fix, repetitions):
        """
        This function implements a preliminary search in order to fix some
        of the best hyperparameters to the current best values and then
        search again for the remaining hyperparameters

        Parameters
        ----------
        dataset: int
            the dataset's index in self.monk

        to_fix: list
            a list of hyperparameters' names to fix

        repetitions: int
            cross validation's repetitions

        Returns
        -------
        """
        for prm in to_fix:
            self.selection.\
                search(self.train_set[:, 1:],
                       self.train_set[:, 0].reshape(-1, 1), save_results=True,
                       fname='../data/model_selection_results_monk_{}.json'.
                       format(dataset + 1), par_name=prm.upper())

            best_hyps = self.selection.\
                select_best_hyperparams(
                    fname='../data/model_selection_results_monk_{}.json'.
                    format(dataset + 1))
            self.param_ranges[prm] = best_hyps[0]['hyperparams'][prm]

            self.grid = val.HyperGrid(self.param_ranges, size=self.grid_size,
                                      seed=datetime.now())
            self.selection = val.ModelSelectionCV(self.grid, repetitions=1)

    def save_best_result(self, dataset, best_model, bca):
        """
        This function saves the results obtained on the test set by the
        model trained with the best set of hyperparameters as a JSON file.

        Parameters
        ----------
        dataset: int
            the dataset's index in self.monk

        best_model: nn.NeuralNetwork
            the neural network trained with the best set of hyperparameters

        bca: metrics.BinaryClassifierAssessment
            pass

        Returns
        -------
        """
        f = open('../data/best_results_monk_{}.json'.format(dataset + 1), 'w')
        results = {'hyperparameters': best_model.get_params(),
                   'accuracy': bca.accuracy, 'precision': bca.precision,
                   'recall': bca.recall, 'f1_score': bca.f1_score}
        json.dump(results, f, indent=4)

    def plot_best_result(self, dataset, best_model):
        """
        This function plots the learning curve for the best model obtained
        through the searching for the best set of hyperparameters.

        Parameters
        ----------
        dataset: int
            the dataset's index in self.monk

        best_model: nn.NeuralNetwork
            the neural network trained with the best set of hyperparameters

        Returns
        -------
        """
        plt.plot(range(len(best_model.error_per_epochs)),
                 best_model.error_per_epochs)
        plt.title("BEST RESULT'S LEARING CURVE FOR MONK {}".
                  format(dataset + 1))
        plt.ylabel('ERROR PER EPOCH')
        plt.xlabel('EPOCHS')
        plt.grid()
        plt.savefig('../images/learning_curve_monk_{}.pdf'.format(dataset + 1),
                    bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    param_ranges = {'eta': (0.1, 0.9), 'alpha': (0.1, 0.9),
                    'epsilon': (0.1, 0.5), 'batch_size': (1, 100),
                    'hidden_sizes': [(1, 3), (1, 3)],
                    'reg_lambda': (0.01, 0.4), 'reg_method': 'l2',
                    'epochs': (200, 1000)}

    monk_test = MonksTest(size=40, **param_ranges)
    monk_test.test(0, repetitions=1, preliminary_search=True,
                   to_fix=['epochs', 'batch_size'])


import numpy as np


class BinaryClassifierAssessment():
    """
    Binary Classifier Assessment

    Attributes
    ----------

    tp: int
        the number of examples classified as true positives

    tn: int
        the number of examples classified as true negatives

    fp: int
        the number of examples classified as false positives

    fn: int
        the number of examples classified as false negatives

    confusion_matrix: numpy.ndarray
        the confusion matrix

    accuracy: float
        the accuracy score

    precision: float
        the precision score

    recall: float
        the recall score

    f1_score: float
        the f1 score

    """
    def __init__(self, y_true, y_pred, printing=True):
        """
        Computes metrics for binary classification task.

        Parameters
        ----------
        y_true : numpy.ndarray or list
            ground truth values.

        y_pred : numpy.ndarray or list
            classifier predictions.

        printing: bool
            whether of not to print the classification's results
            (Default value = True)

        Returns
        -------
        """

        if (type(y_true) is not type(y_pred)):
            raise TypeError('y_true and y_pred must have the same type')

        if type(y_true) is np.ndarray and (y_true.shape != y_pred.shape):
            raise ValueError('y_true and y_pred must have the same shape')

        if (type(y_true) is list) and (len(y_pred) != len(y_true)):
            raise ValueError('y_true and y_pred must have the same length')

        if type(y_true) is list:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

        y_pred_bool = y_pred.astype(bool)
        y_true_bool = y_true.astype(bool)

        tp = np.logical_and(y_pred_bool, y_true_bool).sum()
        tn = np.logical_and(np.logical_not(y_pred_bool),
                            np.logical_not(y_true_bool)).sum()

        fp = np.logical_and(y_pred_bool,
                            np.logical_not(y_true_bool)).sum()

        fn = np.logical_and(np.logical_not(y_pred_bool),
                            y_true_bool).sum()

        # Tot = P + N
        P = tp + fn
        N = tn + fp

        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
        # print 'tp ' + str(tp)
        # print 'tn ' + str(tn)
        # print 'fp ' + str(fp)
        # print 'fn ' + str(fn)

        self.confusion_matrix = np.array((tp, fp, fn, tn)).reshape((2, 2))

        self.accuracy = float((tp+tn)) / len(y_true)  # P+N
        if tp < 1:
            self.precision = 0
            self.recall = 0
        else:
            self.precision = float(tp) / (tp+fp)
            self.recall = float(tp) / P
        # self.fp_rate = float(fp) / P

        if self.precision == 0 or self.recall == 0:
            self.f1_score = 0.
        else:
            self.f1_score = 2. / (1/self.precision + 1/self.recall)

        if printing:
            print self

    def __str__(self):

        info_matrix = """ TP: {:4} | FP: {:4} \n\
----------------------\n \
FN: {:4} | TN: {:4}\n""".format(self.tp, self.fp, self.fn, self.tn)

        return '-- Binary Classifier Assessment --\n'\
            + 'accuracy: {}\n'.format(self.accuracy)\
            + 'f1_score: {}\n'.format(self.f1_score)\
            + 'precision: {}\n'.format(self.precision)\
            + 'recall (tp rate): {}\n'.format(self.recall)\
            + 'confusion_matrix:\n'\
            + info_matrix

    def get_as_dict(self):
        """
        Returns assessment as a dict.

        Parameters
        ----------

        Returns
        -------
        A dictionary containing the assessment's parameters.
        """
        out = dict()
        out['confusion_matrix'] = self.confusion_matrix
        out['accuracy'] = self.accuracy
        out['f1_score'] = self.f1_score
        out['precision'] = self.precision
        out['recall'] = self.recall

        return out


def mse(y_true, y_pred):
    """
    Mean Square Error

    Parameters
    ----------
    y_true: numpy.ndarray
        the ground truth

    y_pred: numpy.ndarray
        the network's prediction

    Returns
    -------
    A float representing the mean square error.
    """
    if type(y_true) is list:
        y_true = np.array(y_true)
    if type(y_pred) is list:
        y_pred = np.array(y_pred)

    assert y_true.shape == y_pred.shape
    p = y_true.shape[0]

    return np.sum((y_true-y_pred)**2)/p


def mee(y_true, y_pred):
    """
    Mean Euclidean Error

    Parameters
    ----------
    y_true :

    y_pred :


    Returns
    -------

    """
    if type(y_true) is list:
        y_true = np.array(y_true)
    if type(y_pred) is list:
        y_pred = np.array(y_pred)

    assert y_true.shape == y_pred.shape
    p = y_true.shape[0]

    return np.sum(np.sqrt(np.sum((y_true-y_pred)**2, axis=1)))/p


if __name__ == "__main__":
    y_true = np.hstack((np.zeros(10), np.ones(10)))
    y_pred = np.hstack((np.zeros(12), np.ones(8)))
    ass = BinaryClassifierAssessment(y_true, y_pred)

    import sklearn.metrics as metrics

    print '-- sklearn results --'
    print (metrics.confusion_matrix(y_true, y_pred))
    print metrics.accuracy_score(y_true, y_pred)
    print metrics.f1_score(y_true, y_pred)
    print metrics.precision_score(y_true, y_pred)
    print metrics.recall_score(y_true, y_pred)

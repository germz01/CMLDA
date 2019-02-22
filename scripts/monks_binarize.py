import numpy as np
import utils as u
import csv

"""
The MONK's problems are a collection of three binary classification
problems over a six-attribute discrete domain. Each training/test data
is of the form

  <name>: <value1> <value2> <value3> <value4> <value5> <value6> -> <class>

where <name> is an ASCII-string, <value n> represents the value of
attribute # n, and <class> is either 0 or 1, depending on the class
this example belongs to. The attributes may take the following values:

attribute#1 :   {1, 2, 3}
attribute#2 :   {1, 2, 3}
attribute#3 :   {1, 2}
attribute#4 :   {1, 2, 3}
attribute#5 :   {1, 2, 3, 4}
attribute#6 :   {1, 2}

Thus, the six attributes span a space of 432=3x3x2x3x4x2 examples.

7. Attribute information:
    1. class: 0, 1
    2. a1:    1, 2, 3
    3. a2:    1, 2, 3
    4. a3:    1, 2
    5. a4:    1, 2, 3
    6. a5:    1, 2, 3, 4
    7. a6:    1, 2
    8. Id:    (A unique symbol for each instance)
"""


def read_monk(fname):
    """ read monk datasets """
    df = []
    with open(fname) as f:
        for line in f:
            row = (line.strip().split(" ")[: -1])
            pattern = [int(x) for x in row]
            df.append(pattern)
    out_set = np.array(df)

    return out_set


fpath = '../datasets/monks/'

names = ['monks-1_train',
         'monks-1_test',
         'monks-2_train',
         'monks-2_test',
         'monks-3_train',
         'monks-3_test']


datasets = {name: read_monk(fpath+name+'.csv') for name in names}

# monk1_train = read_monk('../datasets/monks/monks-1_train.csv')
# monk1_test = read_monk('../datasets/monks/monks-1_test.csv')

categories_sizes = [3, 3, 2, 3, 4, 2]

for name in names:
    """ split each monk set, binarize, merge again, export to csv """
    monk = datasets[name]

    y, X = np.hsplit(monk, [1])
    X_bin = u.binarize(X, categories_sizes)
    d_bin = np.hstack((y, X_bin))

    with open(fpath + name + '_bin.csv', 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        for row in range(d_bin.shape[0]):
            csv_writer.writerow(d_bin[row, :])

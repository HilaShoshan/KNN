from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import math


def run(points_df, labels_df, k, p):

    # Split the data randomly into 0.5 training-set and 0.5 testing-set
    X_train, X_test, y_train, y_test = train_test_split(points_df, labels_df, test_size=0.5, shuffle=True)

    y_pred = []  # predictions for each point on X_test

    # for each point on X_test:
    # find it's k-nearest neighbors according to lp distance, and predict it's value.

    for i in range(X_test.shape[0]):  # for each row

        nn_distance = math.inf  # save the minimum distance between the point to a neighbor on the training set
        nn_label = None

    return train_err, test_err

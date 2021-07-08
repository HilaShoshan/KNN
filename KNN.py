from sklearn.metrics import mean_squared_error
import numpy as np


def compute_lp(p, x, y):
    """
    compute lp norms, ||x-y||_p, for every p, even for infinity (Frechet distance).
    can be used to compute Euclidean distance (p=2) or Manhattan distance (p=1) etc.
    x, y are tuples from size d that represent 2 points on the d-dimensional space.
    """
    d = len(x)  # == len(y)
    differs = []  # list of all the |xi - yi|
    for i in range(d):
        differs.append(abs(x[i]-y[i]))
    if p == 'inf':
        return np.max(differs)
    differs = list(map(lambda diff: pow(diff, p), differs))  # map differs to |xi - yi|^p
    sum_d = np.sum(differs)
    return pow(sum_d, 1/p)


def predict(x, y, X_train, y_train, k, p):
    """
    given a 2D point (x,y):
    - find the k nearest neighbors on the training set
    - determine the point's prediction according to the majority vote
    - return it
    """
    k_min_distances = []  # list that saves the minimum distances between the point to another point on train
    k_labels = []         # the labels of the k points that are closest to (x,y)
    for i in range(X_train.shape[0]):
        # get the values of x and y and label for the neighbor
        x_n = X_train.iloc[i].loc['x']
        y_n = X_train.iloc[i].loc['y']
        label_n = y_train.iloc[i].loc['label']
        distance = compute_lp(p, (x, y), (x_n, y_n))
        if len(k_min_distances) < k:  # we've found < k neighbors so far
            k_min_distances.append(distance)
            k_labels.append(y_train.iloc[i].loc['label'])
        else:
            max_val = max(k_min_distances)
            if distance < max_val:  # replace the maximum with it
                max_ind = k_min_distances.index(max_val)
                del k_min_distances[max_ind]
                del k_labels[max_ind]
                k_min_distances.append(distance)
                k_labels.append(label_n)
    majority = (max(set(k_min_distances), key=k_min_distances.count))
    return majority


def run(X_train, X_test, y_train, y_test, k, p):

    y_pred = []  # predictions for each point on X_test

    # for each point on X_test:
    # find it's k-nearest neighbors according to lp distance, and predict it's value.

    for i in range(X_test.shape[0]):  # for each row
        x = X_test.iloc[i].loc['x']
        y = X_test.iloc[i].loc['y']
        prediction = predict(x, y, X_train, y_train, k, p)
        y_pred.append(prediction)  # add the prediction at the end of the list

    test_err = mean_squared_error(y_test, y_pred)
    print("____________________________________________________\n"
          "The test error of classifier with parameters k =", k, " and p =", p, "is:\n",
          test_err)

    if k == 1:
        # the nearest neighbor of a point in the training set is the point itself (with distance=0, for every p).
        # so of course the label we'll choose for it is the true label - we are never make a mistake here.
        train_err = 0

    else:
        # repeat the process that for the X_train set now

        y_pred = []  # predictions for each point on X_train

        for i in range(X_train.shape[0]):  # for each row
            x = X_train.iloc[i].loc['x']
            y = X_train.iloc[i].loc['y']
            prediction = predict(x, y, X_train, y_train, k, p)  # the base set is still the training set
            y_pred.append(prediction)  # add the prediction at the end of the list

        train_err = mean_squared_error(y_train, y_pred)

    print("The empirical error of classifier with parameters k =", k, " and p =", p, "is:\n",
            train_err)

    return train_err, test_err

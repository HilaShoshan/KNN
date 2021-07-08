# python version: 3.7

import pandas as pd
import matplotlib.pyplot as plt
from KNN import *


# create a df of points (x,y) and their labels
df = pd.read_table('rectangle.txt', delim_whitespace=True, names=('x', 'y', 'label'))
points_df = df[['x', 'y']].copy()
labels_df = df[['label']].copy()


def average(lst):
    return sum(lst) / len(lst)


def plot_err(vals_range, train_err, test_err):
    """
    plot the k errors as a function of k values
    """
    plt.title("The average empirical and true errors as a function of k")
    plt.xlabel('k values')
    plt.ylabel('prediction error')
    plt.plot(vals_range, train_err, color='blue', linewidth=3, label='train error')
    plt.plot(vals_range, test_err, color='red', linewidth=3, label='test error')
    plt.legend()
    plt.show()


def main():
    """
    print(df.head(5))
    print(df.shape)

    print("points: \n", points_df.head(5))
    print(points_df.shape)

    print("labels: \n", labels_df.head(5))
    print(labels_df.shape)
    """

    k_list = [1, 3, 5, 7, 9]   # the k values that will be use on the KNN classifier
    p_list = [1, 2, 'inf']     # the p values for computing lp distance

    k_train_err = [[] for _ in range(5)]  # list of 5 lists which contain all the errors for k = 1,3,5,7,9 accordingly
    k_test_err = [[] for _ in range(5)]
    p_train_err = [[] for _ in range(3)]  #  "   "  3   "     "      "     "    "    "    "  p = 1,2,inf       "
    p_test_err = [[] for _ in range(3)]

    # k_p_list = list(itertools.product(k_list, p_list))   # all the combinations of k and p
    # k_p_errors = []

    iterations = 100

    for i in range(iterations):

        print("################################################\n"
              "iteration #", i, "/", iterations, "...\n"
              "################################################")

        # Split the data randomly into 0.5 training-set and 0.5 testing-set
        X_train, X_test, y_train, y_test = train_test_split(points_df, labels_df, test_size=0.5, shuffle=True)

        k_ind = 0
        for k in k_list:
            p_ind = 0
            for p in p_list:

                # evaluate the KNN classifier on the test set, under the lp distance
                train_err, test_err = run(X_train, X_test, y_train, y_test, k, p)

                k_train_err[k_ind].append(train_err)
                k_test_err[k_ind].append(test_err)
                p_train_err[p_ind].append(train_err)
                p_test_err[p_ind].append(test_err)

                p_ind += 1
            k_ind += 1
        print('\n')

    # map each sub-list of errors to it's average
    k_train_err = list(map(lambda lst: average(lst), k_train_err))
    k_test_err = list(map(lambda lst: average(lst), k_test_err))
    p_train_err = list(map(lambda lst: average(lst), p_train_err))
    p_test_err = list(map(lambda lst: average(lst), p_test_err))

    # print the average empirical and true errors for each p and k

    for k_ind in range(k_list):
        print("k = ", k_list[k_ind])
        print("average train error: ", k_train_err[k_ind])
        print("average test error: ", k_test_err[k_ind])
        print("---")

    print()

    for p_ind in range(p_list):
        print("p = ", p_list[p_ind])
        print("average train error: ", p_train_err[p_ind])
        print("average test error: ", p_test_err[p_ind])
        print("---")

    plot_err([*range(1, 9)], k_train_err, k_test_err)


if __name__ == '__main__':
    main()

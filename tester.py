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


def main():
    x = (3,1)
    y = (6,3)
    print(compute_lp(2,x,y))


if __name__ == '__main__':
    main()
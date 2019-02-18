import numpy as np
import matplotlib.pyplot as plt


def kl_divergence(p, q, dx=0.001):
    p = p + dx
    q = q + dx
    return np.sum(p * (np.log(p / q)))


def main():
    a = np.array([10,0,0,0,0,0,0])
    b = np.array([0,0,5,0,0,0,5])
    print(a)
    print(b)

    target = np.array([9,1,0,0,0,0,0])
    kl = kl_divergence(a, target)
    print(kl)

    kl = kl_divergence(b, target)
    print(kl)


if __name__ == '__main__':
    main()

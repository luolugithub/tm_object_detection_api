import numpy as np


def kl_divergence(p, q, dx=0.001):
    p = p + dx
    q = q + dx
    return np.sum(p * (np.log(p / q)))


def main():
    a = np.array([1,1,1,1,1,1,1])
    b = np.array([1,1,2,0,1,1,1])
    print(a)
    print(b)

    kl = kl_divergence(a, b)
    print(kl)
    


if __name__ == '__main__':
    main()

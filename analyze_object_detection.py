import sys
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


recipe_range = [(0, 350),
                (351, 670),
                (671, 1950),
                (1951, 2880),
                (1951, 2880),
                (2881, 3250),
                (3251, 3440),
                (3441, 3540),
                (3541, 3600),
                (3601, 7350),
                (7351, 7440),
                (7441, 13100),
                (13101, 13200),
                (13201, 13300),
                (13301, 13500),
                (13501, 14380),]


def kl_divergence(p, q, dx=0.001):
    p = p + dx
    q = q + dx
    return np.sum(p * (np.log(p / q)))


def main(argv):
    df = pd.read_csv(argv, header=None, dtype='int')
    print(df)
    print(len(df))

    top_category = df[1]
    print(top_category)

    plt.plot(top_category)
    plt.show()

    for i in range(len(recipe_range)):
        print(i)
        print(recipe_range[i])
        # plt.plot(top_category[recipe_range[i][0]:recipe_range[i][1]])
        # plt.show()
        # plt.hist(top_category[recipe_range[i][0]:recipe_range[i][1]])
        # plt.show()

    start_pos = 0
    end_pos = 100

    moving_range_length = int(len(top_category) / end_pos) + 1
    print('moving_range_length', moving_range_length)
    for i in range(moving_range_length):
        print(start_pos)
        print(end_pos)
        plt.hist(top_category[start_pos:end_pos], bins=7)
        plt.xlim([1,7])
        plt.show()
        start_pos += 100
        end_pos += 100
        if i == 0:
            p = np.ones(7)
            q = np.ones(7)
        else:
            if end_pos > len(top_category):
                end_pos = len(top_category)            
            current_eval = top_category[start_pos:end_pos]
            print('1:', len(current_eval[current_eval == 1]))
            print('2:', len(current_eval[current_eval == 2]))
            print('3:', len(current_eval[current_eval == 3]))
            print('4:', len(current_eval[current_eval == 4]))
            print('5:', len(current_eval[current_eval == 5]))
            print('6:', len(current_eval[current_eval == 6]))
            print('7:', len(current_eval[current_eval == 7]))
            q = np.array([
                          len(current_eval[current_eval == 1]),
                          len(current_eval[current_eval == 2]),
                          len(current_eval[current_eval == 3]),
                          len(current_eval[current_eval == 4]),
                          len(current_eval[current_eval == 5]),
                          len(current_eval[current_eval == 6]),
                          len(current_eval[current_eval == 7]),
                          ])
            print('p')
            print(p)
            print('q')
            print(q)
            kl = kl_divergence(p, q)
            print('kl_divergence')
            print(kl)
            p = q
            

    # recipe_01 = top_category[0:350]
    # recipe_02 = top_category[351:670]
    # recipe_03 = top_category[671:1950]
    # recipe_04 = top_category[1951:2880]
    # recipe_05 = top_category[2881:3250]
    # recipe_06 = top_category[3251:3440]
    # recipe_07 = top_category[3441:3540]
    # recipe_08 = top_category[3541:3600]
    # recipe_09 = top_category[3601:7350]
    # recipe_10 = top_category[7351:7440]
    # recipe_11 = top_category[7441:13100]
    # recipe_12 = top_category[13101:13200]
    # recipe_13 = top_category[13201:13300]
    # recipe_14 = top_category[13301:13500]
    # recipe_15 = top_category[13501:14380]
    

if __name__ == '__main__':
    argv = sys.argv[1]
    main(argv)

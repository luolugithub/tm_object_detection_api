import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

recipe_range = [
    (0, 350),
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
    (13501, 14380),
]

state = {
    0: 'フライパンを置いて油を引く',
    1: '餃子を並べる',
    2: '片栗粉を入れる',
    3: '蓋を閉める',
    4: '蒸し焼きにする',
    5: '蓋を取る',
    6: '水分を飛ばして完成',
}


def kl_divergence(p, q, dx=0.001):
    p = p + dx
    q = q + dx
    return np.sum(p * (np.log(p / q)))


def logging_histogram(df):
    print('1:', len(df[df == 1]))
    print('2:', len(df[df == 2]))
    print('3:', len(df[df == 3]))
    print('4:', len(df[df == 4]))
    print('5:', len(df[df == 5]))
    print('6:', len(df[df == 6]))
    print('7:', len(df[df == 7]))

    return


def dataframe_to_histogram(df):
    histogram = np.array([
        len(df[df == 1]),
        len(df[df == 2]),
        len(df[df == 3]),
        len(df[df == 4]),
        len(df[df == 5]),
        len(df[df == 6]),
        len(df[df == 7]),
    ])

    return histogram


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

    window_size = 10
    start_pos = 0
    end_pos = window_size
    kld_scores = []
    change_state_hash = {}
    change_state_threshold = 70
    current_state_num = 0
    current_state = state[current_state_num]
    state_history = {}

    # processing for each 100 frames
    # all
    moving_range_length = int(len(top_category) / end_pos) + 1
    # # setting manually
    # moving_range_length = 300
    print('moving_range_length', moving_range_length)
    for i in range(moving_range_length):
        print(start_pos)
        print(end_pos)

        # # visualize histogram
        # plt.hist(top_category[start_pos:end_pos], bins=7)
        # plt.xlim([1,7])
        # plt.show()

        # initialize
        if i == 0:
            p = np.ones(7)
            q = np.ones(7)
        else:
            if end_pos > len(top_category):
                end_pos = len(top_category)
            current_eval = top_category[start_pos:end_pos]

            logging_histogram(current_eval)

            q = dataframe_to_histogram(current_eval)
            # # plot
            # plt.hist(top_category[start_pos:end_pos])
            # plt.xlim([1, 7])
            # plt.show()

            print('p :', p)
            print('q :', q)
            kl = kl_divergence(p, q)
            kld_scores.append(kl)
            print('kl_divergence :', kl)
            p = q

            if kl > change_state_threshold:
                current_state = state[current_state_num + 1]
                current_state_num += 1
                memo_state = {start_pos: current_state}
                state_history.update(memo_state)

        start_pos += window_size
        end_pos += window_size
    plt.plot(kld_scores)
    plt.show()
    print(state_history)
    print(len(state_history))

    # state1
    # recipe_01 = top_category[0:350]            frying-pan
    # recipe_02 = top_category[351:670]          frying-pan, cup, spoon
    # recipe_03 = top_category[671:1950]         frying-pan
    # state2
    # recipe_04 = top_category[1951:2880]        frying-pan, gyoza, hand
    # recipe_05 = top_category[2881:3250]        frying-pan, gyoza
    # state3
    # recipe_06 = top_category[3251:3440]        frying-pan, gyoza, cup, hand
    # recipe_07 = top_category[3441:3540]        frying-pan, gyoza
    # state4
    # recipe_08 = top_category[3541:3600]        frying-pan, gyoza, close, hand
    # recipe_09 = top_category[3601:7250]        close
    # state5
    # recipe_09 = top_category[7251:7350]        open
    # state6
    # recipe_10 = top_category[7351:7440]        open, hand
    # state7
    # recipe_11 = top_category[7441:13100]       gyoza
    # recipe_12 = top_category[13101:13200]      ?
    # recipe_13 = top_category[13201:13300]      ?
    # recipe_14 = top_category[13301:13500]      ?
    # recipe_15 = top_category[13501:14380]      ?


if __name__ == '__main__':
    argv = sys.argv[1]
    main(argv)

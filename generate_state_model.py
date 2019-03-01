import argparse
import os
import pickle

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


np.random.seed(0)


def bayes_training(training_data, num_classes, model_savepath):
    # count for each categories
    count_dict = {}
    for i in range(1, num_classes+1):
        data_counts = len(training_data[training_data[1] == i])
        count_category = {i: data_counts}
        count_dict.update(count_category)
    # print('count_dict')
    # print(count_dict)

    # convert count to probabilities
    prob_list = []
    for k, v in sorted(count_dict.items()):
        print(k, v)
        prob_list.append(v)
    prob_list = np.array(prob_list)
    # print(prob_list)
    prob_list = prob_list / len(training_data)
    # print(prob_list)

    # generate random training data
    training_data_list = [np.random.multinomial(i+1, prob_list)
                          for i in range(1000)]
    # print(training_data_list)

    # training
    alpha = [1] * num_classes # hyper parameter alpha = [1, 1, 1, 1, 1, 1]
    pred_list = []
    m = np.diag(np.ones(num_classes))
    for data in training_data_list:
        prior = np.random.dirichlet([data[i] + alpha[i] for i in range(num_classes)])
        # print('prior')
        # print(prior)
        pred = [stats.multinomial.pmf(i, 1, [p for p in prior]) for i in m]
        pred_list.append(pred)
    pred_list = np.array(pred_list)

    count = 1000
    for i in range(num_classes):
        plt.plot(np.arange(1, count+1, 1), pred_list[:, i], label=str(i+1))
    plt.xlabel('count')
    plt.ylabel('pred')
    plt.ylim(0, 1.1)
    plt.legend()   
    plot_savepath, _ = os.path.splitext(model_savepath)
    plot_savepath = os.path.join(plot_savepath + '.png')
    plt.savefig(plot_savepath)
    plt.show()

    # model = np.array([
    #     pred_list[-1][0],
    #     pred_list[-1][1],
    #     pred_list[-1][2],
    #     pred_list[-1][3],
    #     pred_list[-1][4],
    #     pred_list[-1][5],
    # ])
    model = np.array([pred_list[-1][c] for c in range(num_classes)])
    
    # XXX
    # model = [np.array([pred_list[-1][c]]) for c in range(num_classes)]
    
    with open(model_savepath, 'wb') as m:
        pickle.dump(model, m)
    
    return


def main(log_dir, num_classes):
    # # define filepath
    data_dst_dir = 'model/data'
    logdir_name = os.path.basename(log_dir)
    print('logdir_name')
    print(logdir_name)
    data_dst_dir = os.path.join(data_dst_dir, logdir_name)
    if os.path.isdir(data_dst_dir) is False:
        os.makedirs(data_dst_dir)

    label_dir = 'model/label'

    model_dst_dir = os.path.join('model', logdir_name)
    if os.path.isdir(model_dst_dir) is False:
        os.makedirs(model_dst_dir)
    
    # with open(label_file, 'r') as f:
    #     lines = f.readlines()

    # label_dict = {}
    # for line in lines:
    #     line = line.split()[0]
    #     category = line.split(':')[0]
    #     position = line.split(':')[1]
    #     start_pos = position.split(',')[0]
    #     end_pos = position.split(',')[1]
    #     label = {category: (start_pos, end_pos)}
    #     label_dict.update(label)
    # print(label_dict)

    # df = pd.read_csv(observation, dtype='int', header=None)
    # df_top = df[0]
    # print(df_top.head())

    # for k, v in label_dict.items():
    #     print(k)
    #     category_dir = os.path.join(data_dst_dir, k)
    #     if os.path.isdir(category_dir) is False:
    #         os.makedirs(category_dir)
    #     print(v[0], v[1])

    #     output_df = df_top[int(v[0]):int(v[1])]
    #     observation_fname = os.path.basename(observation)
    #     output_df_path = os.path.join(category_dir, observation_fname)
    #     output_df.to_csv(output_df_path)

    # generate label data
    log_files = os.listdir(log_dir)
    split_exts = [f.replace('.csv', '') for f in log_files]
    test_number = [('_').join(f.split('_')[1:]) for f in split_exts]
    test_number = set(test_number)
    print(test_number)
    for i in sorted(test_number):
        print(i)
        label_file = os.path.join(label_dir, 'labels_' + str(i) + '.txt')
        # print(label_file)
        with open(label_file, 'r') as f:
            lines = f.readlines()
        label_dict = {}
        for line in lines:
            line = line.split()[0]
            category = line.split(':')[0]
            position = line.split(':')[1]
            start_pos = position.split(',')[0]
            end_pos = position.split(',')[1]
            label = {category: (start_pos, end_pos)}
            label_dict.update(label)
        # print(label_dict)

        observation_file = os.path.join(log_dir, 'classes_' + str(i) + '.csv')
        df = pd.read_csv(observation_file, dtype='int', header=None)
        df_top = df[0]

        for k, v in label_dict.items():
            print(k)
            category_dir = os.path.join(data_dst_dir, k)
            if os.path.isdir(category_dir) is False:
                os.makedirs(category_dir)
            output_df = df_top[int(v[0]):int(v[1])]
            observation_fname = os.path.basename(observation_file)
            output_df_path = os.path.join(category_dir, observation_fname)
            output_df.to_csv(output_df_path)

    # # generate_model
    data_categories = os.listdir(data_dst_dir)
    print('data_dst_dir')
    print(data_dst_dir)
    print('data_categories')
    print(data_categories)
    # print(data_categories)
    for c in data_categories:
        model_category_dir = os.path.join(model_dst_dir, c)
        # print(model_category_dir)
        if os.path.isdir(model_category_dir) is False:
            os.makedirs(model_category_dir)
        current_data_category = os.path.join(data_dst_dir, c)
        target_files = os.listdir(current_data_category)
        sampling = pd.DataFrame({})
        for f in target_files:
            current_data_file = os.path.join(current_data_category, f)
            print('current_data_file')
            print(current_data_file)
            if os.path.isfile(current_data_file):
                current_sampling = pd.read_csv(
                    current_data_file,
                    index_col=0,
                    header=None,
                )
                # current_sampling.hist()
                plt.show()
                sampling = pd.concat([sampling, current_sampling], axis=0)
            else:
                pass
        print('################ {} ################'.format(c))
        model_savepath = os.path.join(model_category_dir, str(c) + '.pkl')
        bayes_training(sampling, num_classes, model_savepath)
        sampling_path = os.path.join(model_category_dir, str(c) + '.csv')
        sampling.to_csv(sampling_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='python generate_state_model.py --log_dir=log/20190222_fix-lid_125833 --num_classes=6')
    parser.add_argument(
        '--log_dir',
        dest='log_dir',
        type=str,
        default=None,
        help='please enter the log directory path')
    parser.add_argument(
        '--num_classes',
        dest='num_classes',
        type=int,
        default=6,
        help='please enter the number of classes')
    argv = parser.parse_args()
    main(argv.log_dir, argv.num_classes)

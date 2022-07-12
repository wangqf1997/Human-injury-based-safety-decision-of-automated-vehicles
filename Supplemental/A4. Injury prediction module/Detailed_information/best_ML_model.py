# -*- coding: utf-8 -*-
'''
-------------------------------------------------------------------------------------------------
This code accompanies the paper titled "Human injury-based safety decision of automated vehicles"
Author: Qingfan Wang, ***,
Corresponding author: Bingbing Nie (nbb@tsinghua.edu.cn)
-------------------------------------------------------------------------------------------------
'''


import argparse
import random
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import confusion_matrix


__author__ = "Qingfan Wang"


def load_data(data):
    ''' Load and process the crash data. '''

    # Divide the dataset into three parts: training, validation, and testing.
    shuffle = np.random.permutation(len(data))
    data = data[shuffle]
    data_train = data[:int(len(data) * 0.7)]
    data_test = data[int(len(data) * 0.7):int(len(data) * 0.85)]
    data_val = data[int(len(data) * 0.85):]

    return data_val, data_test


def evaluate_model(true, pred, pri, case):
    ''' Evaluate the model. '''

    accu = 100. * (1 - np.count_nonzero(true - pred) / float(len(true)))
    conf_mat = confusion_matrix(true, pred)
    G_mean = geometric_mean_score(true, pred)
    report = classification_report_imbalanced(true, pred, digits=3)
    if pri:
        if case == 'val':
            print('Validation | Accuracy: ' + str(np.around(accu, 1)) + '%')
            print('Validation | G-mean: ' + str(np.around(G_mean, 3)))
            print(conf_mat)
            print(report)
        elif case == 'test':
            print('Test | Accuracy: ' + str(np.around(accu, 1)) + '%')
            print('Test | G-mean: ' + str(np.around(G_mean, 3)))
            print(conf_mat)
            print(report)


def test_SVM(data_val, data_test, opt):
    ''' Test the SVM-based occupant injury prediction model. '''

    # Load the model with the highest accuracy.
    SVM = joblib.load('Saved_Model_params/Best_params/model_SVM_%s.m' % opt.re_samp)

    # Obtain the best prediction performance.
    print('SVM performance:')
    evaluate_model(data_val[:, -1], SVM.predict(data_val[:, :-1]), opt.print_inf, 'val')
    evaluate_model(data_test[:, -1], SVM.predict(data_test[:, :-1]), opt.print_inf, 'test')


def test_DT(data_val, data_test, opt):
    ''' Test the DT-based occupant injury prediction model. '''

    # Load the model with the highest accuracy.
    DT = joblib.load('Saved_Model_params/Best_params/model_DT_%s.m' % opt.re_samp)

    # Obtain the best prediction performance.
    print('DT performance:')
    evaluate_model(data_val[:, -1], DT.predict(data_val[:, :-1]), opt.print_inf, 'val')
    evaluate_model(data_test[:, -1], DT.predict(data_test[:, :-1]), opt.print_inf, 'test')


def test_KNN(data_val, data_test, opt):
    ''' Test the KNN-based occupant injury prediction model. '''

    # Load the model with the highest accuracy.
    KNN = joblib.load('Saved_Model_params/Best_params/model_KNN_%s.m' % opt.re_samp)

    # Obtain the best prediction performance.
    print('KNN performance:')
    evaluate_model(data_val[:, -1], KNN.predict(data_val[:, :-1]), opt.print_inf, 'val')
    evaluate_model(data_test[:, -1], KNN.predict(data_test[:, :-1]), opt.print_inf, 'test')


def test_NB(data_val, data_test, opt):
    ''' Test the NB-based occupant injury prediction model. '''

    # Load the model with the highest accuracy.
    NB = joblib.load('Saved_Model_params/Best_params/model_NB_%s.m' % opt.re_samp)

    # Obtain the best prediction performance.
    print('NB performance:')
    evaluate_model(data_val[:, -1], NB.predict(data_val[:, :-1]), opt.print_inf, 'val')
    evaluate_model(data_test[:, -1], NB.predict(data_test[:, :-1]), opt.print_inf, 'test')


def test_AB(data_val, data_test, opt):
    ''' Test the AB-based occupant injury prediction model. '''

    # Load the model with the highest accuracy.
    AB = joblib.load('Saved_Model_params/Best_params/model_AB_%s.m' % opt.re_samp)

    # Obtain the best prediction performance.
    print('AB performance:')
    evaluate_model(data_val[:, -1], AB.predict(data_val[:, :-1]), opt.print_inf, 'val')
    evaluate_model(data_test[:, -1], AB.predict(data_test[:, :-1]), opt.print_inf, 'test')


def main():
    ''' Train and test the machine-learning occupant injury prediction models. '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--rand_seed', type=int, default=123, help='Random seed')
    parser.add_argument('--re_samp', type=str, default='CS', help='Re-sampling methods: US, OS, CS')
    parser.add_argument('--print_inf', action='store_false', help='print the information of the training process')
    opt = parser.parse_args()

    # Define the random seed.
    seed = opt.rand_seed
    np.random.seed(seed)
    random.seed(seed)

    # Load the real-world crash data.
    data = np.load('dataset/data_pro.npy')
    data_val, data_test = load_data(data)

    # Test the five machine-learning models with the best performance.
    test_SVM(data_val, data_test, opt)
    test_DT(data_val, data_test, opt)
    test_KNN(data_val, data_test, opt)
    test_NB(data_val, data_test, opt)
    test_AB(data_val, data_test, opt)


if __name__ == "__main__":
    main()

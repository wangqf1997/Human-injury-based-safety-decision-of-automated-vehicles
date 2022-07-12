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


def load_data(data, resample, seed):
    ''' Load and process the crash data. '''

    # Divide the dataset into three parts: training, validation, and testing.
    shuffle = np.random.permutation(len(data))
    data = data[shuffle]
    data_train = data[:int(len(data) * 0.7)]
    data_test = data[int(len(data) * 0.7):int(len(data) * 0.85)]
    data_val = data[int(len(data) * 0.85):]

    # Data re-sampling to reduce imbalance problems.
    if resample == 'US':
        enn = EditedNearestNeighbours(sampling_strategy=[0], n_neighbors=5, kind_sel="all")
        X_enn, y_enn = enn.fit_resample(data_train[:, :-1], data_train[:, -1])
        data_train = np.zeros((len(X_enn), 10))
        data_train[:, :-1], data_train[:, -1] = X_enn, y_enn
        enn = EditedNearestNeighbours(sampling_strategy=[1], n_neighbors=3, kind_sel="all")
        X_enn, y_enn = enn.fit_resample(data_train[:, :-1], data_train[:, -1])
        data_train = np.zeros((len(X_enn), 10))
        data_train[:, :-1], data_train[:, -1] = X_enn, y_enn

    elif resample == 'OS':
        smo = SMOTE(random_state=seed, sampling_strategy={1: 1900, 2: 1400, 3: 1000})
        X_smo, y_smo = smo.fit_resample(data_train[:, :-1], data_train[:, -1])
        data_train = np.zeros((len(X_smo), 10))
        data_train[:, :-1], data_train[:, -1] = X_smo, y_smo

    elif resample == 'CS':
        smo = SMOTE(random_state=seed, sampling_strategy={1: 2000, 2: 1200, 3: 800})
        enn = EditedNearestNeighbours(sampling_strategy=[0, 1, 2, 3], n_neighbors=3)
        smo_enn = SMOTEENN(random_state=seed, smote=smo, enn=enn)
        X_enn, y_enn = smo_enn.fit_resample(data_train[:, :-1], data_train[:, -1])
        data_train = np.zeros((len(X_enn), 10))
        data_train[:, :-1], data_train[:, -1] = X_enn, y_enn

    else:
        print('Wrong re-sampling method!')
        return

    return data_train, data_val, data_test


def evaluate_model(true, pred, pri):
    ''' Evaluate the model. '''
    accu = 100. * (1 - np.count_nonzero(true - pred) / float(len(true)))
    conf_mat = confusion_matrix(true, pred)
    G_mean = geometric_mean_score(true, pred)
    report = classification_report_imbalanced(true, pred, digits=3)
    if pri:
        print('Test | Accuracy: ' + str(np.around(accu, 1)) + '%')
        print('Test | G-mean: ' + str(np.around(G_mean, 3)))
        print(conf_mat)
        print(report)


def train_SVM(data_train, data_val, data_test, opt):
    ''' Train and test the SVM-based occupant injury prediction model. '''

    # Define the parameter matrix for grid search.
    C_list = [1, 10, 100]
    kernel_list = ['rbf', 'sigmoid'] * 3
    Gamma_list = [0.1, 0.01, 0.001, 'auto'] * 6
    best_G_mean, best_i = 0, 0

    # Start the grid search for the optimal parameter combination.
    for i in range(24):
        # Obtain parameters.
        C = C_list[i // 8]
        kernel = kernel_list[i // 4]
        Gamma = Gamma_list[i]

        # Load the SVM-based model.
        SVM = SVC(C=C, kernel=kernel, gamma=Gamma)

        # Train the model.
        SVM.fit(data_train[:, :-1], data_train[:, -1])

        # Calculate the prediction accuracy.
        pred = SVM.predict(data_val[:, :-1])
        true = data_val[:, -1]
        G_mean = geometric_mean_score(true, pred)

        # Save the model with the highest accuracy.
        if G_mean > best_G_mean:
            best_G_mean = G_mean
            best_i = i

    # Load the model with the highest accuracy.
    C = C_list[best_i // 8]
    kernel = kernel_list[best_i // 4]
    Gamma = Gamma_list[best_i]
    SVM = SVC(C=C, kernel=kernel, gamma=Gamma)
    SVM.fit(data_train[:, :-1], data_train[:, -1])

    # Save the optimal model parameters.
    if opt.save_para:
        joblib.dump(SVM, "Saved_Model_params\model_SVM_%s.m" % opt.re_samp)

    # Obtain the prediction performance.
    evaluate_model(data_test[:, -1], SVM.predict(data_test[:, :-1]), opt.print_inf)

    return SVM


def train_DT(data_train, data_val, data_test, opt):
    ''' Train and test the DT-based occupant injury prediction model. '''

    # Define the parameter matrix for grid search.
    criterion_list = ['entropy', 'gini']
    splitter_list = ['best', 'random'] * 2
    maxdepth_list = [None, 10, 20, 50] * 4
    best_G_mean, best_i = 0, 0

    # Start the grid search for the optimal parameter combination.
    for i in range(16):
        # Obtain parameters.
        criterion = criterion_list[i // 8]
        splitter = splitter_list[i // 4]
        maxdepth = maxdepth_list[i]

        # Load the SVM-based model.
        DT = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=maxdepth)

        # Train the model.
        DT.fit(data_train[:, :-1], data_train[:, -1])

        # Calculate the prediction accuracy.
        pred = DT.predict(data_val[:, :-1])
        true = data_val[:, -1]
        G_mean = geometric_mean_score(true, pred)

        # Save the model with the highest accuracy.
        if G_mean > best_G_mean:
            best_G_mean = G_mean
            best_i = i

    # Load the model with the highest accuracy.
    criterion = criterion_list[best_i // 8]
    splitter = splitter_list[best_i // 4]
    maxdepth = maxdepth_list[best_i]
    DT = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=maxdepth)
    DT.fit(data_train[:, :-1], data_train[:, -1])

    # Save the optimal model parameters.
    if opt.save_para:
        joblib.dump(DT, "Saved_Model_params\model_DT_%s.m" % opt.re_samp)

    # Obtain the prediction performance.
    evaluate_model(data_test[:, -1], DT.predict(data_test[:, :-1]), opt.print_inf)

    return DT


def train_KNN(data_train, data_val, data_test, opt):
    ''' Train and test the KNN-based occupant injury prediction model. '''

    # Define the parameter matrix for grid search.
    n_neighbors_list = [3, 5, 10]
    algorithm_list = ['ball_tree', 'kd_tree', 'brute'] * 3
    p_list = [1, 2, 3] * 9
    best_G_mean, best_i = 0, 0

    # Start the grid search for the optimal parameter combination.
    for i in range(27):
        # Obtain parameters.
        n_neighbors = n_neighbors_list[i // 9]
        algorithm = algorithm_list[i // 3]
        p = p_list[i]

        # Load the KNN-based model.
        KNN = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm, p=p)

        # Train the model.
        KNN.fit(data_train[:, :-1], data_train[:, -1])

        # Calculate the prediction accuracy.
        pred = KNN.predict(data_val[:, :-1])
        true = data_val[:, -1]
        G_mean = geometric_mean_score(true, pred)

        # Save the model with the highest accuracy.
        if G_mean > best_G_mean:
            best_G_mean = G_mean
            best_i = i

    # Load the model with the highest accuracy.
    n_neighbors = n_neighbors_list[best_i // 9]
    algorithm = algorithm_list[best_i // 3]
    p = p_list[best_i]
    KNN = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm, p=p)
    KNN.fit(data_train[:, :-1], data_train[:, -1])

    # Save the optimal model parameters.
    if opt.save_para:
        joblib.dump(KNN, "Saved_Model_params\model_KNN_%s.m" % opt.re_samp)

    # Obtain the prediction performance.
    evaluate_model(data_test[:, -1], KNN.predict(data_test[:, :-1]), opt.print_inf)

    return KNN


def train_NB(data_train, data_val, data_test, opt):
    ''' Train and test the NB-based occupant injury prediction model. '''

    # Load the model with the highest accuracy.
    NB = GaussianNB()
    NB.fit(data_train[:, :-1], data_train[:, -1])

    # Save the optimal model parameters.
    if opt.save_para:
        joblib.dump(NB, "Saved_Model_params\model_NB_%s.m" % opt.re_samp)

    # Obtain the prediction performance.
    evaluate_model(data_test[:, -1], NB.predict(data_test[:, :-1]), opt.print_inf)

    return NB


def train_AB(data_train, data_val, data_test, opt, base_estimator_list, seed):
    ''' Train and test the AB-based occupant injury prediction model. '''

    # Define the parameter matrix for grid search.
    base_estimator_list = base_estimator_list
    n_estimators_list = [3, 10, 30] * 4
    learning_rate_list = [0.1, 0.01] * 12
    best_G_mean, best_i = 0, 0

    # Start the grid search for the optimal parameter combination.
    for i in range(18):
        # Obtain parameters.
        base_estimator = base_estimator_list[i // 6]
        n_estimators = n_estimators_list[i // 2]
        learning_rate = learning_rate_list[i]

        # Load the AB-based model.
        AB = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate,
                                algorithm='SAMME', random_state=seed)

        # Train the model.
        AB.fit(data_train[:, :-1], data_train[:, -1])

        # Calculate the prediction accuracy.
        pred = AB.predict(data_val[:, :-1])
        true = data_val[:, -1]
        G_mean = geometric_mean_score(true, pred)

        # Save the model with the highest accuracy.
        if G_mean > best_G_mean:
            best_G_mean = G_mean
            best_i = i

    # Load the model with the highest accuracy.
    base_estimator = base_estimator_list[best_i // 6]
    n_estimators = n_estimators_list[best_i // 2]
    learning_rate = learning_rate_list[best_i]
    AB = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate,
                            algorithm='SAMME', random_state=seed)
    AB.fit(data_train[:, :-1], data_train[:, -1])

    # Save the optimal model parameters.
    if opt.save_para:
        joblib.dump(AB, "Saved_Model_params\model_AB_%s.m" % opt.re_samp)

    # Obtain the prediction performance.
    evaluate_model(data_test[:, -1], AB.predict(data_test[:, :-1]), opt.print_inf)

    return AB


def main():
    ''' Train and test the machine-learning occupant injury prediction models. '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--rand_seed', type=int, default=123, help='Random seed')
    parser.add_argument('--re_samp', type=str, default='OS', help='Re-sampling methods: US, OS, CS')
    parser.add_argument('--print_inf', action='store_false', help='print the information of the training process')
    parser.add_argument('--save_para', action='store_false', help='save the model parameters')
    opt = parser.parse_args()

    # Define the random seed.
    seed = opt.rand_seed
    np.random.seed(seed)
    random.seed(seed)

    # Load the real-world crash data.
    data = np.load('dataset/data_pro.npy')
    data_train, data_val, data_test = load_data(data, opt.re_samp, seed)

    # Train the five machine-learning models.
    SVM = train_SVM(data_train, data_val, data_test, opt)
    DT = train_DT(data_train, data_val, data_test, opt)
    KNN = train_KNN(data_train, data_val, data_test, opt)
    NB = train_NB(data_train, data_val, data_test, opt)
    AB = train_AB(data_train, data_val, data_test, opt, [SVM, DT, NB], seed)


if __name__ == "__main__":
    main()

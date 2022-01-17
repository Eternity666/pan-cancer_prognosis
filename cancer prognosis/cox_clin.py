import os, sys
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn._reduction as _Reduction
import random
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn import decomposition, preprocessing
from sklearn.model_selection import KFold, train_test_split
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import prettytable as pt

sys.path.append('..')
from pytorchtools import EarlyStopping
from module import ResNet
from matplotlib import pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Hyper Parameters
# DEVICE = 'cpu'
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
MAX_EPOCHS = 100
FOLDS = 9
NUM_FEATURES = 3000
# Feature selection algorithms (10 folds)
FSA = []


def data_generator(cancers, train_cancers):
    print("\nStart to load and process raw data...")
    train_valid_data = pd.DataFrame()
    test_data = pd.DataFrame()
    independent_data = pd.DataFrame()

    cancer_index = dict.fromkeys(train_cancers)
    start_index = 0
    count_train = 0
    count_independent = 0

    for cancer_type in cancers:
        single_clinical_data_path = f"/mnt/yzh/pancancer/cancer_dataset/{cancer_type}/data_clinical_patient.txt"
        single_clinical_data = pd.read_table(single_clinical_data_path, index_col=0, low_memory=False)

        single_clinical_data.drop(single_clinical_data.head(4).index, inplace=True)

        # 临床数据：Cancer Type, Gender, Race, Diagnosis Age, Neoplasm Disease Stage (AJCC), Prior Diagnosis,
        # Person Neoplasm Cancer Status, Cancer Tumor Stage (AJCC)
        single_clinical_data = single_clinical_data.loc[:, ['Overall Survival (Months)',
                                                            'Overall Survival Status',
                                                            'Diagnosis Age',
                                                            'TCGA PanCanAtlas Cancer Type Acronym',
                                                            'Sex',
                                                            'Race Category',
                                                            'Prior Diagnosis',
                                                            'Person Neoplasm Cancer Status',
                                                            'Neoplasm Disease Stage American Joint Committee on Cancer Code',
                                                            'American Joint Committee on Cancer Tumor Stage Code',
                                                            'Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code',
                                                            'New Neoplasm Event Post Initial Therapy Indicator',
                                                            'Primary Lymph Node Presentation Assessment'
                                                            ]
                               ]

        # 取 Overall Survival 为存活时间
        single_overall_survival = single_clinical_data.loc[:, ['Overall Survival (Months)', 'Overall Survival Status']]
        single_overall_survival.dropna(axis=0, how='any', inplace=True)

        single_clinical_data.drop(['Overall Survival (Months)', 'Overall Survival Status'], axis=1, inplace=True)
        single_clinical_data = single_clinical_data.T

        single_survival_time = single_overall_survival.loc[:, ['Overall Survival (Months)']]
        single_survival_time = single_survival_time.T

        # vital status
        single_vital_status = single_overall_survival.loc[:, ['Overall Survival Status']]
        single_vital_status['Overall Survival Status'] = \
            single_vital_status['Overall Survival Status'].map({'0:LIVING': 0, '1:DECEASED': 1})
        single_vital_status = single_vital_status.T

        single_data = pd.concat(
            [single_survival_time, single_vital_status, single_clinical_data], join='inner')

        if cancer_type in train_cancers:
            single_data = single_data.T
            single_train, single_test = train_test_split(single_data, test_size=0.1, random_state=2021)
            single_train = single_train.T
            single_test = single_test.T
            single_data = single_data.T
            if count_train == 0:
                train_valid_data = pd.concat([train_valid_data, single_train], axis=1, join='outer')
                test_data = pd.concat([test_data, single_test], axis=1, join='outer')
            else:
                train_valid_data = pd.concat([train_valid_data, single_train], axis=1, join='inner')
                test_data = pd.concat([test_data, single_test], axis=1, join='inner')

            count_train += 1
            cancer_index[cancer_type] = [start_index, start_index + single_train.shape[1]]
            start_index += single_train.shape[1]

        else:
            if count_independent == 0:
                independent_data = pd.concat([independent_data, single_data], axis=1, join='outer')
            else:
                independent_data = pd.concat([independent_data, single_data], axis=1, join='inner')
            count_independent += 1

        print(f"Cancer type: {cancer_type} \t| Number of patients: {single_data.shape[1]} | Done!")

    # print("All done!")
    # print(f"Total number of patients: {data.shape[1]}\n")
    # print(f"Total number of clinical features: {data.shape[0] - 2}")

    all_data = pd.concat([train_valid_data, test_data, independent_data], axis=1, join='inner')
    all_data = all_data.T
    all_data['Diagnosis Age'] = all_data['Diagnosis Age'].astype(np.float)
    mean_val = all_data['Diagnosis Age'].mean(skipna=True)
    all_data['Diagnosis Age'].fillna(mean_val, inplace=True)
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    all_data[['Diagnosis Age']] = all_data[['Diagnosis Age']].apply(max_min_scaler)

    # one-hot
    all_data = pd.get_dummies(all_data,
                              columns=['TCGA PanCanAtlas Cancer Type Acronym',
                                       'Sex',
                                       'Race Category',
                                       'Prior Diagnosis',
                                       'Person Neoplasm Cancer Status',
                                       'Neoplasm Disease Stage American Joint Committee on Cancer Code',
                                       'American Joint Committee on Cancer Tumor Stage Code',
                                       'Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code',
                                       'New Neoplasm Event Post Initial Therapy Indicator',
                                       'Primary Lymph Node Presentation Assessment'],
                              dummy_na=False)

    all_data = all_data.T  # 一列是一个病人
    train_valid_data = all_data.iloc[:, 0: train_valid_data.shape[1]]
    start_index = train_valid_data.shape[1]
    test_data = all_data.iloc[:, start_index: start_index + test_data.shape[1]]
    start_index += test_data.shape[1]
    independent_data = all_data.iloc[:, start_index: start_index + independent_data.shape[1]]

    print("All done!")
    print(f"Total patients: {all_data.shape[1]}")
    print(f"Number of patients in train and validation data: {train_valid_data.shape[1]}")
    print(f"Number of patients in test data: {test_data.shape[1]}")
    print(f"Number of patients in independent data: {independent_data.shape[1]}")
    print(f"Total number of clinical features after one-hot: {all_data.shape[0] - 2}")

    # 取数据
    train_valid_data = train_valid_data.to_numpy(dtype=np.float).T
    test_data = test_data.to_numpy(dtype=np.float).T
    independent_data = independent_data.to_numpy(dtype=np.float).T
    print(train_valid_data.shape, test_data.shape, independent_data.shape)

    return train_valid_data, test_data, independent_data, cancer_index


def run_training(x_train, y_train, status_train,
                 x_valid, y_valid, status_valid,
                 x_test, y_test, status_test,
                 x_independent, y_independent, status_independent):

    data_train = np.concatenate((y_train[:, None], status_train[:, None], x_train), axis=1)
    data_train = pd.DataFrame(data_train)
    data_train = data_train.rename(columns={0: 'Survival Time', 1: 'Survival Status'})

    data_test = np.concatenate((y_test[:, None], status_test[:, None], x_test), axis=1)
    data_test = pd.DataFrame(data_test)
    data_test = data_test.rename(columns={0: 'Survival Time', 1: 'Survival Status'})

    data_independent = np.concatenate((y_independent[:, None], status_independent[:, None], x_independent), axis=1)
    data_independent = pd.DataFrame(data_independent)
    data_independent = data_independent.rename(columns={0: 'Survival Time', 1: 'Survival Status'})

    print("Start to build CPH...")
    cph = CoxPHFitter()
    cph.fit(data_train, duration_col='Survival Time', event_col='Survival Status')
    # cph.print_summary()
    hazard_pred = cph.predict_partial_hazard(data_train.drop(columns=['Survival Time', 'Survival Status']))
    train_c_index = concordance_index(data_train['Survival Time'], -hazard_pred, data_train['Survival Status'])
    print("C-index of train dataset: ", train_c_index)

    hazard_pred = cph.predict_partial_hazard(data_test.drop(columns=['Survival Time', 'Survival Status']))
    test_c_index = concordance_index(data_test['Survival Time'], -hazard_pred, data_test['Survival Status'])
    print("C-index of test dataset: ", test_c_index)

    hazard_pred = cph.predict_partial_hazard(data_independent.drop(columns=['Survival Time', 'Survival Status']))
    independent_c_index = concordance_index(data_independent['Survival Time'], -hazard_pred,
                                            data_independent['Survival Status'])
    print("C-index of independent validation dataset: ", independent_c_index)

    return test_c_index, independent_c_index


def train_val_test(train_valid_data, test_data, independent_data, cancer_index, train_cancers, fold):

    print('\n............................................................................')
    print('Train Validation Test, beginning')

    clin_test = test_data[:, 2:]
    y_test = test_data[:, 0]
    status_test = test_data[:, 1].astype(np.bool)

    clin_independent = independent_data[:, 2:]
    y_independent = independent_data[:, 0]
    status_independent = independent_data[:, 1].astype(np.bool)

    cancer_valid_index = dict.fromkeys(train_cancers)
    for cancer_type in train_cancers:
        single_data = train_valid_data[cancer_index[cancer_type][0]: cancer_index[cancer_type][1]]

        valid_indices = []
        for train_index, valid_index in fold.split(single_data):
            valid_indices.append(valid_index)

        cancer_valid_index[cancer_type] = valid_indices

    test_c_indexes = []
    independent_c_indexes = []

    for cv_round in range(FOLDS):
        print('\n............................................................................')
        print('Global cross validation, fold %d, beginning' % (cv_round + 1))
        train_data = np.empty([0, train_valid_data.shape[1]])
        valid_data = np.empty([0, train_valid_data.shape[1]])
        for cancer_type in train_cancers:
            single_data = train_valid_data[cancer_index[cancer_type][0]: cancer_index[cancer_type][1]]
            valid_index = cancer_valid_index[cancer_type][cv_round]
            total_index = list(range(cancer_index[cancer_type][1] - cancer_index[cancer_type][0]))
            train_index = list(set(total_index).difference(set(valid_index)))
            single_train = single_data[train_index]
            single_valid = single_data[valid_index]
            # print(train_data.shape, single_train.shape)
            train_data = np.concatenate((train_data, single_train), axis=0)
            valid_data = np.concatenate((valid_data, single_valid), axis=0)

        # Data information
        print("\nData Overview")
        print("Training Cancers:", train_cancers)
        print(f"Number of patients in training data:\t{train_data.shape[0]}")
        print(f"Number of patients in validation data:\t{valid_data.shape[0]}")

        clin_train = train_data[:, 2:]
        y_train = train_data[:, 0]
        status_train = train_data[:, 1].astype(np.bool)

        clin_valid = valid_data[:, 2:]
        y_valid = valid_data[:, 0]
        status_valid = valid_data[:, 1].astype(np.bool)

        print("Start to build Deep Neural Network...")
        test_c_index, independent_c_index = \
            run_training(clin_train, y_train, status_train,
                         clin_valid, y_valid, status_valid,
                         clin_test, y_test, status_test,
                         clin_independent, y_independent, status_independent
                         )

        test_c_indexes.append(test_c_index)
        independent_c_indexes.append(independent_c_index)

    print('---------------------Final Result-----------------------\n')
    tb = pt.PrettyTable()
    tb.field_names = ['Datasets', 'Mean C-index', 'Max C-index']
    tb.add_row(['Test', round(np.mean(test_c_indexes), 5), round(np.max(test_c_indexes), 5)])
    tb.add_row(['Independent', round(np.mean(independent_c_indexes), 5), round(np.max(independent_c_indexes), 5)])
    print(tb)


if __name__ == '__main__':
    # cancers_list = ['SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
    # cancers_list = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COADREAD', 'DLBC', 'ESCA',
    #                'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC',
    #                'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'SARC']
    cancers_list = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COADREAD', 'DLBC', 'ESCA',
                    'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC',
                    'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'SARC',
                    'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
    # train_cancers_list = ['GBM', 'LAML', 'MESO', 'PAAD', 'ESCA',
    #                      'BLCA', 'KIRP', 'PCPG',
    #                      'KICH', 'ACC', 'COADREAD',
    #                      'LGG', 'KIRC', 'CESC', 'BRCA', 'DLBC']

    train_cancers_list = ['GBM', 'LAML', 'MESO', 'PAAD', 'ESCA',
                          'UCS', 'STAD', 'BLCA', 'KIRP', 'PCPG',
                          'TGCT', 'KICH', 'SKCM', 'ACC', 'COADREAD',
                          'LGG', 'KIRC', 'CESC', 'BRCA', 'DLBC']

    train_valid_data, test_data, independent_data, cancer_dict = data_generator(cancers_list, train_cancers_list)
    print(cancer_dict)

    # x_all = data[:, 2:]
    # y_all = data[:, 0]
    # status_all = np.logical_not(data[:, 1].astype(np.int))

    # # Data Preprocessing

    # z-score
    # x_all = preprocessing.scale(x_all)

    # Min-Max Scale
    # x_all = preprocessing.minmax_scale(x_all)

    # data = np.concatenate((y_all[:, None], status_all[:, None], x_all), axis=1)

    folds = KFold(n_splits=FOLDS, shuffle=True)
    #
    train_val_test(train_valid_data, test_data, independent_data,
                   cancer_dict, train_cancers_list, folds)



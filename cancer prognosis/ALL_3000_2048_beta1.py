import os
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
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from itertools import accumulate

from matplotlib import pyplot as plt


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Hyper Parameters
# DEVICE = 'cpu'
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
MAX_EPOCHS = 1000
NUM_FEATURES = 3000


def data_generator(cancers):
    print("\nStart to load and process raw data...")
    data = pd.DataFrame()

    cancer_index = dict.fromkeys(cancers)
    start_index = 0

    for cancer_type in cancers:
        single_clinical_data_path = f"/mnt/yzh/pancancer/cancer_dataset/{cancer_type}/data_clinical_patient.txt"
        single_gene_expression_path = f"/mnt/yzh/pancancer/cancer_dataset/{cancer_type}/data_RNA_Seq_v2_expression_median.txt"
        single_clinical_data = pd.read_table(single_clinical_data_path, index_col=0, low_memory=False)
        single_gene_expression = pd.read_table(single_gene_expression_path, index_col=1, low_memory=False)

        single_clinical_data.drop(single_clinical_data.head(4).index, inplace=True)
        single_gene_expression.drop(columns=['Hugo_Symbol'], inplace=True)
        single_gene_expression.columns = single_gene_expression.columns.map(lambda x: x[:-3])  # 将病人编号后面的"-01"去掉

        # 取 Overall Survival 为存活时间
        single_overall_survival = single_clinical_data.loc[:, ['Overall Survival (Months)', 'Overall Survival Status']]
        single_overall_survival.dropna(axis=0, how='any', inplace=True)

        single_survival_time = single_overall_survival.loc[:, ['Overall Survival (Months)']]
        single_survival_time = single_survival_time.T

        # vital status
        single_vital_status = single_overall_survival.loc[:, ['Overall Survival Status']]
        single_vital_status['Overall Survival Status'] = \
            single_vital_status['Overall Survival Status'].map({'0:LIVING': 0, '1:DECEASED': 1})
        single_vital_status = single_vital_status.T

        single_data = pd.concat([single_survival_time, single_vital_status, single_gene_expression], join='inner')

        if cancer_type == cancers[0]:
            data = pd.concat([data, single_data], axis=1, join='outer')
        else:
            data = pd.concat([data, single_data], axis=1, join='inner')

        # the features with 80% or more missing values were discarded
        cutoff = 0.8
        num_to_retain = int((1 - cutoff) * data.shape[1])  # 保留至少有 num_to_retain 个非空的行
        data = data.dropna(thresh=num_to_retain)

        # Data imputation step: the mean value was used if the feature values in some patients were missing.
        # data.mean(axis=1, skipna=True)    # 计算的全是NaN
        for index in list(data.index[data.isnull().sum(axis=1) > 0]):
            mean_val = data.loc[index, :].mean(skipna=True)
            data.loc[index, :].fillna(mean_val, inplace=True)

        cancer_index[cancer_type] = [start_index, start_index + single_data.shape[1]]
        start_index = single_data.shape[1]

        print(f"Cancer type: {cancer_type} \t| Number of patients: {single_data.shape[1]} | Done!")

    print("All done!")
    print(f"Total number of patients: {data.shape[1]}\n")
    print(f"Total number of genes retained: {data.shape[0] - 2}")

    # 取数据
    data = data.to_numpy(dtype=np.float).T

    gene_expression_data = data[:, 2:]
    survival_time = data[:, 0]
    vital_status = np.logical_not(data[:, 1].astype(np.int))

    return gene_expression_data, survival_time, vital_status, cancer_index, gene_expression_data.shape[1]


class CancerDataset:
    def __init__(self, gene, survival, status):
        self.gene = gene
        self.survival = survival
        self.status = status

    def __len__(self):
        return self.gene.shape[0]

    def __getitem__(self, idx):
        dct = {
            'gene': torch.tensor(self.gene[idx, :], dtype=torch.float),
            'survival': torch.tensor(self.survival[idx], dtype=torch.float),
            'status': torch.tensor(self.status[idx], dtype=torch.int)
        }
        return dct


def train_func(model, optimizer, loss_func, data_loader, device):
    model.train()   # 如果有batch normalization
    total_loss = 0
    train_pred = []
    train_target = []
    train_status = []

    for data in data_loader:
        optimizer.zero_grad()
        input_gene = data['gene'].to(device)
        survival_target = data['survival'].to(device)
        survival_status = data['status'].to(device)
        if torch.sum(torch.logical_not(survival_status)).item() < 2:
            continue
        survival_output = model(input_gene)
        # survival_target.view(-1, 1)
        # print(survival_target.size())
        # survival_output.squeeze(-1)
        # print(survival_output.size())

        # L1正则化
        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(torch.abs(param))

        tmp_loss = loss_func(survival_output, survival_target) #+ 1e-4 * regularization_loss
        # print(f"train_tmp_loss: {tmp_loss}")
        tmp_loss.backward()
        optimizer.step()
        # scheduler.step()

        total_loss += tmp_loss.item()

        train_pred.append(survival_output.detach().cpu().numpy())
        train_target.append(survival_target.detach().cpu().numpy())
        train_status.append(survival_status.detach().cpu().numpy())

    total_loss /= len(data_loader)
    train_pred = np.concatenate(train_pred)
    train_target = np.concatenate(train_target)
    train_status = np.concatenate(train_status)
    c_index = concordance_index(train_target, train_pred, np.logical_not(train_status))

    return total_loss, c_index


def valid_func(model, loss_func, data_loader, device):
    model.eval()
    total_loss = 0
    valid_pred = []
    valid_target = []
    valid_status = []

    for data in data_loader:
        input_gene = data['gene'].to(device)
        survival_target = data['survival'].to(device)
        survival_status = data['status'].to(device)
        if torch.sum(torch.logical_not(survival_status)).item() < 2:
            continue
        survival_output = model(input_gene)

        # L1正则化
        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(torch.abs(param))

        tmp_loss = loss_func(survival_output, survival_target) #+ 1e-4 * regularization_loss
        # print(f"valid_tmp_loss: {tmp_loss}")
        total_loss += tmp_loss.item()

        valid_pred.append(survival_output.detach().cpu().numpy())
        valid_target.append(survival_target.detach().cpu().numpy())
        valid_status.append(survival_status.detach().cpu().numpy())

    total_loss /= len(data_loader)
    valid_pred = np.concatenate(valid_pred)
    valid_target = np.concatenate(valid_target)
    valid_status = np.concatenate(valid_status)
    c_index = concordance_index(valid_target, valid_pred, np.logical_not(valid_status))

    return total_loss, c_index


class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.num_feature = NUM_FEATURES
        # self.hidden_units = [400, 1000, 500, 500, 100]
        self.hidden1 = nn.Linear(self.num_feature, 2048)
        self.batch_norm_1 = nn.BatchNorm1d(2048)
        self.hidden2 = nn.Linear(2048, 1024)
        self.batch_norm_2 = nn.BatchNorm1d(1024)
        self.hidden3 = nn.Linear(1024, 512)
        self.batch_norm_3 = nn.BatchNorm1d(512)
        self.hidden4 = nn.Linear(500, 100)
        self.batch_norm_4 = nn.BatchNorm1d(100)
        self.output = nn.Linear(512, 1)

    def forward(self, x):
        # 最简单版本的DNN，下次尝试 batch norm —— 若使用了，则在训练前要model.train()，测试时要model.eval()
        # 再下次初始化 weight 再 matmul + bias
        x = F.dropout(x, 0.7, training=self.training)
        x = self.hidden1(x)
        x = self.batch_norm_1(x)
        x = torch.relu(x)
        # x = F.dropout(x, 0.5, training=self.training)
        x = self.hidden2(x)
        x = self.batch_norm_2(x)
        x = torch.relu(x)
        # x = F.dropout(x, 0.5, training=self.training)
        x = self.hidden3(x)
        x = self.batch_norm_3(x)
        x = torch.relu(x)
        x = F.dropout(x, 0.5, training=self.training)
        # x = self.hidden4(x)
        # x = self.batch_norm_4(x)
        # x = torch.tanh(x)
        x = self.output(x)
        # 解决 UserWarning: Using a target size (torch.Size([64])) that is different to the input size (torch.Size([64,
        # 1]))
        x = x.squeeze(-1)
        return x


def run_training(x_train, y_train, x_valid, y_valid, status_train, status_valid, cv_round):
    train_dataset = CancerDataset(x_train, y_train, status_train)
    valid_dataset = CancerDataset(x_valid, y_valid, status_valid)
    train_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # drop_last
    valid_loader = Data.DataLoader(valid_dataset, batch_size=BATCH_SIZE)     # 不知道要不要设置 batch_size 和 shuffle
    network = Network()

    print('\n---Network Structure---')
    print(network)
    print()

    network.to(DEVICE)
    # optimizer = torch.optim.SGD(network.parameters(), lr=0.4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96, last_epoch=-1)
    # last_epoch 最后一个更新lr的epoch

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    # weight_decay 实现L2正则化

    # loss_func = torch.nn.MSELoss()
    # loss_func = torch.nn.L1Loss()
    loss_func = torch.nn.SmoothL1Loss(beta=1)
    # loss_func = LogCoshLoss()
    # loss_func = COXLoss()

    train_losses = []
    train_c_indexes = []
    valid_losses = []
    valid_c_indexes = []

    pre_loss = 0
    final_loss = 0
    final_c_index = 0
    final_epoch = 0

    # training
    for epoch in range(MAX_EPOCHS):
        train_loss, train_c_index = train_func(network, optimizer, loss_func, train_loader, DEVICE)
        if epoch in range(10) or (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch+1}, train loss: {train_loss}, train C-index: {train_c_index}")
        valid_loss, valid_c_index = valid_func(network, loss_func, valid_loader, DEVICE)
        if epoch in range(10) or (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch+1}, valid loss: {valid_loss}, valid C-index: {valid_c_index}")
        if epoch == 0:
            pre_loss = valid_loss

        # 怎么取final还有待商榷
        if valid_loss <= final_loss - 1:
            final_c_index = valid_c_index
            final_loss = valid_loss
            final_epoch = epoch
        elif abs(valid_loss - pre_loss) < 0.5 and final_c_index < valid_c_index:
            final_c_index = valid_c_index
            final_loss = valid_loss
            final_epoch = epoch

        pre_loss = valid_loss

        train_losses.append(train_loss)
        train_c_indexes.append(train_c_index)
        valid_losses.append(valid_loss)
        valid_c_indexes.append(valid_c_index)

    print(f"Fold: {cv_round + 1} | Epoch: {final_epoch} | C-index: {final_c_index}")

    train_losses = np.array(train_losses)
    train_c_indexes = np.array(train_c_indexes)
    valid_losses = np.array(valid_losses)
    valid_c_indexes = np.array(valid_c_indexes)
    epochs = np.linspace(1, MAX_EPOCHS, MAX_EPOCHS).astype(np.int)
    plt.subplot(1, 2, 1)
    plt.title("Smooth L1 Loss")
    plt.plot(epochs, train_losses, color='royalblue', label='train loss')
    plt.plot(epochs, valid_losses, color='springgreen', label='validation loss')
    plt.xlabel("Training Epochs")
    # plt.ylabel("Loss")
    plt.legend(loc=1)

    plt.subplot(1, 2, 2)
    plt.title("C-index")
    plt.plot(epochs, train_c_indexes, color='royalblue', label='train C-index')
    plt.plot(epochs, valid_c_indexes, color='springgreen', label='validation C-index')
    plt.xlabel("Training Epochs")
    # plt.ylabel("C-index")
    plt.legend(loc=2)

    # plt.show()

    plt.savefig(f"/mnt/yzh/pancancer/plot_greedy/KIRC_{NUM_FEATURES}_L1_CV{cv_round + 1}.jpg", format='jpg')
    plt.close()

    return final_c_index


def greedy_cross_validation(x, y, status, cancer_index, fold, cancers, train_cancers, num_gene):
    c_indexes = []

    cancer_valid_index = dict.fromkeys(cancers)
    for cancer_type in cancers:
        single_x = x[cancer_index[cancer_type][0]: cancer_index[cancer_type][1]]
        single_y = y[cancer_index[cancer_type][0]: cancer_index[cancer_type][1]]
        single_status = status[cancer_index[cancer_type][0]: cancer_index[cancer_type][1]]

        valid_indices = []
        for train_index, valid_index in fold.split(single_x, single_y):
            valid_indices.append(valid_index)

        cancer_valid_index[cancer_type] = valid_indices

    x_train = np.empty([0, num_gene])
    y_train = np.empty([0])
    status_train = np.empty([0])

    x_valid = np.empty([0, num_gene])
    y_valid = np.empty([0])
    status_valid = np.empty([0])

    for cv_round in range(10):
        print('\n............................................................................')
        print('Global cross validation, fold %d, beginning' % (cv_round + 1))
        
        x_train = np.empty([0, num_gene])
        y_train = np.empty([0])
        status_train = np.empty([0])

        x_valid = np.empty([0, num_gene])
        y_valid = np.empty([0])
        status_valid = np.empty([0])

        for cancer_type in cancers:
            single_x = x[cancer_index[cancer_type][0]: cancer_index[cancer_type][1]]
            single_y = y[cancer_index[cancer_type][0]: cancer_index[cancer_type][1]]
            single_status = status[cancer_index[cancer_type][0]: cancer_index[cancer_type][1]]
            if cancer_type in train_cancers:
                valid_index = cancer_valid_index[cancer_type][cv_round]
                total_index = list(range(cancer_index[cancer_type][1] - cancer_index[cancer_type][0]))
                train_index = list(set(total_index).difference(set(valid_index)))

                single_x_train = single_x[train_index]
                single_y_train = single_y[train_index]
                single_status_train = single_status[train_index]
                x_train = np.concatenate((x_train, single_x_train), axis=0)
                y_train = np.concatenate((y_train, single_y_train))
                status_train = np.concatenate((status_train, single_status_train))

                single_x_valid = single_x[valid_index]
                single_y_valid = single_y[valid_index]
                single_status_valid = single_status[valid_index]
                x_valid = np.concatenate((x_valid, single_x_valid), axis=0)
                y_valid = np.concatenate((y_valid, single_y_valid))
                status_valid = np.concatenate((status_valid, single_status_valid))

            else:
                valid_index = cancer_valid_index[cancer_type][cv_round]
                single_x_valid = single_x[valid_index]
                single_y_valid = single_y[valid_index]
                single_status_valid = single_status[valid_index]
                x_valid = np.concatenate((x_valid, single_x_valid), axis=0)
                y_valid = np.concatenate((y_valid, single_y_valid))
                status_valid = np.concatenate((status_valid, single_status_valid))

        # Data information
        print("\nData Overview")
        print("Training Cancers:", train_cancers)
        print(f"Number of patients in training data:\t{y_train.shape[0]}")
        print(f"Number of patients in validation data:\t{y_valid.shape[0]}")

        # Feature Selection Algorithms
        print("\nStart to select features...")

        # f_regression
        selector = SelectKBest(score_func=f_regression, k=NUM_FEATURES).fit(x_train, y_train)
        # p_values = selector.pvalues_
        print("Feature Selection Algorithms: f_regression")
        x_train = selector.transform(x_train)
        x_valid = selector.transform(x_valid)
        print("Num of features selected:", NUM_FEATURES)
        print("Done!\n")

        print("Start to build Deep Neural Network...")
        c_index = run_training(x_train, y_train, x_valid, y_valid, status_train, status_valid, cv_round)
        # c_index, r2 = different_model(model, method, x_train, y_train, x_test, y_test)
        c_indexes.append(c_index)

    average_c_index = np.mean(c_indexes)
    # average_r2 = np.mean(r2_scores)
    print('--------------------------------------------------------')
    print("Average C-index:\t", average_c_index)


if __name__ == '__main__':
    # cancers_list = ['BLCA', 'BRCA']
    cancers_list = ['BLCA', 'BRCA', 'CESC', 'COADREAD',
               'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG',
               'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD',
               'PRAD', 'SKCM', 'STAD', 'THCA', 'UCEC']
    train_cancers_list = ['BLCA', 'BRCA', 'CESC', 'COADREAD',
                        'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG',
                        'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD',
                        'PRAD', 'SKCM', 'STAD', 'THCA', 'UCEC']

    x_all, y_all, status_all, cancer_dict, num_genes = data_generator(cancers_list)

    # # Data Preprocessing

    # z-score
    # x_all = preprocessing.scale(x_all)

    # Min-Max Scale
    x_all = preprocessing.minmax_scale(x_all)

    folds = KFold(n_splits=10, shuffle=True)

    greedy_cross_validation(x_all, y_all, status_all, cancer_dict, folds, cancers_list, train_cancers_list, num_genes)


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
from itertools import accumulate
import prettytable as pt

sys.path.append('..')
from pytorchtools import EarlyStopping
from module import ResNet, COXLoss
from matplotlib import pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Hyper Parameters
# DEVICE = 'cpu'
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
MAX_EPOCHS = 100
FOLDS = 9
NUM_GENE_FEATURES = 3000


def sort_data(x, y, z, t, e):
    sort_idx = np.argsort(t)[::-1]  # 就应该从大到小排列
    x = x[sort_idx]
    y = y[sort_idx]
    z = z[sort_idx]
    e = e[sort_idx]
    t = t[sort_idx]
    return x, y, z, t, e


def data_generator(cancers, train_cancers):
    print("\nStart to load and process raw data...")
    train_valid_data = pd.DataFrame()
    test_data = pd.DataFrame()
    independent_data = pd.DataFrame()

    gene_expression = pd.DataFrame()

    modalities = ['clin', 'gene', 'micro']

    cancer_index = dict.fromkeys(cancers)
    modality_index = dict.fromkeys(modalities)
    start_index = 0
    count_train = 0
    count_independent = 0

    for cancer_type in cancers:
        single_clinical_data_path = f"F:\\cancer dataset\\{cancer_type}\\data_clinical_patient.txt"
        single_gene_expression_path = f"F:\\cancer dataset\\{cancer_type}\\data_RNA_Seq_v2_expression_median.txt"
        single_microbiome_path = f"F:\\cancer dataset\\{cancer_type}\\data_microbiome.txt"
        single_clinical_data = pd.read_table(single_clinical_data_path, index_col=0, low_memory=False)
        single_gene_expression = pd.read_table(single_gene_expression_path, index_col=1, low_memory=False)
        single_microbiome = pd.read_table(single_microbiome_path, index_col=0, low_memory=False)

        single_clinical_data.drop(single_clinical_data.head(4).index, inplace=True)
        single_gene_expression.drop(columns=['Hugo_Symbol'], inplace=True)
        single_gene_expression.columns = single_gene_expression.columns.map(lambda x: x[:-3])  # 将病人编号后面的"-01"去掉
        single_microbiome.drop(columns=['NAME', 'DESCRIPTION', 'URL'], inplace=True)
        single_microbiome.columns = single_microbiome.columns.map(lambda x: x[:-3])  # 将病人编号后面的"-01"去掉

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

        single_overall_survival = single_clinical_data.loc[:, ['Overall Survival (Months)', 'Overall Survival Status']]
        single_overall_survival.dropna(axis=0, how='any', inplace=True)

        single_clinical_data.drop(['Overall Survival (Months)', 'Overall Survival Status'], axis=1, inplace=True)

        single_survival_time = single_overall_survival.loc[:, ['Overall Survival (Months)']]
        single_vital_status = single_overall_survival.loc[:, ['Overall Survival Status']]
        single_vital_status['Overall Survival Status'] = \
            single_vital_status['Overall Survival Status'].map({'0:LIVING': 0, '1:DECEASED': 1})

        single_overall_survival = pd.concat([single_survival_time, single_vital_status], axis=1, join='inner')
        single_clinical_data = pd.merge(single_overall_survival, single_clinical_data,
                                        left_index=True, right_index=True, how='left')

        single_gene_expression = single_gene_expression.T  # 一列是一个基因
        single_microbiome = single_microbiome.T  # 一列是一个micro
        modality_index['clin'] = [2, single_clinical_data.shape[1]]
        tmp = single_clinical_data.shape[1]
        modality_index['gene'] = [tmp, tmp + single_gene_expression.shape[1]]
        tmp += single_gene_expression.shape[1]
        modality_index['micro'] = [tmp, tmp + single_microbiome.shape[1]]

        single_data = pd.merge(single_clinical_data, single_gene_expression,
                               left_index=True, right_index=True, how='left')
        single_data = pd.merge(single_data, single_microbiome,
                               left_index=True, right_index=True, how='left')
        single_data = single_data.T     # 一行是一个特征

        # 把gene和micro取出来
        single_gene_expression = single_data.iloc[modality_index['gene'][0]: modality_index['gene'][1], :]
        single_microbiome = single_data.iloc[modality_index['micro'][0]: modality_index['micro'][1], :]
        cutoff = 0.8
        num_to_retain = int((1 - cutoff) * single_gene_expression.shape[1])  # 保留至少有 num_to_retain 个非空的行
        single_gene_expression = single_gene_expression.dropna(thresh=num_to_retain)
        single_microbiome = single_microbiome.dropna(thresh=num_to_retain)

        for index in list(single_gene_expression.index[single_gene_expression.isnull().sum(axis=1) > 0]):
            mean_val = single_gene_expression.loc[index, :].mean(skipna=True)
            single_gene_expression.loc[index, :].fillna(mean_val, inplace=True)
        for index in list(single_microbiome.index[single_microbiome.isnull().sum(axis=1) > 0]):
            mean_val = single_microbiome.loc[index, :].mean(skipna=True)
            single_microbiome.loc[index, :].fillna(mean_val, inplace=True)

        single_clinical_data = single_clinical_data.T   # 一行是一个特征
        single_data = pd.concat([single_clinical_data, single_gene_expression, single_microbiome], axis=0)

        # single_gene_expression 一列是一个病人
        # 这一步用于统计基因的数量
        if cancer_type == cancers[0]:
            gene_expression = pd.concat([gene_expression, single_gene_expression], axis=1, join='outer')
        else:
            gene_expression = pd.concat([gene_expression, single_gene_expression], axis=1, join='inner')

        if cancer_type in train_cancers:
            single_data = single_data.T
            single_train, single_test = train_test_split(single_data, test_size=0.1)
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

        print(f"Cancer type: {cancer_type} \t| Number of patients: {single_data.shape[1]} \t| Done!")

    print("All done!")

    # modality_index['gene'] = [modality_index['gene'][0], modality_index['gene'][0] + gene_expression.shape[0]]
    all_data = pd.concat([train_valid_data, test_data, independent_data], axis=1, join='inner')  # 一行是一个特征
    clinical_data = all_data.iloc[0: modality_index['clin'][1], :]
    gene_microbiome = all_data.iloc[modality_index['clin'][1]:, :]
    # gene_expression = all_data.iloc[modality_index['clin'][1]: modality_index['gene'][1], :]
    # micro = all_data.iloc[modality_index['gene'][1]:, :]

    clinical_data = clinical_data.T
    clinical_data['Diagnosis Age'] = clinical_data['Diagnosis Age'].astype(np.float)
    mean_val = clinical_data['Diagnosis Age'].mean(skipna=True)
    clinical_data['Diagnosis Age'].fillna(mean_val, inplace=True)
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    clinical_data[['Diagnosis Age']] = clinical_data[['Diagnosis Age']].apply(max_min_scaler)

    # one-hot
    clinical_data = pd.get_dummies(clinical_data,
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
                                  dummy_na=False
                                    )
    clinical_data = clinical_data.T     # 变成一行是一个特征了

    all_data = pd.concat([clinical_data, gene_microbiome], axis=0)
    modality_index['clin'] = [2, clinical_data.shape[0]]
    modality_index['gene'] = [clinical_data.shape[0], clinical_data.shape[0] + gene_expression.shape[0]]
    modality_index['micro'] = [modality_index['gene'][1], all_data.shape[0]]

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
    print(f"Total number of clinical features: {modality_index['clin'][1] - modality_index['clin'][0]}")
    print(f"Total number of genes retained: {modality_index['gene'][1] - modality_index['gene'][0]}")
    print(f"Total number of Microbiome features: {modality_index['micro'][1] - modality_index['micro'][0]}")

    # 取数据
    train_valid_data = train_valid_data.to_numpy(dtype=np.float).T
    test_data = test_data.to_numpy(dtype=np.float).T
    independent_data = independent_data.to_numpy(dtype=np.float).T

    return train_valid_data, test_data, independent_data, cancer_index, modality_index


class CancerDataset:
    def __init__(self, gene, clin, micro, survival, status):
        self.gene = gene
        self.clin = clin
        self.micro = micro
        self.survival = survival
        self.status = status

    def __len__(self):
        return self.gene.shape[0]

    def __getitem__(self, idx):
        dct = {
            'gene': torch.tensor(self.gene[idx, :], dtype=torch.float),
            'clin': torch.tensor(self.clin[idx, :], dtype=torch.float),
            'micro': torch.tensor(self.micro[idx, :], dtype=torch.float),
            'survival': torch.tensor(self.survival[idx], dtype=torch.float),
            'status': torch.tensor(self.status[idx], dtype=torch.int)
        }
        return dct


def train_func(model, optimizer, loss_func, data_loader, device):
    model.train()  # 如果有batch normalization
    total_loss = 0
    train_pred = []
    train_target = []
    train_status = []

    for data in data_loader:
        optimizer.zero_grad()
        input_gene = data['gene'].to(device)
        input_clin = data['clin'].to(device)
        input_micro = data['micro'].to(device)
        survival_target = data['survival'].to(device)
        survival_status = data['status'].to(device)
        survival_output = model(input_gene, input_clin, input_micro)

        # L1正则化
        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(torch.abs(param))

        tmp_loss = loss_func(survival_output, survival_status)  # + 1e-4 * regularization_loss
        tmp_loss.backward()
        optimizer.step()
        # scheduler.step()

        total_loss += tmp_loss.item()

        train_pred.append(survival_output.detach().cpu().numpy())
        train_target.append(survival_target.detach().cpu().numpy())
        train_status.append(survival_status.detach().cpu().numpy())

    # total_loss /= len(data_loader)
    train_pred = np.concatenate(train_pred)
    train_target = np.concatenate(train_target)
    train_status = np.concatenate(train_status)
    # print(train_pred)
    c_index = concordance_index(train_target, -train_pred, train_status)

    return total_loss, c_index


def valid_func(model, loss_func, data_loader, device):
    model.eval()
    total_loss = 0
    valid_pred = []
    valid_target = []
    valid_status = []

    for data in data_loader:
        input_gene = data['gene'].to(device)
        input_clin = data['clin'].to(device)
        input_micro = data['micro'].to(device)
        survival_target = data['survival'].to(device)
        survival_status = data['status'].to(device)
        survival_output = model(input_gene, input_clin, input_micro)

        # L1正则化
        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(torch.abs(param))

        tmp_loss = loss_func(survival_output, survival_status)  # + 1e-4 * regularization_loss
        # print(f"valid_tmp_loss: {tmp_loss}")
        total_loss += tmp_loss.item()

        valid_pred.append(survival_output.detach().cpu().numpy())
        valid_target.append(survival_target.detach().cpu().numpy())
        valid_status.append(survival_status.detach().cpu().numpy())

    # total_loss /= len(data_loader)
    valid_pred = np.concatenate(valid_pred)
    valid_target = np.concatenate(valid_target)
    valid_status = np.concatenate(valid_status)
    c_index = concordance_index(valid_target, -valid_pred, valid_status)

    return total_loss, c_index


def test_func(model, loss_func, data_loader, device):
    model.eval()
    total_loss = 0
    test_pred = []
    test_target = []
    test_status = []

    for data in data_loader:
        input_gene = data['gene'].to(device)
        input_clin = data['clin'].to(device)
        input_micro = data['micro'].to(device)
        survival_target = data['survival'].to(device)
        survival_status = data['status'].to(device)

        with torch.no_grad():
            survival_output = model(input_gene, input_clin, input_micro)

        # L1正则化
        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(torch.abs(param))

        tmp_loss = loss_func(survival_output, survival_status)  # + 1e-4 * regularization_loss
        # print(f"valid_tmp_loss: {tmp_loss}")
        total_loss += tmp_loss.item()

        test_pred.append(survival_output.detach().cpu().numpy())
        test_target.append(survival_target.detach().cpu().numpy())
        test_status.append(survival_status.detach().cpu().numpy())

    # total_loss /= len(data_loader)
    test_pred = np.concatenate(test_pred)
    test_target = np.concatenate(test_target)
    test_status = np.concatenate(test_status)
    c_index = concordance_index(test_target, -test_pred, test_status)

    return total_loss, c_index


class Network(torch.nn.Module):

    def __init__(self, num_gene, num_clin, num_micro):
        super(Network, self).__init__()

        self.fc_gene = nn.Linear(num_gene, 512)
        self.resNet_gene = ResNet(512, 9)

        self.fc_clin = nn.Linear(num_clin, 512)
        self.resNet_clin = ResNet(512, 9)

        self.fc_micro = nn.Linear(num_micro, 512)
        self.resNet_micro = ResNet(512, 2)

        self.bn1 = nn.BatchNorm1d(512*3 + 1024)
        self.fc1 = nn.Linear(512*3 + 1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.output = nn.Linear(512, 1)

    def forward(self, gene, clin, micro):
        x = self.fc_gene(gene)
        x = self.resNet_gene(x)

        y = self.fc_clin(clin)
        y = self.resNet_clin(y)

        w = self.fc_micro(micro)
        w = self.resNet_micro(w)

        mean = torch.mean(torch.stack([x, y, w]), dim=0)
        hadamard = x * y * w

        fusion = torch.cat((x, y, w, mean, hadamard), dim=1)
        fusion = self.bn1(fusion)
        fusion = self.fc1(fusion)
        fusion = self.bn2(fusion)
        fusion = self.output(fusion)
        # 解决 UserWarning: Using a target size (torch.Size([64])) that is different to the input size (torch.Size([64,
        # 1]))
        fusion = fusion.squeeze(-1)
        return fusion


def run_training(x_train, clin_train, micro_train, y_train, status_train,
                 x_valid, clin_valid, micro_valid, y_valid, status_valid,
                 x_test, clin_test, micro_test, y_test, status_test,
                 x_independent, clin_independent, micro_independent, y_independent, status_independent):

    train_dataset = CancerDataset(x_train, clin_train, micro_train, y_train, status_train)
    valid_dataset = CancerDataset(x_valid, clin_valid, micro_valid, y_valid, status_valid)
    test_dataset = CancerDataset(x_test, clin_test, micro_test, y_test, status_test)
    independent_dataset = CancerDataset(x_independent, clin_independent, micro_independent,
                                        y_independent, status_independent)

    train_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)  # drop_last
    valid_loader = Data.DataLoader(valid_dataset, batch_size=x_valid.shape[0])  # 不知道要不要设置 batch_size 和 shuffle
    test_loader = Data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    independent_loader = Data.DataLoader(independent_dataset, batch_size=BATCH_SIZE)
    network = Network(num_gene=x_train.shape[1], num_clin=clin_train.shape[1], num_micro=micro_train.shape[1])
    early_stopping = EarlyStopping(patience=20, verbose=True,
                                   path='F:\\model\\BEST_MODEL.pth')

    print('\n----------Network Structure----------')
    print(network)
    print()

    network.to(DEVICE)
    # optimizer = torch.optim.SGD(network.parameters(), lr=0.4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96, last_epoch=-1)
    # last_epoch 最后一个更新lr的epoch

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-1)
    # weight_decay 实现L2正则化

    # loss_func = torch.nn.MSELoss()
    # loss_func = torch.nn.L1Loss()
    # loss_func = torch.nn.SmoothL1Loss()
    # loss_func = LogCoshLoss()
    loss_func = COXLoss()

    train_losses = []
    train_c_indexes = []
    valid_losses = []
    valid_c_indexes = []

    valid_c_index = 0

    final_epoch = MAX_EPOCHS

    # training
    for epoch in range(MAX_EPOCHS):
        train_loss, train_c_index = train_func(network, optimizer, loss_func, train_loader, DEVICE)
        valid_loss, valid_c_index = valid_func(network, loss_func, valid_loader, DEVICE)
        # test_loss, test_c_index = test_func(network, loss_func, test_loader, DEVICE)
        if epoch in range(10) or (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch + 1}, train loss: {train_loss}, train C-index: {train_c_index}")
            print(f"Epoch: {epoch + 1}, valid loss: {valid_loss}, valid C-index: {valid_c_index}")
            # print(f"Epoch: {epoch + 1}, test loss: {test_loss}, test C-index: {test_c_index}")
            print("---------------------------------------------------------------------------")

        train_losses.append(train_loss)
        train_c_indexes.append(train_c_index)
        valid_losses.append(valid_loss)
        valid_c_indexes.append(valid_c_index)

        early_stopping(1 - valid_c_index, network)
        if early_stopping.early_stop:
            final_epoch = epoch + 1
            print(f"Early Stopping! EPOCH: {final_epoch}")
            break

    print(f"Epoch: {final_epoch} | C-index: {valid_c_index}")

    # test
    print("\n----------Start testing----------")
    network = Network(num_gene=x_test.shape[1], num_clin=clin_test.shape[1], num_micro=micro_test.shape[1])
    network.load_state_dict(torch.load(f"F:\\model\\BEST_MODEL.pth"))
    network.to(DEVICE)
    _, test_c_index = test_func(network, loss_func, test_loader, DEVICE)
    print(f"Performance of the EPOCH_{final_epoch} model | C-index: {test_c_index}")
    print("--------Testing Finished---------")

    # independent validation
    print("\n------Independent Validation------")
    _, independent_c_index = test_func(network, loss_func, independent_loader, DEVICE)
    print(f"Performance of the EPOCH_{final_epoch} model | C-index: {independent_c_index}")
    print("------------Finished-------------\n")

    epochs = len(train_losses)
    train_losses = np.array(train_losses)
    train_c_indexes = np.array(train_c_indexes)
    valid_losses = np.array(valid_losses)
    valid_c_indexes = np.array(valid_c_indexes)

    epochs = np.linspace(1, epochs, epochs).astype(np.int)
    plt.subplot(1, 2, 1)
    plt.title("Smooth L1 Loss")
    plt.plot(epochs, train_losses, color='royalblue', label='train loss')
    plt.plot(epochs, valid_losses, color='springgreen', label='validation loss')
    # plt.plot(epochs, test_losses, color='deeppink', label='test loss')
    plt.xlabel("Training Epochs")
    # plt.ylabel("Loss")
    plt.legend(loc=1)

    plt.subplot(1, 2, 2)
    plt.title("C-index")
    plt.plot(epochs, train_c_indexes, color='royalblue', label='train C-index')
    plt.plot(epochs, valid_c_indexes, color='springgreen', label='validation C-index')
    # plt.plot(epochs, test_c_indexes, color='deeppink', label='test C-index')
    plt.xlabel("Training Epochs")
    # plt.ylabel("C-index")
    plt.legend(loc=2)

    # plt.show()

    plt.savefig(f"F:\\plot\\gene+clin+micro.jpg", format='jpg')
    plt.close()

    return valid_c_index, test_c_index, independent_c_index


def train_val_test(train_valid_data, test_data, independent_data,
                   cancer_index, modality_index, fold, train_cancers):

    print('\n............................................................................')
    print('Train Validation Test, beginning')

    gene_test = test_data[:, modality_index['gene'][0]: modality_index['gene'][1]]
    clin_test = test_data[:, modality_index['clin'][0]: modality_index['clin'][1]]
    micro_test = test_data[:, modality_index['micro'][0]: modality_index['micro'][1]]
    y_test = test_data[:, 0]
    status_test = test_data[:, 1].astype(np.bool)

    gene_independent = independent_data[:, modality_index['gene'][0]: modality_index['gene'][1]]
    clin_independent = independent_data[:, modality_index['clin'][0]: modality_index['clin'][1]]
    micro_independent = independent_data[:, modality_index['micro'][0]: modality_index['micro'][1]]
    y_independent = independent_data[:, 0]
    status_independent = independent_data[:, 1].astype(np.bool)

    cancer_valid_index = dict.fromkeys(train_cancers)
    for cancer_type in train_cancers:
        single_data = train_valid_data[cancer_index[cancer_type][0]: cancer_index[cancer_type][1]]

        valid_indices = []
        for train_index, valid_index in fold.split(single_data):
            valid_indices.append(valid_index)

        cancer_valid_index[cancer_type] = valid_indices

    valid_c_indexes = []
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

        gene_train = train_data[:, modality_index['gene'][0]: modality_index['gene'][1]]
        clin_train = train_data[:, modality_index['clin'][0]: modality_index['clin'][1]]
        micro_train = train_data[:, modality_index['micro'][0]: modality_index['micro'][1]]
        y_train = train_data[:, 0]
        status_train = train_data[:, 1].astype(np.bool)
        gene_train, clin_train, micro_train, y_train, status_train = \
            sort_data(gene_train, clin_train, micro_train, y_train, status_train)

        gene_valid = valid_data[:, modality_index['gene'][0]: modality_index['gene'][1]]
        clin_valid = valid_data[:, modality_index['clin'][0]: modality_index['clin'][1]]
        micro_valid = valid_data[:, modality_index['micro'][0]: modality_index['micro'][1]]
        y_valid = valid_data[:, 0]
        status_valid = valid_data[:, 1].astype(np.bool)
        gene_valid, clin_valid, micro_valid, y_valid, status_valid = \
            sort_data(gene_valid, clin_valid, micro_valid, y_valid, status_valid)

        # zscore
        zscore = preprocessing.StandardScaler().fit(gene_train)
        gene_feature_train = zscore.transform(gene_train)
        gene_feature_valid = zscore.transform(gene_valid)
        gene_feature_test = zscore.transform(gene_test)
        gene_feature_independent = zscore.transform(gene_independent)

        zscore = preprocessing.StandardScaler().fit(micro_train)
        micro_feature_train = zscore.transform(micro_train)
        micro_feature_valid = zscore.transform(micro_valid)
        micro_feature_test = zscore.transform(micro_test)
        micro_feature_independent = zscore.transform(micro_independent)

        # Feature Selection Algorithms
        print("\nStart to select features...")

        # f_regression
        selector_gene = SelectKBest(score_func=f_regression,
                                    k=NUM_GENE_FEATURES).fit(gene_feature_train[status_train, :], y_train[status_train])
        # p_values = selector.pvalues_
        print("Feature Selection Algorithms of Genes: f_regression")
        gene_feature_train = selector_gene.transform(gene_feature_train)
        gene_feature_valid = selector_gene.transform(gene_feature_valid)
        gene_feature_test = selector_gene.transform(gene_feature_test)
        gene_feature_independent = selector_gene.transform(gene_feature_independent)
        print("Num of features selected:", gene_feature_train.shape[1])
        print("Done!\n")

        print("Start to build Deep Neural Network...")
        valid_c_index, test_c_index, independent_c_index = \
            run_training(gene_feature_train, clin_train, micro_feature_train, y_train, status_train,
                         gene_feature_valid, clin_valid, micro_feature_valid, y_valid, status_valid,
                         gene_feature_test, clin_test, micro_feature_test, y_test, status_test,
                         gene_feature_independent, clin_independent, micro_feature_independent, y_independent, status_independent
                         )

        valid_c_indexes.append(valid_c_index)
        test_c_indexes.append(test_c_index)
        independent_c_indexes.append(independent_c_index)

    print('\n--------------------------------------------------------')
    print('---------------------Final Result-----------------------\n')
    tb = pt.PrettyTable()
    tb.field_names = ['Data sets', 'Mean C-index', 'Max C-index']
    tb.add_row(['Validation', np.mean(valid_c_indexes), np.max(valid_c_indexes)])
    tb.add_row(['Test', np.mean(test_c_indexes), np.max(test_c_indexes)])
    tb.add_row(['Independent', np.mean(independent_c_indexes), np.max(independent_c_indexes)])
    print(tb)


if __name__ == '__main__':
    cancers_list = ['BLCA', 'KICH', 'CESC']
    # cancers_list = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COADREAD', 'DLBC', 'ESCA',
    #                 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC',
    #                 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'SARC',
    #                 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
    train_cancers_list = ['BLCA', 'KICH']
    # train_cancers_list = ['GBM', 'LAML', 'MESO', 'PAAD', 'ESCA',
    #                       'UCS', 'STAD', 'BLCA', 'KIRP', 'PCPG',
    #                       'TGCT', 'KICH', 'SKCM', 'ACC', 'COADREAD',
    #                       'LGG', 'KIRC', 'CESC', 'BRCA', 'DLBC']

    train_valid_data, test_data, independent_data, cancer_dict, modality_dict = \
        data_generator(cancers_list, train_cancers_list)
    print(cancer_dict)
    print(modality_dict)

    folds = KFold(n_splits=FOLDS, shuffle=True)
    #
    train_val_test(train_valid_data, test_data, independent_data,
                   cancer_dict, modality_dict, folds, train_cancers_list)

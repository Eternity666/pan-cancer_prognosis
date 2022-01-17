import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn import decomposition
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import svm, tree, neighbors, linear_model, ensemble
from scipy.stats import pearsonr
from minepy import MINE
from lifelines.utils import concordance_index


# def drop_index(df, index_name, cutoff=0.8):
#     n = df.shape[1]
#     cnt = len(df[[index_name]].count())
#     if 1 - (float(cnt) / n) >= cutoff:
#         df.drop(index_name, axis=0, inplace=True)


def data_generator(cancers):
    print("\nStart to load and generate data...")
    data = pd.DataFrame()

    for cancer_type in cancers:
        single_clinical_data_path = f"F:\\cancer dataset\\{cancer_type}\\data_clinical_patient.txt"
        single_gene_expression_path = f"F:\\cancer dataset\\{cancer_type}\\data_RNA_Seq_v2_expression_median.txt"
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

        print(f"Cancer type: {cancer_type} \t| Number of patients: {single_data.shape[1]} | Done!")

    print("All done!")
    print(f"Total number of patients: {data.shape[1]}\n")

    # 取数据
    data = data.to_numpy(dtype=np.float).T
    gene_expression_data = data[:, 2:]
    survival_time = data[:, 0]
    vital_status = data[:, 1]

    return gene_expression_data, survival_time, vital_status


def different_model(method, x_train, y_train, x_test, y_test, status_test):
    print("Start to adjust the parameters...")  # RandomizedSearchCV / Hyperopt包

    model = linear_model.BayesianRidge()
    # SVM Regression
    if method == 'SVM Regression':
        model_svr = svm.SVR(kernel='rbf', gamma='scale')
        param_svr = {
            'C': [int(x) for x in 10 * np.linspace(1, 11, 11)] + [1, 5, 15, 25]
        }
        model = RandomizedSearchCV(
            estimator=model_svr,
            param_distributions=param_svr,
            scoring=metrics.make_scorer(concordance_index, greater_is_better=True),
            cv=5,
            n_iter=10
        ).fit(x_train, y_train)
        print(model.best_params_)
        model = model.best_estimator_

    # kNN Regression
    # model_kNN = neighbors.KNeighborsRegressor()
    # Method = 'kNN Regression'

    # Decision Tree Regression
    elif method == 'Decision Tree Regression':
        model_DT = tree.DecisionTreeRegressor(
            criterion='mse', splitter='random', min_samples_leaf=20, min_samples_split=20
        )
        param_DT = {
            'max_features': ['log2', 'sqrt'],
            'max_depth': [int(x) for x in np.linspace(start=10, stop=100, num=10)],
            'max_leaf_nodes': [5, 50, 500, 5000]
        }
        model = RandomizedSearchCV(
            estimator=model_DT,
            param_distributions=param_DT,
            scoring=metrics.make_scorer(concordance_index, greater_is_better=True),
            cv=5,
            n_iter=100
        ).fit(x_train, y_train)
        print(model.best_params_)
        model = model.best_estimator_

    # LASSO Regression (L1) -- lasso CV 可获得最佳lambda值，
    elif method == 'Lasso Regression':
        # lambdas = 10 ** np.linspace(- 3, 3, 100)
        # lambdas = np.logspace(-5, 2, 200)
        lambdas = np.logspace(-5, 0, 20)
        model_lassoCV = linear_model.LassoCV(
            alphas=lambdas, cv=5, selection='random', random_state=0, max_iter=100
        ).fit(x_train, y_train)
        lasso_best_alpha = model_lassoCV.alpha_
        print("Best alpha of Lasso: ", lasso_best_alpha)
        model = linear_model.Lasso(alpha=lasso_best_alpha).fit(x_train, y_train)

    elif method == 'Ridge Regression':
        lambdas = np.logspace(-5, 1, 200)
        # lambdas = 10 ** np.linspace(- 3, 3, 100)
        model_RidgeCV = linear_model.RidgeCV(
            alphas=lambdas,
            scoring=metrics.make_scorer(concordance_index, greater_is_better=True),
            cv=5
        ).fit(x_train, y_train)
        ridge_best_alpha = model_RidgeCV.alpha_
        print("Best alpha of Ridge: ", ridge_best_alpha)
        model = linear_model.Ridge(
            alpha=ridge_best_alpha
        ).fit(x_train, y_train)

    # Bayesian Ridge Regression (L2)
    # 也可以试试 ARDRegression
    elif method == 'Bayesian Ridge Regression':
        model_bayes = linear_model.BayesianRidge()
        param_bayes = {
            'alpha_1': list(10 ** np.linspace(-6, -1, 6)) + [0.5],
            'alpha_2': list(10 ** np.linspace(-6, -1, 6)),
            'lambda_1': list(10 ** np.linspace(-6, -1, 6)),
            'lambda_2': list(10 ** np.linspace(-6, -1, 6))
        }
        model = RandomizedSearchCV(
            estimator=model_bayes,
            param_distributions=param_bayes,
            scoring=metrics.make_scorer(concordance_index, greater_is_better=True),
            cv=5,
            n_iter=50
        ).fit(x_train, y_train)
        print(model.best_params_)
        model = model.best_estimator_
        # model = model_bayes.fit(x_train, y_train)
        # print(model.alpha_, model.lambda_)

    # Random Forest Regression
    elif method == 'Random Forest Regression':
        # model_RF = ensemble.RandomForestRegressor(
        #     random_state=0, oob_score=True
        # )
        # param_RF = {
        #     'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        #     'max_features': ['auto', 'sqrt', 'log2', 0.2],
        #     'min_samples_leaf': [1, 5, 10, 50, 100, 200, 500],
        #     'max_depth': list(np.linspace(10, 100, 10)),
        #     'min_samples_split': list(np.linspace(10, 100, 10).astype(np.int))
        # }
        # model = RandomizedSearchCV(
        #     estimator=model_RF,
        #     param_distributions=param_RF,
        #     scoring=metrics.make_scorer(concordance_index, greater_is_better=True),
        #     cv=5,
        #     n_iter=20
        # ).fit(x_train, y_train)
        # print(model.best_params_)
        # model = model.best_estimator_
        model = ensemble.RandomForestRegressor(
            random_state=0,
            oob_score=True,
            n_estimators=500,
            min_samples_split=10,
            min_samples_leaf=100,
            max_features=0.2,
            max_depth=80
        ).fit(x_train, y_train)

    # Adaboost Regression
    model_ada = ensemble.AdaBoostRegressor(n_estimators=50)
    # Method = 'Adaboost Regression'

    # Gradient BoostRegression Tree
    model_GBRT = ensemble.GradientBoostingRegressor(n_estimators=50)
    # Method = 'Gradient BoostRegression Tree'

    # model.fit(x_train, y_train)
    # score = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    # print(y_test.shape, y_pred.shape, status_test.shape)
    c_index = concordance_index(y_test, y_pred, np.logical_not(status_test))
    ev = metrics.explained_variance_score(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    print(f"Method: {method}\tC-index: {round(c_index, 4)}")
    print(f"Explain variance score: {round(ev, 4)}\tR^2: {round(r2, 4)}")
    return c_index, r2

    # plt.figure()
    # plt.plot(np.arange(len(result)), y_test, "go-", label="True value")
    # plt.plot(np.arange(len(result)), result, "ro-", label="Predict value")
    # plt.title(f"Method:{method}---Score:{score}")
    # plt.legend(loc="best")
    # plt.show()


def cross_validation(x_all, y_all, fold, method, status):
    cv_round = 1
    c_indexes = []
    r2_scores = []
    for train_index, test_index in fold:
        print('........................................................')
        print('Global cross validation, round %d, beginning' % cv_round)
        x_train = x_all[train_index]
        y_train = y_all[train_index]
        x_test = x_all[test_index]
        y_test = y_all[test_index]
        status_test = status[test_index]
        c_index, r2 = different_model(method, x_train, y_train, x_test, y_test, status_test)
        c_indexes.append(c_index)
        r2_scores.append(r2)
        cv_round += 1
    average_c_index = np.mean(c_indexes)
    average_r2 = np.mean(r2_scores)
    print('--------------------------------------------------------')
    print("C-index:\t", average_c_index)
    print("R-square:\t", average_r2)


# 互信息计算
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return m.mic(), 0.5


NUM_FEATURES = 400

if __name__ == '__main__':
    # cancer_list = ['BRCA', 'COADREAD', 'LUAD', 'PRAD', 'STAD']
    cancer_list = ['BRCA']

    x_all, y_all, status_all = data_generator(cancer_list)
    index = [i for i in range(len(x_all))]
    random.shuffle(index)
    x_all = x_all[index]
    y_all = y_all[index]
    status_all = status_all[index]

    # # Data Preprocessing
    # # # 再试一下 log2(x+1)
    # z-score
    x_all = preprocessing.scale(x_all)

    # Min-Max Scale
    # x_all = preprocessing.minmax_scale(x_all)

    # Feature Selection Algorithms
    print("Start to select features...")

    # 方差选择法（计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征）
    # x_feature = VarianceThreshold(threshold=1).fit_transform(x_all)
    # print(f"Number of features: {x_feature.shape[1]}")

    # f_regression
    # selector = SelectKBest(score_func=f_regression, k=NUM_FEATURES).fit(x_all, y_all)
    # p_values = selector.pvalues_
    # x_feature = SelectKBest(score_func=f_regression, k=NUM_FEATURES).fit_transform(x_all, y_all)

    # mutual_info_regression
    x_feature = SelectKBest(
        score_func=mutual_info_regression,
        k=NUM_FEATURES
    ).fit_transform(x_all, y_all)

    # Pearson 相关系数
    # print("Feature Selection Algorithm: Pearson's Correlation Coefficient")
    # x_feature = SelectKBest(lambda X, Y: np.array(
    #     list(map(lambda x: pearsonr(x, Y), X.T))).T[0], k=NUM_FEATURES).fit_transform(x_all, y_all)
    # print("Number of features selected:", x_feature.shape[1])
    # print("Done!")

    # 互信息
    # x_feature = SelectKBest(lambda X, Y: np.array(
    #     list(map(lambda x: mic(x, Y), X.T))).T[0], k=400).fit_transform(x_all, y_all)

    # Feature Extraction Algorithms
    # print("Start to extract features...")

    # PCA
    # print("Feature Extraction Algorithms: PCA")
    # pca = decomposition.PCA(n_components=NUM_FEATURES)
    # x_feature = pca.fit_transform(x_all)
    # print(pca.explained_variance_ratio_.sum())
    # print("Done!\n")
    # exit()

    # KPCA  # 核函数为sigmod时效果很差 c-index=0.50
    # kpca = decomposition.KernelPCA(n_components=None, kernel='rbf')  # gamma=? kernels= ['linear','poly','rbf','sigmoid']
    # x_feature = kpca.fit_transform(x_all)

    # IncrementalPCA ??

    # # Regression Models
    print("Start to build regression model...")
    Methods = ['SVM Regression',
               'Decision Tree Regression',
               'Ridge Regression',
               'Bayesian Ridge Regression',
               'Lasso Regression',
               'Random Forest Regression',
               'Adaboost Regression',
               'Gradient BoostRegression Tree'
               ]
    Method = Methods[0]
    print(f"Regression model: {Method}")

    # Cross Validation
    # folds = StratifiedKFold(n_splits=5)
    folds = KFold(n_splits=10)
    # folds = StratifiedKFold(y_all, n_splits=10, shuffle=True, random_state=np.random.RandomState(1))

    cross_validation(x_feature, y_all, folds.split(x_feature, y_all),
                     method=Method, status=status_all)


y_status_train = np.concatenate((y_train[:, None] * 4, status_train[:, None]), axis=1)
y_status_train = pd.DataFrame(y_status_train)
x_train = pd.DataFrame(x_train)

y_status_test = np.concatenate((y_test[:, None] * 4, status_test[:, None]), axis=1)
y_status_test = pd.DataFrame(y_status_test)
x_test = pd.DataFrame(x_test)

y_status_independent = np.concatenate((y_independent[:, None] * 4, status_independent[:, None]), axis=1)
y_status_independent = pd.DataFrame(y_status_independent)
x_independent = pd.DataFrame(x_independent)

print("Start to build Random Survival Forest...")
rsf = RandomSurvivalForest(n_estimators=1000, n_jobs=-1)
rsf = rsf.fit(x_train, y_status_train)
pred_train = rsf.predict(x_train)
train_c_index = concordance_index(y_train, pred_train, status_train)
print("C-index of train dataset: ", train_c_index)

pred_test = rsf.predict(x_test)
test_c_index = concordance_index(y_test, pred_test, status_test)
print("C-index of test dataset: ", test_c_index)

pred_independent = rsf.predict(x_independent)
independent_c_index = concordance_index(y_independent, pred_independent, status_independent)
print("C-index of independent validation dataset: ", independent_c_index)
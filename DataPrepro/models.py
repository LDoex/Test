import time
from sklearn import metrics
import pickle as pickle
import pandas as pd
import numpy as np



# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.1)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
# def svm_cross_validation(train_x, train_y):
#     from sklearn.model_selection import GridSearchCV
#     from sklearn.svm import SVC
#     model = SVC(kernel='rbf', probability=True)
#     param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
#     grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
#     grid_search.fit(train_x, train_y)
#     best_parameters = grid_search.best_estimator_.get_params()
#     for para, val in list(best_parameters.items()):
#         print(para, val)
#     model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
#     model.fit(train_x, train_y)
#     return model

# 处理csv文件,随机取出1000个样本并保存
def op_csv(set0,set1,output_name,size=1000):
    name0 = "D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/Results_"+str(set0)+".csv"
    name1 = "D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/Results_"+str(set1)+".csv"
    data0 = pd.read_csv(name0);data0.marker=0
    data0=data0.sample(n=size).reset_index(drop=True)  # 随机采样
    data1 = pd.read_csv(name1);data1.marker=1
    data1=data1.sample(n=size).reset_index(drop=True)

    data=data0.append(data1,ignore_index=True)
    data.rename(columns={'marker': 'label'}, inplace=True)
    data=data.sample(frac=1).reset_index(drop=True)
    data.to_csv(output_name,index=None)


def op_data(train_path, test_path, train_frac, test_frac=1):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_num=int(len(train_data) * train_frac)
    train=train_data.loc[train_data['label'] == 0].head(train_num)
    train=train.append(train_data.loc[train_data['label'] == 1].head(train_num), ignore_index=True)
    train=train.sample(frac=1).reset_index(drop=True)
    test = test_data[:int(len(test_data) * test_frac)]

    print("训练集大小:", train.shape)
    print("测试集大小:", test.shape)
    train_y = train.label
    train_x = train.drop('label', axis=1)

    test_y = test.label
    test_x = test.drop('label', axis=1)

    return train_x, train_y, test_x, test_y


def ex_exp(result_path,rounds=2):
    train_path = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/Muticlass_0-1-3_fewtrain.csv'
    test_path = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/Muticlass_0-1-3_fewtest.csv'
    X = pd.read_csv(train_path)
    X_ = pd.read_csv(test_path)
    train_x = X.drop(columns='label')
    train_y = X.label.astype('int')
    test_x = X_.drop(columns='label')
    test_y = X_.label.astype('int')

    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']  # 去掉svmcv
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'GBDT': gradient_boosting_classifier
                   }

    print('reading training and testing data...')

    av_result = pd.DataFrame(
        data=None, index=range(rounds), columns=['classifier', 'av_precision', 'av_recall', 'av_accuracy'])
    comp_result = pd.DataFrame(
        data=None, index=range(rounds), columns=['classifier', 'precision', 'recall', 'accuracy'])

    for j in range(len(test_classifiers)):
        classifier = test_classifiers[j]
        for i in range(rounds):
            print('******************* %s ********************' % classifier)
            start_time = time.time()
            model = classifiers[classifier](train_x, train_y)

            print('training took %fs!' % (time.time() - start_time))
            predict = model.predict(test_x)

            precision = metrics.precision_score(test_y, predict)
            recall = metrics.recall_score(test_y, predict)
            accuracy = metrics.accuracy_score(test_y, predict)
            comp_result.loc[j * rounds + i] = [classifier, precision, recall, accuracy]
            print('\nprecision: %.2f%%, recall: %.2f%%, accuracy: %.2f%%' % ( 100 * precision, 100 * recall, 100 * accuracy))

    comp_result.to_csv(result_path)
    return comp_result

if __name__ == '__main__':
    # comp_results_path = "D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/5shot_0-1-3.csv"
    # ex_exp(comp_results_path,rounds=3)
    train_path = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/Muticlass_0-1-3_fewtrain.csv'
    test_path = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/Muticlass_0-1-3_fewtest.csv'
    X = pd.read_csv(train_path)
    X_ = pd.read_csv(test_path)
    train_x = X.drop(columns='label')
    train_y = X.label.astype('int')
    test_x = X_.drop(columns='label')
    test_y = X_.label.astype('int')
    model = knn_classifier(train_x,train_y)
    predict = model.predict(test_x)


    accuracy = metrics.accuracy_score(test_y, predict)
    print(accuracy)

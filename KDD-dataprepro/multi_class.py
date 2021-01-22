import numpy as np
import pandas as pd
import time
from sklearn import metrics
from sklearn import svm

from sklearn.model_selection import train_test_split

attributes=['R1-PA1:VH', 'R1-PM1:V', 'R1-PA2:VH', 'R1-PM2:V', 'R1-PA3:VH', 'R1-PM3:V', 'R1-PA4:IH',
            'R1-PM4:I', 'R1-PA5:IH', 'R1-PM5:I', 'R1-PA6:IH', 'R1-PM6:I', 'R1-PA7:VH', 'R1-PM7:V',
            'R1-PA8:VH', 'R1-PM8:V', 'R1-PA9:VH', 'R1-PM9:V', 'R1-PA10:IH', 'R1-PM10:I', 'R1-PA11:IH',
            'R1-PM11:I', 'R1-PA12:IH', 'R1-PM12:I', 'R1:F', 'R1:DF', 'R1-PA:Z', 'R1-PA:ZH', 'R1:S',
            'R2-PA1:VH', 'R2-PM1:V', 'R2-PA2:VH', 'R2-PM2:V', 'R2-PA3:VH', 'R2-PM3:V', 'R2-PA4:IH',
            'R2-PM4:I', 'R2-PA5:IH', 'R2-PM5:I', 'R2-PA6:IH', 'R2-PM6:I', 'R2-PA7:VH', 'R2-PM7:V',
            'R2-PA8:VH', 'R2-PM8:V', 'R2-PA9:VH', 'R2-PM9:V', 'R2-PA10:IH', 'R2-PM10:I', 'R2-PA11:IH',
            'R2-PM11:I', 'R2-PA12:IH', 'R2-PM12:I', 'R2:F', 'R2:DF', 'R2-PA:Z', 'R2-PA:ZH', 'R2:S',
            'R3-PA1:VH', 'R3-PM1:V', 'R3-PA2:VH', 'R3-PM2:V', 'R3-PA3:VH', 'R3-PM3:V', 'R3-PA4:IH',
            'R3-PM4:I', 'R3-PA5:IH', 'R3-PM5:I', 'R3-PA6:IH', 'R3-PM6:I', 'R3-PA7:VH', 'R3-PM7:V',
            'R3-PA8:VH', 'R3-PM8:V', 'R3-PA9:VH', 'R3-PM9:V', 'R3-PA10:IH', 'R3-PM10:I', 'R3-PA11:IH',
            'R3-PM11:I', 'R3-PA12:IH', 'R3-PM12:I', 'R3:F', 'R3:DF', 'R3-PA:Z', 'R3-PA:ZH', 'R3:S', 'R4-PA1:VH',
            'R4-PM1:V', 'R4-PA2:VH', 'R4-PM2:V', 'R4-PA3:VH', 'R4-PM3:V', 'R4-PA4:IH', 'R4-PM4:I', 'R4-PA5:IH',
            'R4-PM5:I', 'R4-PA6:IH', 'R4-PM6:I', 'R4-PA7:VH', 'R4-PM7:V', 'R4-PA8:VH', 'R4-PM8:V', 'R4-PA9:VH',
            'R4-PM9:V', 'R4-PA10:IH', 'R4-PM10:I', 'R4-PA11:IH', 'R4-PM11:I', 'R4-PA12:IH', 'R4-PM12:I', 'R4:F',
            'R4:DF', 'R4-PA:Z', 'R4-PA:ZH', 'R4:S', 'control_panel_log1', 'control_panel_log2', 'control_panel_log3',
            'control_panel_log4', 'relay1_log', 'relay2_log', 'relay3_log', 'relay4_log', 'snort_log1', 'snort_log2',
            'snort_log3', 'snort_log4', 'label']
wanted_attr=['R1-PA1:VH', 'R1-PM1:V', 'R1-PA2:VH', 'R1-PM2:V', 'R1-PA3:VH', 'R1-PM3:V', 'R1-PA4:IH','R1-PM4:I', 'R1-PA5:IH', 'R1-PM5:I', 'R1-PA6:IH', 'R1-PM6:I','R1-PA:Z',
             'R2-PA1:VH', 'R2-PM1:V', 'R2-PA2:VH', 'R2-PM2:V', 'R2-PA3:VH', 'R2-PM3:V', 'R2-PA4:IH','R2-PM4:I', 'R2-PA5:IH', 'R2-PM5:I', 'R2-PA6:IH', 'R2-PM6:I','R2-PA:Z',
             'R3-PA1:VH', 'R3-PM1:V', 'R3-PA2:VH', 'R3-PM2:V', 'R3-PA3:VH', 'R3-PM3:V', 'R3-PA4:IH','R3-PM4:I', 'R3-PA5:IH', 'R3-PM5:I', 'R3-PA6:IH', 'R3-PM6:I','R3-PA:Z',
             'R4-PA1:VH', 'R4-PM1:V', 'R4-PA2:VH', 'R4-PM2:V', 'R4-PA3:VH', 'R4-PM3:V', 'R4-PA4:IH','R4-PM4:I', 'R4-PA5:IH', 'R4-PM5:I', 'R4-PA6:IH', 'R4-PM6:I','R4-PA:Z','label']
# data0 = pd.read_csv(name0);data0.marker=0
#     data0=data0.sample(n=size).reset_index(drop=True)

def read_data(sets, each_size=3500, col=attributes):
    joint_data = pd.DataFrame(columns=col)
    joint_data_path = "./data/data_for_models/inf2mean_all_norm/joint_data.csv"
    for i in range(len(sets)):
        temp_path = "./data/data_for_models/inf2mean_all_norm/results" + str(sets[i]) + ".csv"
        # 读取数据，采样，重置index并且不把原index作为column
        temp_data = pd.read_csv(temp_path)
        temp_data=temp_data.sample(n=each_size).reset_index(drop=True)
        temp_data.rename(columns={'marker': 'label'}, inplace=True)
        temp_data.label = i
        joint_data=joint_data.append(temp_data, ignore_index=True)

    joint_data.to_csv(joint_data_path,index=None)
    return joint_data


def op_data(joint_data,shot,test_frac=1,cls_num=3):          # 降维处理
    train_data=pd.DataFrame(columns=wanted_attr)
    joint_data_comp=joint_data.sample(frac=1)
    col=wanted_attr                                          # 降维处理
    joint_data_lessDim = joint_data_comp[col].copy()
    for i in range(cls_num):

        temp_data=joint_data_lessDim.loc[joint_data_lessDim['label']==i].head(shot)
        train_data=train_data.append(temp_data,ignore_index=True)
    test_data=joint_data_lessDim.sample(frac=test_frac)
    train_x = train_data.drop('label', axis=1)
    train_y = train_data.label.astype('int')  # 要把label从object类型转为int类型，分类器才能识别，之前没遇到的问题
    test_x = test_data.drop('label', axis=1)
    test_y = test_data.label.astype('int')
    return train_x, train_y, test_x, test_y


def GaussianNB_classifier(train_x, train_y):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(train_x, train_y)
    return model


def one_knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(train_x, train_y)
    return model

def three_knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(train_x, train_y)
    return model

# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2',max_iter=1000)
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model

def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='sigmoid', probability=True)
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


def reach_a_data(temp_index_li):
    path="data/groups/group"+str(temp_index_li[0])+"/"+str(temp_index_li[1])+".csv"

    a_data=pd.read_csv(path).iloc[temp_index_li[2],:]

    return a_data

def get_data_from_index(data_index,case_size=60):
    for i in range(case_size):
        temp_index_li = data_index.iloc[i, :].values
        a_data = reach_a_data(temp_index_li)
        temp_train=pd.DataFrame(columns=attributes)
        temp_test = pd.DataFrame(columns=attributes)
        print(type(data_index.loc[i,'is_support']))
        if data_index.loc[i,'is_support']==1:
            temp_train=temp_train.append(a_data,ignore_index=True)
        else:
            temp_test = temp_test.append(a_data,ignore_index=True)
    return temp_train,temp_test

def ex_exp(sets,shot,result_path,rounds,test_frac=1):

    joint_data=read_data(sets)
    print("training and sets:", sets)

    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']  # 去掉svmcv
    classifiers = {'NB': GaussianNB_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'GBDT': gradient_boosting_classifier
                   }
    print('reading training and testing data...')

    comp_result = pd.DataFrame(
        data=None, index=None, columns=['classifier', 'precision', 'recall', 'accuracy'])
    av_result = pd.DataFrame(
        data=None, index=None, columns=['classifier', 'av_precision', 'recall', 'accuracy'])
    for i in range(len(test_classifiers)):
        classifier=test_classifiers[i]
        for j in range(rounds):
            train_x, train_y, test_x, test_y = op_data(joint_data,shot,test_frac)
            print(train_x.shape);print(test_x.shape)
            print('******************* %s ********************' % classifier)
            start_time = time.time()
            model = classifiers[classifier](train_x, train_y)
            print('training took %fs!' % (time.time() - start_time))
            predict = model.predict(test_x)
            # precision = metrics.precision_score(test_y, predict)
            # recall = metrics.recall_score(test_y, predict)
            precision = np.nan
            recall = np.nan
            accuracy = metrics.accuracy_score(test_y, predict)
            print('precision: %.2f%%, recall: %.2f%%, '
                  'accuracy: %.2f%%' % (100 * precision, 100 * recall, 100 * accuracy))
            comp_result.loc[i* rounds + j] = [classifier, precision, recall, accuracy]

    comp_result.to_csv(result_path)

if __name__ == '__main__':
     shot_n=601
     types = [[1,2,4]]  # 训练组
     for t in range(len(types)):
        comp_results_path = "./data/data_for_models/inf2mean_all_norm/" \
                            "3way_test_results/%dshot_set%d-%d-%d.csv" % (shot_n,types[t][0],types[t][1],types[t][2])
        ex_exp(types[t],shot_n,comp_results_path,rounds=5)



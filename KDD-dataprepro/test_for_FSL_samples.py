import pandas as pd
import numpy as np
import multi_class as mc
from sklearn import metrics
import sys
print(sys.argv[0])

def get_train_xy(path):
    train_data=pd.read_csv(path)
    train_x = train_data.drop(train_data.columns[324:], axis=1)
    train_y = train_data.result.astype('int')
    print(train_data)
    return train_x, train_y

def get_test_xy(path_li,col,each_size=1000):
    test_data=pd.DataFrame(columns=col)
    for p in path_li:
        temp_path=dir_path+p
        temp_data=pd.read_csv(temp_path)
        temp_data=temp_data.sample(n=each_size).reset_index(drop=True)
        test_data=test_data.append(temp_data,ignore_index=True)
    test_x = test_data.drop('label', axis=1)
    test_y = test_data.label.astype('int')
    print(test_y)
    return test_x,test_y


def one_test(train_x, train_y,test_x,test_y,output_path,rounds=1):
    comp_result = pd.DataFrame(
        data=None, index=None, columns=['classifier', 'precision', 'recall', 'accuracy'])
    for i in range(len(test_classifiers)):
        classifier = test_classifiers[i]
        for j in range(rounds):
            print('******************* %s ********************' % classifier)
            start_time = mc.time.time()
            model = classifiers[classifier](train_x, train_y)
            print('training took %fs!' % (mc.time.time() - start_time))
            predict = model.predict(test_x)
            precision = metrics.precision_score(test_y, predict, average='macro')
            recall = metrics.recall_score(test_y, predict, average='macro')
            F1 = metrics.f1_score(test_y, predict, average='macro')

            # precision = np.nan
            # recall = np.nan
            accuracy = mc.metrics.accuracy_score(test_y, predict)
            print('precision: %.2f%%, recall: %.2f%%, '
                  'accuracy: %.2f%%' % (100 * precision, 100 * recall, 100 * accuracy))
            precisions[classifier].append(precision)
            recalls[classifier].append(recall)
            F1s[classifier].append(F1)
            accuracys[classifier].append(accuracy)
            comp_result.loc[i * rounds + j] = [classifier, precision, recall, accuracy]
    # comp_result.to_csv(output_path)

# def comp_result2av(comp_result_path,clsfier_num=7):
#     data=pd.read_csv(comp_result_path)
#     av_result=pd.DataFrame(columns=data.columns)
#     rounds=data.shape[0]/clsfier_num
#     for i in range(clsfier_num):
#         temp_data=data.iloc[]


dir_path="D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/7/1shot-correct/"
#test_path=dir_path+"test_del_support014.csv"
path_li=["test_class_0.csv","test_class_1.csv","test_class_4.csv"]
acc_df = pd.DataFrame(data=None, columns=['classfier','accuracy'])
test_classifiers = [ 'GNB', '1NN', '3NN', 'SVM']  # 去掉NB
classifiers = {'GNB': mc.GaussianNB_classifier,
                   '1NN': mc.one_knn_classifier,
                   '3NN': mc.three_knn_classifier,
                   'LR': mc.logistic_regression_classifier,
                   'RF':mc.random_forest_classifier,
                   'DT': mc.decision_tree_classifier,
                   'SVM': mc.svm_classifier,
                   'GBDT': mc.gradient_boosting_classifier}
precisions = {'GNB': [],
                   '1NN': [],
                   '3NN': [],
                   'SVM': []}
recalls = {'GNB': [],
                   '1NN': [],
                   '3NN': [],
                   'SVM': []}
F1s = {'GNB': [],
                   '1NN': [],
                   '3NN': [],
                   'SVM': []}
accuracys = {'GNB': [],
                   '1NN': [],
                   '3NN': [],
                   'SVM': []}

col=pd.read_csv("D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/4/5shot/support_ep_0.csv").columns

pd.set_option('display.max_row', None)
for i in range(32):
    train_path=dir_path+"support_ep_"+str(i)+".csv"
    test_path = dir_path+"query_ep_"+str(i)+".csv"
    output_path="D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/results/ep"+str(i)+".csv"
    acc_outpath = "D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/results/acc.csv"
    train_x, train_y=get_train_xy(train_path)
    test_x, test_y = get_train_xy(test_path)
    one_test(train_x, train_y,test_x,test_y,output_path=output_path)

for key in accuracys:
    temp = pd.DataFrame(data=None, columns=['classfier', 'precision', 'recall', 'F1', 'accuracy'])
    precisions[key] = np.array(precisions[key])
    recalls[key] = np.array(recalls[key])
    F1s[key] = np.array(F1s[key])
    accuracys[key] = np.array(accuracys[key])

    pre_avg = np.mean(precisions[key])
    rec_avg = np.mean(recalls[key])
    f1s_avg = np.mean(F1s[key])
    acc_avg = np.mean(accuracys[key])

    temp.loc[0, 'classfier'] = key
    temp.loc[0, 'precision'] = pre_avg
    temp.loc[0, 'recall'] = rec_avg
    temp.loc[0, 'F1'] = f1s_avg
    temp.loc[0, 'accuracy'] = acc_avg
    acc_df = acc_df.append(temp)
acc_df.to_csv(acc_outpath,index=None)



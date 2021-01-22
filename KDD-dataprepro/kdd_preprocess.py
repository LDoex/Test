from keras.utils import np_utils
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn import preprocessing

def pca(data, outputPath):
    x = data.drop(columns=['label'])
    y = data.label
    scaled_data = preprocessing.scale(x)
    feature_num = 37
    pca = PCA(n_components=feature_num)
    pca.fit(scaled_data)
    joblib.dump(pca, outputPath)
    pca_data = pca.transform(scaled_data)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    pca_df = pd.DataFrame(pca_data, columns=labels)
    pca_df['label'] = y
    return pca_df

def load_pca(data, pcaPath):
    x = data.drop(columns=['label'])
    y = data.label
    scaled_data = preprocessing.scale(x)
    pca = joblib.load(pcaPath)
    pca_data = pca.transform(scaled_data)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    pca_df = pd.DataFrame(pca_data, columns=labels)
    pca_df['label'] = y
    return pca_df

if __name__ == '__main__':
    kdd_trainfilepath = 'D:/Users/oyyk/PycharmProjects/F_G_P/KDD-dataprepro/NSL-KDD_traindata.csv'
    kdd_testfilepath = 'D:/Users/oyyk/PycharmProjects/F_G_P/KDD-dataprepro/NSL-KDD_testdata.csv'
    train_output = 'D:/Users/oyyk/PycharmProjects/F_G_P/KDD-dataprepro/NSL-KDD_encoded&pca_traindata.csv'
    test_output = 'D:/Users/oyyk/PycharmProjects/F_G_P/KDD-dataprepro/NSL-KDD_encoded&pca_testdata.csv'
    pcaPath = 'D:/Users/oyyk/PycharmProjects/F_G_P/KDD-dataprepro/pca_path.pt'

    raw_data = pd.read_csv(kdd_testfilepath)
    raw_data.drop(raw_data.columns[0], axis=1, inplace=True)#丢弃第一维特征
    rest_data = raw_data
    rest_data = rest_data.drop(rest_data.columns[0:3], axis=1)#去除前三列单独处理
    protocol_col = raw_data.iloc[:,0]
    service_col = raw_data.iloc[:,1]
    flag_col = raw_data.iloc[:,2]

    protocol_col = np.array(protocol_col)
    service_col = np.array(service_col)
    flag_col = np.array(flag_col)
    #给三种非数值型特征编码
    protocol_col = np_utils.to_categorical(protocol_col, num_classes=3)
    service_col = np_utils.to_categorical(service_col, num_classes=70)
    flag_col = np_utils.to_categorical(flag_col, num_classes=11)

    protocol_col = pd.DataFrame(data=protocol_col, columns=['protocol_{0}'.format(i) for i in range(3)])
    service_col = pd.DataFrame(data=service_col, columns=['service_{0}'.format(i) for i in range(70)])
    flag_col = pd.DataFrame(data=flag_col, columns=['flag_{0}'.format(i) for i in range(11)])

    rest_data = load_pca(rest_data, pcaPath)

    dest_data = pd.concat([protocol_col, service_col, flag_col, rest_data], axis=1)
    dest_data.to_csv(test_output, index=False)

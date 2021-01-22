import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.decomposition import KernelPCA
from sklearn import preprocessing
import  matplotlib.pyplot as plt
# pd.set_option('display.max_row', None)

file_path = "D:/Users/oyyk/PycharmProjects/F_G_P/data/test_few/train_forkpca.csv"
kpca_path = "D:/Users/oyyk/PycharmProjects/F_G_P/scripts/train/few_shot/results/kpca_path.pt"
out_file_path = "D:/Users/oyyk/PycharmProjects/F_G_P/data/test_few/train_afterKpca.csv"
data = pd.read_csv(file_path)
x = data.drop(columns=['label'])
y = data.label
scaled_data = preprocessing.scale(x)
gamma = 0.2
coef0 = 0.05
feature_num = 121
kernels=['linear','poly','rbf','sigmoid']
pca = KernelPCA(n_components=feature_num, kernel=kernels[3], gamma=gamma, coef0=coef0)
pca.fit(scaled_data,y)
joblib.dump(pca, kpca_path)
pca_data = pca.transform(scaled_data)
pca_lamda = pca.lambdas_
labels = ['PC'+str(x) for x in range(1, feature_num+1)]
pca_df = pd.DataFrame(pca_data, columns=labels)
pca_df['label'] = y

pca_df.to_csv(out_file_path, index=None)

# fig=plt.figure()
#     # 颜色集合，不同标记的样本染不同的颜色
# colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),(0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2))
# catlog = (0,1,2,3,4)
#
# pca_df = pd.DataFrame(pca_data, columns=range(len(pca_lamda)))
# plt.title('My KPCA (gamma:{0}, coef0=:{1} Graph'.format(gamma,coef0))
# plt.xlabel('PC1')
# plt.ylabel('PC2')
#
# for label ,color in zip( np.unique(y),colors):
#             position=y==label
#             x=pca_data[position,0]
#             plt.scatter(pca_data[position,0],pca_data[position,1],
#             color=color)
# plt.show()
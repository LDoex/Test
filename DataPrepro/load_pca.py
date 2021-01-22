import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn import preprocessing
import  matplotlib.pyplot as plt
# pd.set_option('display.max_row', None)

file_path = "D:/Users/oyyk/PycharmProjects/F_G_P/data/test_few/test_forpca.csv"
pca_path = "D:/Users/oyyk/PycharmProjects/F_G_P/scripts/train/few_shot/results/pca_path.pt"
out_file_path = "D:/Users/oyyk/PycharmProjects/F_G_P/data/test_few/test_afterpca.csv"

data = pd.read_csv(file_path)
x = data.drop(columns=['label'])
y = data.label
scaled_data = preprocessing.scale(x)
# feature_num = 121
# pca = PCA(n_components=feature_num)
# pca.fit(scaled_data)
pca = joblib.load(pca_path)
pca_data = pca.transform(scaled_data)
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = ['PC'+str(x) for x in range(1, len(per_var)+1)]
pca_df = pd.DataFrame(pca_data, columns=labels)
pca_df['label'] = y

pca_df.to_csv(out_file_path, index=None)

# print(pca_df)
# # plt.scatter(pca_df.PC1, pca_df.PC2)
# plt.title('My PCA_from_2 Graph')
# plt.xlabel('PC1- {0}%'.format(per_var[1]))
# plt.ylabel('PC2- {0}%'.format(per_var[2]))
# labels = ['PC'+str(x) for x in range(1, len(per_var)+1)]
# pca_df = pd.DataFrame(pca_data, columns=labels)
# pca_df['label'] = y
#
#     # 颜色集合，不同标记的样本染不同的颜色
# colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),(0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2))
#
# for label ,color in zip( np.unique(y),colors):
#             position=y==label
#             plt.scatter(pca_data[position,0],pca_data[position,1],
#             color=color)
# # for i in range(pca_df.shape[0]):
# #     plt.annotate(i, (pca_df.PC1.iloc[i],pca_df.PC2.iloc[i]))
# plt.show()
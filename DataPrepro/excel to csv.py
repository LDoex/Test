import pandas as pd
file_path = 'KDDFC_train_0.xls'
data = pd.read_excel(file_path, 'Sheet1', index_col=0)
data.to_csv('KDDFC_train_0.csv', index=False, encoding='gbk')
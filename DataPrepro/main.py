import DataPrepro.functions as func
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
pd.set_option('display.max_row', None)

file_path = "D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/Muticlass_csv_1.csv"

data = pd.read_csv(file_path)
y = data.iloc[:, 26]
print()
x = [i for i in range(78377)]
plt.scatter(x, y)
plt.title(y.name)
plt.show()
# x33 = data[0]
# y33 = data[1]
# plt.plot(x33, y33, color="#800080", linewidth=2.0, linestyle="-", label="y2")
# plt.show()
# col=data.columns
# func.op_csvs(col)




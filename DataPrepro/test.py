import DataPrepro.functionswithone as func
import pandas as pd
import numpy as np
# import scripts.normalization.multi_class as mc
import sys
print(sys.argv[0])
output_name = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/groups/Muticlass_15_outinf.csv'
csv_li=[15]
scenes=func.scenarios()
data=func.csv_joint(csv_li)
data=func.op_inf(data)
# data=func.frames_norm(data)
data=func.label_set(data,scenes)
# func.cut_frames(frames)
data.to_csv(output_name, index=None)



# for i in range(5):
#     temp_results_path="data/results"+str(i)+".csv"
#     temp_results=pd.DataFrame(columns=func.attributes)
#     temp_path0 = "data/groups/group" + str(i)+"/"
#     for j in range(40):
#         temp_path=temp_path0+str(j)+".csv"
#         temp_df=pd.read_csv(temp_path)
#         temp_results=temp_results.append(temp_df,ignore_index=True)
#     temp_results.to_csv(temp_results_path)
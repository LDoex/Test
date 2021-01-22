import numpy as np
import csv
import pandas as pd
from sklearn import preprocessing


#  场景分类
def scenarios():
    # Natural Events,No Events,
    # A1_Data Injection,A2_Remote Tripping,A3Relay Setting Change
    s1 = np.array(np.append(np.arange(1, 7), np.array([13, 14])))
    s2 = np.array([41])
    s3 = np.array(np.arange(7, 13))
    s4 = np.array(np.arange(15, 21))
    s5 = np.array(np.append(np.arange(21, 31), np.arange(35, 41)))
    scenes = np.array([s1, s2, s3, s4, s5])
    return scenes


#  Z-score标准化
def z_scoreNorm(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    z_scoreNormed = (data - mean) / std
    z_scoreNormed = z_scoreNormed.dropna(axis=1)
    return z_scoreNormed


#  min-max标准化
def min_maxNorm(data):
    save_M=data['marker']
    data = data.dropna(axis=0)
    scaler = preprocessing.MinMaxScaler()
    min_maxNorm = scaler.fit_transform(data)
    min_maxNorm[:,-1]=save_M
    print(min_maxNorm[:,-1].shape)
    return min_maxNorm


# 检测含有inf的特征
def search_Inf(data):
    origin_max = data.max()
    features_Inf = np.array([])
    for o in origin_max.index:
        # print("checking", o)
        if origin_max[o] == float('inf'):
            # print("inf detected in",o)
            features_Inf = np.append(features_Inf, o)
    return features_Inf


# 将inf替换为（最大值的multi倍，默认为5）平均值
def op_Inf(data,multi=1):
    features_Inf=search_Inf(data)
    for f in features_Inf:
        flag = f
        temp=np.array([])
        for i in range(data.shape[0]):
            d=data[flag][i]
            if d==float('inf'):
                #print("inf detected at",i)
                data.loc[i,flag]=-999
                temp=np.append(temp,i)
        sec_val= data[flag].mean()
        for i in temp:
            data.loc[i, flag] =float(sec_val)*multi


# 创建空的frames[]
def create_frames(data):
    frames=[]
    frame = pd.DataFrame(data=None, columns=data.columns)
    for i in range(5):
        frames.append(frame)
    return frames


# 将data按照scenes分类
def depart_set(data,scenes):
    frames=create_frames(data)
    for i in range(data.shape[0]):
        label=data['marker'][i]
        print(i,end=" ")
        if label in scenes[0]:
            print("detected in 0")
            frames[0] = frames[0].append(data.loc[i])
        elif label in scenes[1]:
            print("detected in 1")
            frames[1] = frames[1].append(data.loc[i])
        elif label in scenes[2]:
            print("detected in 2")
            frames[2] = frames[2].append(data.loc[i])
        elif label in scenes[3]:
            print("detected in 3")
            frames[3] = frames[3].append(data.loc[i])
        elif label in scenes[4]:
            print("detected in 4")
            frames[4] = frames[4].append(data.loc[i])
    frames=[frames[0], frames[1], frames[2], frames[3], frames[4]]
    return frames


# 将frames[]存为csv文件
def frames_into_csv(frames):
    for i in range(5):
        name = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/Results_' + str(i) + '.csv'
        temp=min_maxNorm(frames[i])
        frames[i] = pd.DataFrame(temp, columns=frames[i].columns)
        frames[i].to_csv(name, index=None)



# 处理单个的data
def op_csv(data):
    scenes=scenarios()
    op_Inf(data, 5)
    frames = depart_set(data, scenes)
    print(type(frames[0]))
    frames_into_csv(frames)


# 处理15个data
def op_csvs(col):
    data = pd.DataFrame(data=None,columns=col)
    for i in range(1,16):
        name="D:/Users/oyyk/PycharmProjects/F_G_P/data/morris_fs/Muticlass_csv_"+str(i)+".csv"
        temp=pd.read_csv(name)
        data=data.append(temp, ignore_index=True)
    print(data)
    op_csv(data)


#  切割csv文件为num个，并放进相应的group文件夹
def cut_csv(num=40):
    for i in range(5):
        csv_name="D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/Results_"+str(i)+".csv"
        group_name="D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/group"+str(i)+"/"
        data=pd.read_csv(csv_name)
        gap=data.shape[0]//num  # 向下取整，末尾的样本丢弃
        for j in range(num):
            start=j*num
            temp_name=group_name+str(j)+".csv"
            temp_frame=data[start:start+gap]
            temp_frame.to_csv(temp_name,index=None)
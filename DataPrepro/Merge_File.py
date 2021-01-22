import csv
import pandas as pd
import os
from ast import literal_eval

#从第一个data中删除第二个data的部分,返回去重后的data
def del_repeat (allfile_name, supportfile_name):
    all_df = pd.read_csv(allfile_name)
    support_df = pd.read_csv(supportfile_name)
    #以三列综合判断支持集中的数据是否在全集中出现，若出现则去掉相应样本
    pc1_bool = all_df['PC1'].isin(support_df['PC1'])
    pc2_bool = all_df['PC2'].isin(support_df['PC2'])
    pc3_bool = all_df['PC3'].isin(support_df['PC3'])
    k = pd.Series(data = None, index=range(len(pc1_bool)), dtype=bool) #把三列的bool值合成一列
    for i in range(pc1_bool.size):
        if pc1_bool.values[i] ==True or pc2_bool.values[i] == True or \
                pc3_bool.values[i]==True:
            k.values[i] = True
        else:
            k.values[i] = False

    df_filter = all_df[~k]
    # for i in range(all_df.shape[0]):
    #     print('The cur_num is {0}'.format(i))
    #     for j in range(support_df.shape[0]):
    #         list1 = all_df.loc[i].astype(str).tolist() #将series转为list,然后join成str，方便比较
    #         str1 = ','.join(list1)
    #         list2 = support_df.loc[i].astype(str).tolist()
    #         str2 = ','.join(list2)
    #         if str1 == str2:
    #             all_df = all_df.drop(index=i)
    return df_filter


# 创建空的frames[]
def create_frames(data, num):
    frames=[]
    frame = pd.DataFrame(data=None, columns=data.columns)
    for i in range(num):
        frames.append(frame)
    return frames

#按指定的索引提出supportset并返回supportset
def find_support(data, allfile_name, per_episode_num):
    support_dfs = [] #supportset集
    query_dfs = [] #query集
    query_df = pd.DataFrame(data=None, columns=range(325))
    support_df = pd.DataFrame(data=None, columns=range(325))
    for i in range(int(data.shape[0]/per_episode_num)):
        support_dfs.append(support_df)
        query_dfs.append(query_df)
    for i in range(data.shape[0]):
        print('current_index:{0}'.format(i))
        sup_num = i//per_episode_num
        classfile_name = allfile_name + 'Muticlass_t_' + str(data.loc[i, 'class_name']) +'/' #定位到类别文件夹
        file_name = classfile_name + str(data.loc[i, 'file_num']) + '.csv'
        temp_data = pd.read_csv(file_name)
        if i == 0:   #如果是第一次处理，给两个集合赋值列索引
            for j in range(len(support_dfs)):
                support_dfs[sup_num].columns = temp_data.columns
                query_dfs[sup_num].columns = temp_data.columns
        index = data.loc[i, 'index_num']
        if data.loc[i, 'is_support'] == 1:
            support_dfs[sup_num-1] = support_dfs[sup_num-1].append(temp_data.loc[index])
        else:
            query_dfs[sup_num-1] = query_dfs[sup_num-1].append(temp_data.loc[index])
    return support_dfs, query_dfs



#将单个类别文件分成多份
def cut_frames(frames, file_name,num=40):
    for i in range(len(frames)):
        group_name = file_name+"group"+str(i)+"/"
        #file_name = "D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/groups/"
        if not os.path.exists(group_name):
            os.mkdir(group_name)
        temp_data = frames[i]
        gap = temp_data.shape[0]//num
        print(gap)
        for j in range(num):
            start = j * gap
            temp_name = group_name + str(j) + ".csv"
            temp_frame = temp_data[start:start + gap]
            temp_frame.to_csv(temp_name, index=None)


#按类别分割一个文件
def split_bylabel(data, label):
    frames = create_frames(data, len(label))
    for i in range(data.shape[0]):
        print(i,end=" ")
        for k in label:
            if data.loc[i, 'label'] == k:
                print("detected in {0}".format(k))
                frames[k] = frames[k].append(data.loc[i])
    frames = [frames[0], frames[1], frames[2], frames[3], frames[4]]
    return frames

#给标签重排序命名, class_index是字典， key对应修改前标签， 值对应修改后标签
def reset_label(data, class_index):
    for i in range(data.shape[0]):
        print('The current_num is {0}...'.format(i))
        for key, value in class_index.items():
            if int(key) == int(data.loc[i, 'label']):
                data.loc[i, 'label'] = class_index[key]
    return data

#按所给标签合并dataframe
def merge_by_label(data, label):
    frames = pd.DataFrame(data=None, columns=data.columns)
    for i in range(data.shape[0]):
        print('The current_num is {0}...'.format(i))
        d_label = data.loc[i,'label']
        for j in range(len(label)):
            if label[j] == d_label:
                frames = frames.append(data.iloc[i])
    return frames

def merge_file(file_name, file_pre , first_num , file_num):
    for i in range(file_num-1):
        #选择第一个文件作为合并后的文件
        file_name_raw = file_name+file_pre+str(first_num)+'.csv'
        #循环处理之后的文件
        file_name_p = file_name + file_pre + str(i+ first_num+ 1) + '.csv'
        reader = csv.DictReader(open(file_name_p))
        header = reader.fieldnames
        with open(file_name_raw,'a',newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = header)
            writer.writerows(reader)

#取对应数量的样本组成dataframe返回
def get_sample(data, class_index, del_row = False):
    data_re = pd.DataFrame(data=None, columns=data.columns)
    for i in range(data.shape[0]):
        for key,value in class_index.items():
            if class_index[key] != 0 and int(key) == int(data.loc[i, 'label']):
                data_re = data_re.append(data.loc[i],ignore_index=True)
                if del_row :
                    data = data.drop(index=i)
                class_index[key] -= 1
    return data, data_re

#把data按对应的support和query分成两份data返回
def split_SandQ(data, class_num, all_num, sup_num, que_num):
    mul = 0
    ever = all_num/class_num
    temp_sup_num = sup_num
    temp_que_num = que_num
    data_sup = pd.DataFrame(data=None,columns=data.columns)
    data_que = pd.DataFrame(data=None, columns=data.columns)
    for i in range(1,class_num+1):
        k = mul #用于更换类别起始索引点
        for j in range(int(ever)):
            ind = j+k
            if temp_sup_num!= 0:
                data_sup = data_sup.append(data.loc[ind])
                temp_sup_num -= 1
            if temp_sup_num== 0 and temp_que_num!= 0:
                data_que = data_que.append(data.loc[ind+1])
                temp_que_num -= 1
            mul += 1
        temp_sup_num = sup_num #重置每个类别shot数
        temp_que_num = que_num
    return data_sup, data_que


if __name__ == '__main__':
        # file_name_all = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/test_few/1to10_train.csv'
        # file_name_train = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/test_few/2-4/train/train_afterpca.csv'
        # file_name_test = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/test_few/2-4/test/test_afterpca.csv'
        # file_name = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/test_few/2-4/test/'

        #####合并文件
        # merge_file_name = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/test_few/'
        # merge_file(merge_file_name, file_pre='kpca_', first_num=1 , file_num=2)


        # file_name_output = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/test_few/2-4/train/train_class_0.csv'
        # fewshot_output = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/'
        # data_train = pd.read_csv(file_name_train)
        # data_test = pd.read_csv(file_name_test)

        ####按标签取出data且合并
        # label = [0]
        # data_merge = merge_by_label(data_train, label)
        # data_merge.to_csv(file_name_output, index=None)


        ####按标签取小样本（用于取小样本去pca）
        # train_shot_num = {'0':20, '1':20, '2':20}
        # test_shot_num = {'2':5, '4':5}
        # train_data = get_sample(data_train, train_shot_num)
        # all_test_data, test_data = get_sample(data_test, test_shot_num, del_row=True)
        # test_data.to_csv(file_name_output,index=None)
        # all_test_data.to_csv(file_name_test, index=None)

        # train_support, train_query = split_SandQ(train_data, 3, 60, 5, 15)
        # test_support, test_query = split_SandQ(test_data, 3, 60, 5, 15)
        # train_support.to_csv(fewshot_output+'Muticlass_0-1-3_fewtrain.csv',index=None)
        # test_query.to_csv(fewshot_output+'Muticlass_0-1-3_fewtest.csv',index=None)


        ####重命名标签, 普通cnn需要按顺序onehot
        # re_label = {'0':3, '1':4}
        # data_relabel = pd.read_csv(file_name_output)
        # data_out = reset_label(data_relabel, re_label)
        # data_out.to_csv(file_name+'2-4_5shot-test_relabel.csv',index=None)


        ###把一个文件按类别分开
        # out_file_name = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/supportsets/'
        # input_path = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/supportsets/test_del_support.csv'
        # data = pd.read_csv(input_path)
        # label = [0,1,2,3,4]
        # df = split_bylabel(data, label)
        # for i in range(len(label)):
        #     df[i].to_csv(out_file_name+'test_class_'+str(i)+'.csv',index=None)

        ###把一个文件按类别分开后分成40个文件
        # train_path = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/test_few/2-4-kpca/train/train_afterKpca.csv'
        # test_path = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/test_few/2-4-kpca/test/test_afterKpca.csv'
        # test_file_name = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/test_few/2-4-kpca/test/'
        # train_file_name = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/test_few/2-4-kpca/train/'
        # data_train = pd.read_csv(train_path)
        # data_test = pd.read_csv(test_path)
        # label = [0,1,2,3,4]
        # df = split_bylabel(data_test, label)
        # for i in range(len(label)):
        #     df[i].to_csv(test_file_name+'test_class_'+str(i)+'.csv',index=None)
        # cut_frames(df,test_file_name,20)

        ####循环输出每个episode的supportset和queryset
        index_file = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/proto_Results.csv'
        all_file = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/'
        output_file = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/7/1shot-correct/'
        index_data = pd.read_csv(index_file)
        support_sets, query_sets = find_support(index_data, all_file, 60)
        for i in range(len(support_sets)):
            support_outpath = output_file + 'support_ep_' + str(i) + '.csv'
            query_outpath = output_file + 'query_ep_' + str(i) + '.csv'
            support_sets[i].to_csv(support_outpath, index=None)
            query_sets[i].to_csv(query_outpath, index=None)




        ###从总测试集中删去支持集
        # all_name = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/test_afterpca.csv'
        # support_name = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/episode_0.csv'
        # output_path = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/test_delsupport.csv'
        # ouput_data = del_repeat(all_name, support_name)
        # ouput_data.to_csv(output_path,index=None)



import os
import sys
import glob
import pandas as pd
from functools import partial
import torch
from torchvision.transforms import ToTensor
import csv
from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler
import numpy as np

Morris_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/Morris')
Morris_CACHE = { }
data_frame = pd.DataFrame(data=None, columns=['class_name', 'file_num', 'index_num'])
data_frame_temp = pd.DataFrame(data=None, columns=['class_name', 'file_num', 'index_num'])
name_dic = {'0':'Muticlass_0', '1':'Muticlass_1', '2':'Muticlass_2', '3':'Muticlass_3',
            '4':'Muticlass_4', '5':'Muticlass_5', '6':'Muticlass_6', '7':'Muticlass_7'}

def record(key, index_num):
    l_ = []
    global data_frame
    for k in name_dic:
        if key.find(name_dic[k])!=-1: #查找相等字符串，查不到返回-1
            if l_:
                l_[0] == k
            else:
                l_.append(k)
    for i in range(20):
        s_ = str(i)+'.csv'
        if key.find(s_)!=-1:
            o = len(l_)
            if o == 2:
                l_[1] = str(i)
            else:
                l_.append(str(i))
    l_.append(index_num)
    l_ = pd.DataFrame(l_)
    l_ = l_.values.reshape(1,-1)
    l_ = pd.DataFrame(l_, columns=data_frame.columns)
    data_frame = data_frame.append(l_, ignore_index=True)

def append_support_index (support_index, data_frame):
    data_frame['is_support'] = '0'
    for i in range(support_index.shape[0]):
        index = support_index[i]
        data_frame.loc[index, 'is_support'] = '1'
    return data_frame



def frames_into_csv(frames):
    name = '../../../data/RawData/output/Train_index_temp.csv'
    frames.to_csv(name, index=None)
    file_name_raw = '../../../data/RawData/output/Train_index.csv'
    reader = csv.DictReader(open(name))
    header = reader.fieldnames
    with open(file_name_raw, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = header)
        writer.writerows(reader)


def data_prepro(key, out_field, d):
    X = pd.read_csv(d[key])
    m_ = X.sample(n=1)

    # record the support_index, just use when training
    #####
    # record(d[key],m_.index.values[0])
    #####

    m_ = m_.values
    m_ = m_[:, :121]
    m_ = np.array(m_)
    d[out_field] = m_
    return d

def convert_tensor(key, d):
    d[key] = 1.0 - torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d[key].shape[0], d[key].shape[1])
    return d

def scale_data(key, height, width, d):
    d[key] = d[key].reshape((height,width))
    return d

def load_class_data(d):

    alphabet = d['class']
    data_dir = os.path.join(Morris_DATA_DIR, 'data__', alphabet)

    class_data = sorted(glob.glob(os.path.join(data_dir, '*.csv')))

    if len(class_data) == 0:
        raise Exception("No data found for omniglot class {} at {}.".format(d['class'], data_dir))

    data_ds = TransformDataset(ListDataset(class_data),
                                compose([partial(convert_dict, 'file_name'),
                                         partial(data_prepro, 'file_name', 'data'),
                                         partial(scale_data, 'data', 11 , 11),
                                         partial(convert_tensor, 'data')]))

    loader = torch.utils.data.DataLoader(data_ds, batch_size=len(data_ds), shuffle=False)
    for sample in loader:
        Morris_CACHE[d['class']] = sample['data']
        break # only need one sample because batch size equal to dataset length


    return { 'class': d['class'], 'data': Morris_CACHE[d['class']] }

def extract_episode(n_support, n_query, d):
    # data: N x C x H x W
    global data_frame
    global data_frame_temp
    n_examples = d['data'].size(0)

    if n_query == -1:
        n_query = n_examples - n_support

    example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = d['data'][support_inds]
    xq = d['data'][query_inds]

    #record the support_index, just use when training
    #####
    # support_index = np.array(support_inds)
    # data_to_csv = append_support_index(support_index, data_frame)
    # frames_into_csv(data_to_csv)
    # data_frame = data_frame_temp
    #####

    return {
        'class': d['class'],
        'xs': xs,
        'xq': xq
    }

def load(opt, splits):
    split_dir = os.path.join(Morris_DATA_DIR, 'splits', opt['data.split'])

    ret = { }
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']


        transforms = [partial(convert_dict, 'class'),
                      load_class_data,
                      partial(extract_episode, n_support, n_query)]

        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms) #check transforms


        class_names = []
        with open(os.path.join(split_dir, "{:s}.txt".format(split)), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))
        ds = TransformDataset(ListDataset(class_names), transforms)

        if opt['data.sequential']:
            sampler = SequentialBatchSampler(len(ds))
        else:
            sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)



    return ret
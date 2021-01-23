import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv, os
import sys

from torch.autograd import Variable
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from protonets.models import register_model
from .dist import euclidean_dist
from .lsoftmax import log_softmax, softmax, DistillingLoss

''''
class LSoftmax(nn.Module):
    def __init__(self):
        self.lsoftmax_linear = LSoftmaxLinear(input_dim= , output_dim= , margin= )

    def reset_parameters(self):
'''


def f_score(cluster, labels):
    TP, TN, FP, FN = 0, 0, 0, 0
    n = len(labels)
    # a lookup table
    for i in range(n):
        if i not in cluster:
            continue
        for j in range(i + 1, n):
            if j not in cluster:
                continue
            same_label = (labels[i] == labels[j])
            same_cluster = (cluster[i] == cluster[j])
            if same_cluster:
                if same_label:
                    TP += 1
                else:
                    FP += 1
            elif same_label:
                FN += 1
            else:
                TN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fscore = 2 * precision * recall / (precision + recall)
    return fscore, precision, recall, TP + FP + FN + TN


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        self.encoder = encoder

    def loss(self, sample):
        xs = Variable(sample['xs'])  # support
        xq = Variable(sample['xq'])  # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)

        
        # save z into file, just use when run eval
        ######
        # index_support = np.row_stack((np.zeros(n_support),np.ones(n_support))).reshape(-1)
        # index_query = np.row_stack((np.zeros(n_query),np.ones(n_query))).reshape(-1)
        # index = np.append(index_support, index_query)
        #
        # result_temp = np.array(z.cpu().detach().numpy())
        # result = np.column_stack((result_temp, index))
        #
        # rows = []
        #
        # for i in range(result.shape[0]):
        #     rows.append(result[i])
        # csv_path = "result.csv"
        # #if the path exists, use 'a' to append, else use 'w' to creat
        # if not os.path.exists(csv_path):
        #     with open(csv_path, 'w', newline='') as file:
        #         csvwriter = csv.writer(file)
        #         csvwriter.writerows(rows)
        # else:
        #     with open(csv_path, 'a', newline='') as file:
        #         csvwriter = csv.writer(file)
        #         csvwriter.writerows(rows)
        # file.close()
        ######

        # online learning, use when training
        #####
        # d = 64
        # itr = int(n_class*(n_support+n_query)/n_support)
        #
        # index_support = np.row_stack((np.zeros(n_support), np.ones(n_support))).reshape(-1)
        # index_query = np.row_stack((np.zeros(n_query), np.ones(n_query))).reshape(-1)
        # index = np.append(index_support, index_query)
        #
        # result_temp = np.array(z.cpu().detach().numpy())
        # result = np.column_stack((result_temp, index))
        #
        # trainSet = np.array(result)
        # ftrl = ft.FTRL(dim=d, l1=0.001, l2=0.1, alpha=0.1, beta=1e-8)
        # all_loss, all_step = ftrl.training(trainSet, batch=n_support, dim=d, max_itr=itr)
        # all_loss = np.array(all_loss)
        # loss_online = np.mean(all_loss)
        #####

        z_dim = z.size(-1)

        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class * n_support:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        _, y_hat = log_p_y.max(2)

        # computing the teacher model softmax
        ###
        distilling_model_path = '../../../scripts/train/few_shot/results/big_best_model.pt'
        model_distilling = torch.load(distilling_model_path)
        model_distilling.eval()
        if xq.is_cuda:
            model_distilling.cuda()
        z_distilling = model_distilling.encoder.forward(x)

        z_distilling_dim = z_distilling.size(-1)

        z_distilling_proto = z_distilling[:n_class * n_support].view(n_class, n_support, z_distilling_dim).mean(1)
        zq_distilling = z_distilling[n_class * n_support:]

        z_distilling_dists = euclidean_dist(zq_distilling, z_distilling_proto)
        T = 20
        student_distilled = F.softmax(-dists * 1.0 / T, dim=1).view(n_class, n_query, -1)
        teacher_distilled = F.softmax(-z_distilling_dists * 1.0 / T, dim=1).view(n_class, n_query, -1)
        ###

        # #distilling loss
        # ####
        # a = 0.8
        # T_ = T*T
        # my_celoss = DistillingLoss()
        # loss_soft = my_celoss(student_distilled, teacher_distilled)
        # KD_loss= nn.KLDivLoss()(F.log_softmax(-dists*1.0/T, dim=1), F.softmax(-z_distilling_dists*1.0/T, dim=1))
        # loss_hard = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        # loss_val = a*KD_loss*T_ + (1-a)*loss_hard

        # #Teacher_Training_loss
        # log_distill = F.log_softmax(-dists * 1.0 / T, dim=1).view(n_class, n_query, -1)
        # loss_distill = -log_distill.gather(2, target_inds).squeeze().view(-1).mean()
        #
        # loss_val = loss_distill

        # eval loss
        ###
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        ###

        # training loss
        #####
        # loss_few = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        # loss_val =0.8*loss_few+0.2*loss_online
        #####

        y_re = target_inds.squeeze()

        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        y_real = np.array(y_re.cpu()).reshape(-1)
        y_pred = np.array(y_hat.cpu()).reshape(-1)
        acc = accuracy_score(y_real, y_pred)  # TP+TN/(TP+FN+FP+TN)
        pre = precision_score(y_real, y_pred, average='binary')  # TP/TP+FP
        rec = recall_score(y_real, y_pred, average='binary')  # TP/TP+FN
        F1s = f1_score(y_real, y_pred, average='binary')  # 2*(pre*recall/(pre+recall))
        # F1s, pre, rec, TP = f_score(y_real, y_pred)

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'Accuracy': acc,
            'Precision': pre,
            'Recall': rec,
            'F1': F1s
        }


@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']


    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], 64),
        # conv_block(16, 32),

        # conv_block(16, 16),
        Flatten()
    )

    return Protonet(encoder)

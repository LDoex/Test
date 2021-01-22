import torch
import numpy as np
import random
# def foo():
#     print("starting...")
#     while True:
#         yield torch.cat((torch.zeros(1).long(), torch.from_numpy(np.array(random.sample(range(1,5),1))).long()))
#         print("res:")
# for i in range(5):
#     g = foo()
#     print(g)

# print(next(g))
# print("*"*20)
# print(next(g))
# print(next(g))
# print(next(g))

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.cat((torch.zeros(1).long(),
                             # torch.ones(1).long(),
                             torch.from_numpy
                             (np.array
                              (random.sample(range(1,self.n_classes),self.n_way-1))).long()))

sampler = EpisodicBatchSampler(5, 2, 20)
sampler.__iter__()
print(sampler)
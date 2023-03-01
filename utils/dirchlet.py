import numpy as np
import math
import torchvision

if __name__ == '__main__':
    size = 4
    dirichlet_param=[10, 5, 3, 2, 3, 1, 1, 3, 4, 5]
    flip = [11-dirichlet_param[i] for i in range(len(dirichlet_param))]
    prob = np.random.default_rng().dirichlet(tuple(flip), 1)
    print(prob)
    probFlip = np.random.default_rng().dirichlet(tuple(dirichlet_param), size-1)
    print('=================')
    print(np.concatenate((prob, np.random.default_rng().dirichlet(tuple(dirichlet_param), size-1))))
    # print(prob)
    print('=================')
    print(np.concatenate((prob, np.random.default_rng().dirichlet(tuple(dirichlet_param), size))))
    # size = 4

    # dataset = [torchvision.datasets.CIFAR10('../data/', download=True) for i in range(size)]
    
    # prob = np.random.default_rng().dirichlet((10, 5, 3, 2, 3, 1, 2, 3, 4, 5), size)
    # prob_col_sum = np.sum(prob, axis=0)
    # max_col = np.max(prob_col_sum)
    # normalized_sum = [col / max_col for col in prob_col_sum]
    # count = [(prob[i] / prob_col_sum[i] * normalized_sum[i] * 5000).tolist() for i in range(size)]
    # for i in range(size): # round to int
    #     for j in range(len(count[i])):
    #         count[i][j] = math.floor(count[i][j])
    # base_targets = dataset[0].targets.copy()
    # client_idx = [[] for i in range(size)]
    # for j in range(10):
    #     class_idx = [i for i in range(len(base_targets)) if base_targets[i] == j]
    #     prev = 0
    #     for i in range(size):
    #         client_idx[i] += class_idx[prev: prev + count[i][j]]
    #         prev = prev + count[i][j]

    # for i in range(size):
    #     idx_set = set(client_idx[i])
    #     mask = [False for _ in range(len(dataset[i].targets))]
    #     for j in range(len(dataset[i].targets)):
    #         if j in idx_set:
    #             mask[j] = True
    #     dataset[i].targets = [dataset[i].targets[k] for k in range(len(mask)) if mask[k]]


        
    #     print('resultant target len:',len(dataset[i].targets))
    # print(len(client_idx[0]), sum(count[0]))
    # print(len(dataset[0].targets))
    # print(dataset[0].targets)
    # print(count[0])
    # cnt = [0 for i in range(10)]
    # for val in dataset[0].targets:
    #     cnt[val] += 1
    # print('cnt:', cnt)
    # print(len(client_idx[1]), sum(count[1]))
    # print(len(client_idx[2]), sum(count[2]))
    

    
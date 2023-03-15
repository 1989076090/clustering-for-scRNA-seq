import random
import numpy as np
import pandas as pd
import torch
from torch_sparse import SparseTensor
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import torch.nn as nn
import math
import pickle as pkl


class CLDataGenerater(Dataset):
    def __init__(self, A):
        self.map2 = {}

        for data in torch.nonzero(A):
            target = data[0].item()
            source = data[1].item()
            if target not in self.map2.keys():
                self.map2[target] = [source]
            else:
                self.map2[target].append(source)
        self.sources = [i for i in range(max(list(self.map2.keys())) + 1)]

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], random.choice(self.map2[self.sources[idx]])


def normalization(features):
    X = features.copy()
    for i in range(len(X)):
        X[i] = X[i] / sum(X[i]) * 100000
    X = np.log2(X + 1)
    return X

def load_data(data_path, data_pca=None, PCA_dim=512):
    # 如果存在，直接加载

    data = pd.read_csv(data_path, index_col=0, sep='\t')
    cells = data.columns.values
    genes = data.index.values
    features = data.values.T
    features = normalization(features)
    if data_pca!=None:
        with open(data_pca, 'rb') as pca_data:
            features = pkl.load(pca_data)
    else:
        if features.shape[0] > PCA_dim and features.shape[1] > PCA_dim:
            pca = PCA(n_components = PCA_dim)
            features = pca.fit_transform(features)
        else:
            var = np.var(features, axis=0)
            min_var = np.sort(var)[-1 * PCA_dim]
            features = features.T[var >= min_var].T
            features = features[:, :PCA_dim]

    features = (features - np.mean(features)) / (np.std(features))
    return features


def generateAdj(matrix, k = 10):

    size = matrix.shape[0]

    edgeList=[]
    for i in np.arange(matrix.shape[0]):
        tmp=matrix[i,:].reshape(1,-1)
        distMat = torch.cdist(tmp,matrix)
        res = distMat.argsort()[:k+1].cpu().detach().numpy()
        for j in np.arange(1,k+1):
            edgeList.append((i,res[0][j],1.0))
    index4 = [[], []]
    data4 = []

    for target, source, data in edgeList:
        index4[0].append(target)
        index4[1].append(source)
        data4.append(data)

    adj = SparseTensor.from_edge_index(torch.LongTensor(index4), torch.FloatTensor(data4), torch.Size([size, size]))
    return adj

class Cell_level_Loss(nn.Module):
    def __init__(self, temperature):
        super(Cell_level_Loss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, features, x_i, x_j, A):
        num_cells = features.shape[0]
        sim = torch.matmul(features, features.T) / self.temperature

        index = torch.cat([x_i.unsqueeze(0),x_j.unsqueeze(0)],dim=0).cpu()
        label_mask = SparseTensor.from_edge_index(torch.LongTensor(index),
                                                  sparse_sizes=torch.Size([num_cells, num_cells])).to_dense().float().to(features.device)

        positive_samples = sim[label_mask.bool()].reshape(num_cells, 1)
        A = 1-A
        negative_samples = sim[A.bool()].reshape(num_cells, -1)

        labels = torch.zeros(num_cells).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)

        return loss

class Cluster_level_Loss(nn.Module):
    def __init__(self, class_num, temperature):
        super(Cluster_level_Loss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
import torch.nn as nn
import math
import pickle as pkl
from sklearn.metrics.pairwise import pairwise_distances
from igraph import *
from torch import optim
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from sklearn.metrics.cluster import contingency_matrix

def normalization(features_):
    features = features_.copy()
    for i in range(len(features)):
        features[i] = features[i] / sum(features[i]) * 100000
    features = np.log2(features + 1)
    return features

def load_label(data_path):
    # 如果存在，直接加载
    with open(data_path, 'rb') as adjFile:
        cell_labels = pkl.load(adjFile)
    return cell_labels, len(set(cell_labels))

def load_data(data_path):
    # 如果存在，直接加载
    data = pd.read_csv(data_path, index_col=0)
    cells = data.columns.values
    genes = data.index.values
    features = data.values.T
    return features, cells, genes

def training(X, model, adjs4CL, device, cell_labels=None, epoch=100, ty=2.0, tc=1.0, lr=3e-4):
    model = model.to(device)
    n_cluster = len(set(cell_labels))
    criterion_cell = Cell_level_loss(X.shape[0], ty)
    criterion_cluster = Cluster_level_loss(n_cluster, tc, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in range(epoch):
        model.train()
        optimizer.zero_grad()

        augment_X = X[[np.random.choice(line) for line in adjs4CL]]
        z_i, z_j, c_i, c_j = model(X, augment_X)

        loss_inst = criterion_cell(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)

        loss = loss_inst+loss_cluster

        if i%10==0 and cell_labels is not None:
            model.eval()
            with torch.no_grad():
                _, hidden = model.forward_cluster(X)
                hidden = hidden.detach().cpu().numpy()
            predict = ClusteringWithClusters(hidden, n_cluster)
            ARI = adjusted_rand_score(cell_labels, predict)
            NMI = normalized_mutual_info_score(cell_labels, predict)
            print('epoch:{}, cluster: {}/{}, ARI: {:.4f},\
            NMI: {:.4f}'.format(i, len(set(predict)), n_cluster, ARI, NMI))

        loss.backward()
        optimizer.step()
    return model

# clustering with specific clusters
def ClusteringWithClusters(hidden, n_cluster):

    resolution = 0.01
    predict = LeidenClustering(hidden, resolution)
    times = 1
    while len(set(predict))!= n_cluster:
        verse = len(set(predict)) - n_cluster

        resolution = resolution - (len(set(predict)) - n_cluster)*resolution/10/times
        predict = LeidenClustering(hidden, resolution)
        if n_cluster == len(set(predict)):
            break
        if (len(set(predict)) - n_cluster) == verse:
            times=1
        else:
            times+=1
    return predict



def generateNeibour(features, k_GAT=100, k_CL=20, pca_path=None):

    if pca_path is not None:
        print('loading saved PCA data')
        with open(pca_path, 'rb') as pca_data:
                features = pkl.load(pca_data)
    else:
        pca = PCA(n_components=20)
        features = pca.fit_transform(normalization(features))

    adj_dist = pairwise_distances(features, metric='euclidean')

    adjs4CL = []
    adjs4GAT_mask = np.zeros(adj_dist.shape)

    for i in np.arange(adj_dist.shape[0]):
        adj_sorts = adj_dist[i].argsort()
        adjs4GAT_mask[i, adj_sorts[:k_GAT + 1]] = 1
        adjs4CL.append(adj_sorts[:k_CL + 1])

    adjs4CL = np.array(adjs4CL)
    adjs4GAT_mask = torch.BoolTensor(adjs4GAT_mask)

    return adjs4GAT_mask, adjs4CL


def LeidenClustering(hidden, resolution, k=20):
    hidden_knn = pairwise_distances(hidden, metric='euclidean')

    hidden_mat = np.zeros(hidden_knn.shape)
    for i in np.arange(hidden_knn.shape[0]):
        adj_sorts = hidden_knn[i].argsort()
        hidden_mat[i, adj_sorts[1:k + 1]] = 1

    hidden_SNN = np.matmul(hidden_mat, hidden_mat.T)

    graph = Graph.Weighted_Adjacency(hidden_SNN, mode=ADJ_UNDIRECTED, attr="weight", loops=False)
    leiden_partition = graph.community_leiden(resolution_parameter=resolution,
                                              weights=graph.es['weight'], n_iterations=10)

    results = np.zeros(hidden_SNN.shape[0])
    for i in range(len(leiden_partition)):
        results[leiden_partition[i]] = int(i)
    return results



class Cell_level_loss(nn.Module):
    def __init__(self, cell_num, temperature):
        super(Cell_level_loss, self).__init__()
        self.cell_num = cell_num
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(cell_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, cell_num):
        N = 2 * cell_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(cell_num):
            mask[i, cell_num + i] = 0
            mask[cell_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.cell_num
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.cell_num)
        sim_j_i = torch.diag(sim, -self.cell_num)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class Cluster_level_loss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(Cluster_level_loss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

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

def purity_score_real(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency = contingency_matrix(y_true, y_pred)

    consMat = np.copy(contingency)
    is_replace = [False for i in range(consMat.shape[0])]  # 代表替换的列

    max_array = np.argsort(-np.amax(consMat, axis=1))  # 从Label被predict最多的开始算

    for i in max_array:
        j = np.argmax(consMat[i])
        if i == np.argmax(consMat.T[j]) and is_replace[j] == False:  # 如果该列没被替换
            consMat[:, [i, j]] = consMat[:, [j, i]]
            is_replace[i] = True  # 该列已经替换到i

    ingore_index = np.where(np.array(is_replace) == False)[0]

    for i in ingore_index:
        arg_index = np.argsort(-consMat[i])
        for j in arg_index:
            if is_replace[j] == False:  # 如果该列没被替换
                consMat[:, [i, j]] = consMat[:, [j, i]]
                is_replace[i] = True  # 该列已经替换到i
                break

    # return purity
    return np.sum(np.diag(consMat)) / np.sum(consMat)

def getARI(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)

def getNMI(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred)

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency = contingency_matrix(y_true, y_pred)

    # return purity
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

def get_ECP(y_true, y_pred):
    contmat = contingency_matrix(y_true, y_pred)
    ECP = []
    for i in range(contmat.shape[0]):
        t = contmat[i][contmat[i] > 0]
        t = t / np.sum(t)
        ECP.append(-np.sum(t * np.log(t)))
    return np.mean(ECP)

def get_ECA(y_true, y_pred):
    contmat = contingency_matrix(y_true, y_pred)
    ECA = []
    for i in range(contmat.T.shape[0]):
        t = contmat.T[i][contmat.T[i] > 0]
        t = t / np.sum(t)
        ECA.append(-np.sum(t * np.log(t)))
    return np.mean(ECA)

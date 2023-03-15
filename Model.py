import torch.nn as nn
import torch
from torch.nn.functional import normalize
import torch.nn.functional as F
# from SparseGATConv import SparseGATConv
from graphgallery.nn.init.pytorch import glorot_uniform, zeros
from torch_geometric.nn.conv import GATConv

class GAT_encoder(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 attn_heads=1,
                 dropout_rate=0.2):
        super(GAT_encoder, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.attn_heads = attn_heads
        self.dropout_rate = dropout_rate

        # 512->64
        # self.conv1 = GATConv(self.in_features, self.hidden_features, heads=self.attn_heads)
        # 64->16
        # self.conv2 = GATConv(self.hidden_features*self.attn_heads, self.out_features, heads=self.attn_heads, concat=False)
        self.conv1 = SparseGATConv(self.in_features,
                                   self.hidden_features,
                                   attn_heads=self.attn_heads,
                                   dropout=self.dropout_rate)
        self.conv2 = SparseGATConv(self.hidden_features*self.attn_heads,
                                   self.out_features,
                                   attn_heads=self.attn_heads,
                                   dropout=self.dropout_rate,
                                   reduction='average')

    def forward(self, features, adj):
        x = F.relu(self.conv1(features, adj))
        x = F.relu(self.conv2(x, adj))

        return x

class Network(nn.Module):
    def __init__(self, pca_dim, hidden_dim, feature_dim, head, dropout_rate, class_num, X, adj):
        super(Network, self).__init__()

        self.encoder = GAT_encoder(pca_dim, hidden_dim, feature_dim, head, dropout_rate=dropout_rate)

        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.cell_projector = nn.Sequential(
            nn.Linear(self.encoder.out_features, self.encoder.out_features),
            nn.ReLU(),
            nn.Linear(self.encoder.out_features, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.encoder.out_features, self.encoder.out_features),
            nn.ReLU(),
            nn.Linear(self.encoder.out_features, self.cluster_num),
            nn.Softmax(dim=1)
        )

        self.X = X
        self.adj = adj

    def forward(self, x_i, x_j):
        outputs = self.encoder(self.X, self.adj)
        h_i = outputs[x_i]
        h_j = outputs[x_j]

        z_i = normalize(self.cell_projector(h_i), dim=1)
        z_j = normalize(self.cell_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self):
        h = self.encoder(self.X, self.adj)

        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c,h

    def get_cell_hidden(self):
        h = self.encoder(self.X, self.adj)
        e = self.cell_projector(h)
        return e


class Network2(nn.Module):
    def __init__(self, pca_dim, hidden_dim, feature_dim, head, dropout_rate, class_num, X, adj):
        super(Network2, self).__init__()

        self.encoder = GAT_encoder(pca_dim, hidden_dim, feature_dim, head, dropout_rate=dropout_rate)

        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.cell_projector = nn.Sequential(
            nn.Linear(self.encoder.out_features, self.encoder.out_features),
            nn.ReLU(),
            nn.Linear(self.encoder.out_features, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.encoder.out_features, self.encoder.out_features),
            nn.ReLU(),
            nn.Linear(self.encoder.out_features, self.cluster_num),
            nn.Softmax(dim=1)
        )

        self.X = X
        self.adj = adj

    def forward(self, x_i, x_j):
        outputs = self.encoder(self.X, self.adj)
        h_i = outputs[x_i]
        h_j = outputs[x_j]

        z_i = normalize(self.cell_projector(h_i), dim=1)
        z_j = normalize(self.cell_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self):
        h = self.encoder(self.X, self.adj)

        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c,h

    def get_cell_hidden(self):
        h = self.encoder(self.X, self.adj)
        e = self.cell_projector(h)
        return e

    def save_model(self, file_name):
        # print('saved model')
        torch.save(self.cpu().state_dict(), file_name)

    def load_model(self, path):
        print('loaded model')
        self.load_state_dict(torch.load(path))

class SparseGATConv(nn.Module):
    """
    Sparse version of GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self,
                 in_features,
                 out_features,
                 attn_heads=8,
                 alpha=0.2,
                 reduction='concat',
                 dropout=0.6,
                 bias=False):
        super().__init__()

        if reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.in_features = in_features
        self.out_features = out_features

        self.dropout = nn.Dropout(dropout)
        self.attn_heads = attn_heads
        self.reduction = reduction

        self.kernels = nn.ModuleList()
        self.att_kernels = nn.ModuleList()

        self.biases = nn.ParameterList()
        self.bias = bias

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            W = nn.Linear(in_features, out_features, bias=False)
            self.kernels.append(W)

            a = nn.Linear(2 * out_features, 1, bias=False)
            self.att_kernels.append(a)

            if bias:
                self.biases.append(nn.Linear(out_features, 1, bias=False))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):

        dv = x.device
        N = x.size()[0]
        # edges = adj._indices()
        edges = adj.coo()[:2]

        outputs = []
        for head in range(self.attn_heads):

            # 这里有改变
            W, a = self.kernels[head], self.att_kernels[head]
            Wh = W(x)
            Whij = torch.cat([Wh[edges[0]], Wh[edges[1]]], dim=1)
            edge_e = -self.leakyrelu(a(Whij))

            attention = self.dropout(self.sparse_softmax(edges, edge_e, N, dv))
            h_prime = self.sparse_matmul(edges, attention, Wh)

            if self.bias:
                h_prime += self.biases[head]

            outputs.append(h_prime)

        if self.reduction == 'concat':
            output = torch.cat(outputs, dim=1)
        else:
            output = torch.mean(torch.stack(outputs), 0)

        return output

    @staticmethod
    def sparse_softmax(edges, edge_e, N, device):
        """Softmax for sparse adjacency matrix"""

        source = edges[0]
        e_max = edge_e.max()
        e_exp = torch.exp(edge_e - e_max)
        e_exp_sum = torch.zeros(N, 1, device=device)
        e_exp_sum.scatter_add_(
            dim=0,
            index=source.unsqueeze(1),
            src=e_exp
        )
        e_exp_sum += 1e-10
        e_softmax = e_exp / e_exp_sum[source]

        return e_softmax

    @staticmethod
    def sparse_matmul(edges, attention, Wh):
        """Matmul for sparse adjacency matrix"""

        source, target = edges
        h_prime = torch.zeros_like(Wh)
        h_prime.scatter_add_(
            dim=0,
            index=source.expand(Wh.size(1), -1).t(),
            src=attention * Wh[target]
        )

        return h_prime

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
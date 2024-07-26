import math

import torch
import torch.nn as nn


class GraphSelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_attention_heads=4,
                 attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2):
        super(GraphSelfAttention, self).__init__()

        self.attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        self.AE = nn.Linear(input_size, hidden_size)

        self.query_head = nn.ModuleList()
        self.key_head = nn.ModuleList()
        self.value_head = nn.ModuleList()
        for i in range(num_attention_heads):
            self.query_head.append(nn.Linear(hidden_size, hidden_size))
            self.key_head.append(nn.Linear(hidden_size, hidden_size))
            self.value_head.append(nn.Linear(hidden_size, hidden_size))

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.ReLU = nn.ReLU()
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor, masking, is_drop=True):

        input_tensor = self.AE(input_tensor)

        outputs = []
        for i in range(self.attention_heads):
            query = self.query_head[i]
            key = self.key_head[i]
            value = self.value_head[i]

            query_layer = query(input_tensor)
            key_layer = key(input_tensor)
            value_layer = value(input_tensor)

            attention_scores = torch.matmul(query_layer, key_layer.transpose(
                -1, -2))

            attention_scores = attention_scores / math.sqrt(
                self.hidden_size)

            attention_scores[masking==False] = -1e38

            attention_scores = nn.Softmax(dim=-1)(attention_scores)


            if is_drop:
                attention_scores = self.attn_dropout(attention_scores)
            context_layer = torch.matmul(attention_scores, value_layer)

            outputs.append(context_layer)

        output = torch.mean(torch.stack(outputs), 0)

        hidden_states = self.dense(output)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    def save_model(self, file_name):
        torch.save(self.cpu().state_dict(), file_name)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

class CLNetwork(nn.Module):
    def __init__(self, input_size, middle_size, hidden_size, cluster_num, adjs4GAT_mask,
                 att_heads=4, attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2):
        super(CLNetwork, self).__init__()

        self.encoder1 = GraphSelfAttention(input_size, middle_size, att_heads)
        self.encoder2 = GraphSelfAttention(middle_size, hidden_size, att_heads)
        self.masking = adjs4GAT_mask

        self.instance_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.cluster_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder2(self.encoder1(x_i, self.masking), self.masking)
        h_j = self.encoder2(self.encoder1(x_j, self.masking), self.masking)

        z_i = self.instance_projector(h_i)
        z_j = self.instance_projector(h_j)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.encoder2(self.encoder1(x, self.masking), self.masking)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c, h

    def save_model(self, file_name):
        torch.save(self.cpu().state_dict(), file_name)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):

        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        # ipdb.set_trace()
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

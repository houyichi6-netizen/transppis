import torch
import torch.nn.functional as F
import time
from torch import nn, optim
from utils import try_gpu, loss_function
import numpy as np
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GlobalAttentionPooling



EPS = 1e-15
LEARNING_RATE = 1E-2
WEIGHT_DECAY = 5E-3


class VGAE(nn.Module):
    """
    The self-supervised module of prestudy
    """

    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(VGAE, self).__init__()
        self.gc1 = GraphConv(input_feat_dim, hidden_dim1, activation=F.relu)  # F.relu
        self.gc2 = GraphConv(hidden_dim1, hidden_dim2)  # lambda x: x
        self.gc3 = GraphConv(hidden_dim1, hidden_dim2)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, adj):
        hidden1 = self.dropout(self.gc1(x, adj))
        mu = self.dropout(self.gc2(hidden1, adj))  # Mean of the latent distribution
        logstd = self.dropout(self.gc3(hidden1, adj))
        return mu, logstd

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd.clamp(min=-10, max=10))
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        adj_hat = torch.mm(z, z.t())
        return adj_hat

    def forward(self, x, adj, sigmoid: bool = True):
        mu, logstd = self.encode(x, adj)
        z = self.reparameterize(mu, logstd)
        return (torch.sigmoid(self.decode(z)), z, mu, logstd) if sigmoid else (self.decode(z), z, mu, logstd)


class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0.):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        input = F.dropout(x, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class transPPI(nn.Module):
    """ model  not share GNN weight """

    def __init__(self, hidden_dim=512):
        super(transPPI, self).__init__()

        self.res_embedding = nn.Embedding(25, 128)
        self.struct_linear = nn.Linear(61, 384)
        self.seq_linear = nn.Linear(1280, 512)

        self.struct_gc1 = GraphConv(hidden_dim, hidden_dim, activation=F.leaky_relu)
        self.struct_gc2 = GraphConv(hidden_dim, 256, activation=F.leaky_relu)
        # self.struct_gc2 = GraphConv(hidden_dim, 256, )
        # self.struct_gc1 = GraphConv(hidden_dim, hidden_dim)

        self.seq_gc1 = GraphConv(hidden_dim, hidden_dim, activation=F.leaky_relu)
        self.seq_gc2 = GraphConv(hidden_dim, 256, activation=F.leaky_relu)

        # self.seq_gc1 = GraphConv(hidden_dim, hidden_dim, )
        # self.seq_gc2 = GraphConv(hidden_dim, 256, )

        self.str_gap = GlobalAttentionPooling(nn.Linear(256, 1))
        self.seq_gap = GlobalAttentionPooling(nn.Linear(256, 1))

        self.linear2 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(p=0.0)
        self.final_layer = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(),
                                         nn.Linear(128, 2))


        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.seq_linear.weight)
        nn.init.xavier_uniform_(self.struct_linear.weight)
        nn.init.xavier_uniform_(self.struct_gc1.weight)
        nn.init.xavier_uniform_(self.struct_gc2.weight)
        nn.init.xavier_uniform_(self.seq_gc1.weight)
        nn.init.xavier_uniform_(self.seq_gc2.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.final_layer[0].weight)
        nn.init.xavier_uniform_(self.final_layer[2].weight)

    def forward(self, sturct_G, seq_graph, kmerseq):
        seq_x = seq_graph.ndata['attr']
        min_values = torch.min(seq_x, dim=0).values
        max_values = torch.max(seq_x, dim=0).values

        seq_x = (seq_x - min_values) / (max_values - min_values)
        res_array = sturct_G.ndata['res_array']

        struct_x = sturct_G.ndata['node_features']
        struct_x1 = self.res_embedding(res_array)
        struct_x2 = self.struct_linear(struct_x)
        struct_x = torch.cat((struct_x1, struct_x2), dim=-1)
        min_values = torch.min(struct_x, dim=0).values
        max_values = torch.max(struct_x, dim=0).values
 
        struct_x = (struct_x - min_values) / (max_values - min_values)

        seq_x = self.seq_linear(seq_x)

        struct_x = self.dropout(self.struct_gc1(sturct_G, struct_x))
        struct_x = self.dropout(self.struct_gc2(sturct_G, struct_x))
        _, struct_attention = self.str_gap(sturct_G, struct_x, get_attention=True)

        seq_x = self.dropout(self.seq_gc1(seq_graph, seq_x, edge_weight=seq_graph.edata['weight']))
        seq_x = self.dropout(self.seq_gc2(seq_graph, seq_x, edge_weight=seq_graph.edata['weight']))
        _, seq_attention = self.seq_gap(seq_graph, seq_x, get_attention=True)
        # 在模型外面需要对seq_attention 进行处理
        # 要确保序列节点和结构节点数量关系是一致的
        seq_x, seq_attention = self.recover_kmer_vectors_batch(seq_x.detach().cpu().numpy(),
                                                               seq_attention.detach().cpu().numpy(),
                                                               kmerseq, k=3)
        struct_x = self.linear2(struct_x)
        seq_x = self.linear2(seq_x)

        x = torch.mul(struct_x, seq_x)
        # x = self.dropout(self.linear2(x))
        final_x = self.final_layer(x)

        return final_x, struct_attention, seq_attention / seq_attention.sum(), x

    def recover_kmer_vectors_batch(self, seq_x, seq_attention, kmerseq, k=3):

        kmerseq = kmerseq[0].astype(int)
        kmer_x_new = seq_x[kmerseq]  # k-mer 
        new_seq_attention = seq_attention[kmerseq]  # k-mer attention
        num_kmers, feature_dim_x = kmer_x_new.shape
        _, feature_dim_attention = new_seq_attention.shape

        seq_len = len(kmerseq) + k - 1  
        residu_x = np.zeros((seq_len, feature_dim_x))  
        residu_x_attention = np.zeros((seq_len, feature_dim_attention))  
        overlap_count = np.zeros(seq_len) 


        for i in range(len(kmer_x_new)):
            residu_x[i:i + k] += kmer_x_new[i]  
            residu_x_attention[i:i + k] += new_seq_attention[i] 
            overlap_count[i:i + k] += 1 

        residu_x /= overlap_count[:, None]  
        residu_x_attention /= overlap_count[:, None]  

        return torch.tensor(residu_x, dtype=torch.float32).cuda(), torch.tensor(residu_x_attention,
                                                                                dtype=torch.float32).cuda()


class transPPI_share(nn.Module):
    """ model share GNN weight """

    def __init__(self, input_dim, hidden_dim):
        super(transPPI_share, self).__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)

        self.gc1 = GCNLayer(hidden_dim, 256, 0.1)
        self.gc2 = GCNLayer(hidden_dim, 256, 0.1)
        self.gap = Global_attention_Pooling(256)

        self.linear2 = nn.Linear(256 * 2, 256)
        self.dropout = nn.Dropout(p=0.4)
        self.final_layer = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                                         nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, struct_x, seq_x, struct_edge, seq_edge):
        struct_x = F.relu(self.linear(struct_x))
        seq_x = F.relu(self.linear(seq_x))

        struct_x = F.relu(self.gc1(struct_x, struct_edge))
        struct_x = F.relu(self.gc2(struct_x, struct_edge))
        struct_attention = self.gap(struct_x)

        seq_x = F.relu(self.gc1(seq_x, seq_edge))
        seq_x = F.relu(self.gc2(seq_x, seq_edge))
        seq_attention = self.gap(seq_x)

        residu_x = recover_kmer_vectors_batch(seq_x, struct_x.shape[1], k=3)
        residu_x = torch.tensor(residu_x)
        x = torch.cat((struct_x, residu_x), dim=1)
        x = self.dropout(self.linear2(x))
        return self.final_layer(x), struct_attention, seq_attention, x

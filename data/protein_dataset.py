import pickle
import math
import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import scipy.sparse as sp
import warnings
from dgl.nn.pytorch import EdgeWeightNorm

warnings.filterwarnings("ignore")

# Feature Path
Feature_Path = "./data/Feature/"
# Seed
SEED = 2020
K_MER_SIZE = 3
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

# model parameters
ADD_NODEFEATS = 'all'  # all/atom_feats/psepose_embedding/no
USE_EFEATS = True  # True/False
MAP_CUTOFF = 14
DIST_NORM = 15

# INPUT_DIM
if ADD_NODEFEATS == 'all':  # add atom features and psepose embedding
    INPUT_DIM = 54 + 7 + 1
elif ADD_NODEFEATS == 'atom_feats':  # only add atom features
    INPUT_DIM = 54 + 7
elif ADD_NODEFEATS == 'psepose_embedding':  # only add psepose embedding
    INPUT_DIM = 54 + 1
elif ADD_NODEFEATS == 'no':
    INPUT_DIM = 54
DROPOUT = 0.1
ALPHA = 0.7
LAMBDA = 1.5

WEIGHT_DECAY = 0
NUM_CLASSES = 2  # [not bind, bind]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# sturct embedding：残基物化性质7维、dssp二级结构14维 PSSM hmm矩阵一共40维 --》总计61维                          残基one-hot编码1维，需使用nn.embedding处理
def embedding(sequence_name):
    pssm_feature = np.load(Feature_Path + "pssm/" + sequence_name + '.npy')
    hmm_feature = np.load(Feature_Path + "hmm/" + sequence_name + '.npy')
    seq_embedding = np.concatenate([pssm_feature, hmm_feature], axis=1)
    return seq_embedding.astype(np.float32)


def get_dssp_features(sequence_name):
    dssp_feature = np.load(Feature_Path + "dssp/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)


def get_res_atom_features(sequence_name):
    res_atom_feature = np.load(Feature_Path + "resAF/" + sequence_name + '.npy')
    return res_atom_feature.astype(np.float32)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def cal_edges(sequence_name, radius=MAP_CUTOFF):  # to get the index of the edges
    dist_matrix = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dist_matrix >= 0) * (dist_matrix <= radius))
    adjacency_matrix = mask.astype(np.int32)
    radius_index_list = np.where(adjacency_matrix == 1)
    radius_index_list = [list(nodes) for nodes in radius_index_list]
    return radius_index_list


def load_graph(sequence_name):
    dismap = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    adjacency_matrix = mask.astype(np.int32)
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return norm_matrix


def graph_collate(samples):
    sequence_name, sequence, label, sturct_G, seq_grap, seqid = map(list,
                                                                    zip(*samples))
    # label = torch.Tensor(label)
    sturct_G_batch = dgl.batch(sturct_G)
    seq_grap_batch = dgl.batch(seq_grap)

    return sequence_name, sequence, label, sturct_G_batch, seq_grap_batch, seqid


class ProDataset(Dataset):
    def __init__(self, dataframe, radius=MAP_CUTOFF, dist=DIST_NORM, ):
        # train_dataframe = {"ID": IDs, "sequence": sequences, "seq_array":seq_array, "label": labels}
        # 还有一个 dataframe 其中是 IDs 和 sequence_esm

        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.res_array = dataframe['res_array'].values
        self.labels = dataframe['label'].values
        self.esm_feature = dataframe['esm_feature'].values
        self.radius = radius
        self.dist = dist

    def __getitem__(self, index):

        kmers2id = {}
        id2kmers = []
        esm_feature = self.esm_feature[index]  # 序列图的特征
        sequence = self.sequences[index]
        sequence_name = self.names[index]
        res_array = torch.from_numpy(self.res_array[index])  # 结构图的残基嵌入 narray
        label = np.array(self.labels[index])

        nodes_num = len(sequence)
        sequence_embedding = embedding(sequence_name)  # 插入蛋白质残基的pssm和hmm特征
        structural_features = get_dssp_features(sequence_name)  # 残基二级结构特征
        str_node_features = np.concatenate([sequence_embedding, structural_features], axis=1)
        str_node_features = torch.from_numpy(str_node_features)
        if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'atom_feats':
            res_atom_features = get_res_atom_features(sequence_name)  # 插入物化性质
            res_atom_features = torch.from_numpy(res_atom_features)
            str_node_features = torch.cat([str_node_features, res_atom_features], dim=-1)

        radius_index_list = cal_edges(sequence_name, MAP_CUTOFF)
        sturct_G = dgl.DGLGraph()
        sturct_G.add_nodes(nodes_num)
        self.add_edges_custom(sturct_G,
                              radius_index_list, )
        str_adj_matrix = load_graph(sequence_name)

        sturct_G.ndata["node_features"] = str_node_features
        sturct_G.ndata["res_array"] = res_array

        # 构造德布鲁因图 并构造esm嵌入向量
        # 根据序列信息得到kmer_list
        kmersCnt = 0
        kmer_list, kmer_vectors = self.construct_kmer_vectors(sequence, esm_feature, K_MER_SIZE,
                                                              method="mean")
        for mer in kmer_list:
            if mer not in kmers2id:
                kmers2id[mer] = kmersCnt  # self.kmers2id kmer映射到id的字典
                id2kmers.append(mer)  # id2kmers 列表 其中按照id顺序存储kmer
                kmersCnt += 1
        idseq = np.array([kmers2id[kmer] for kmer in kmer_list], dtype=object)  # 得到了序列的id列表

        seq_adj_matrix = np.zeros((kmersCnt, kmersCnt))
        # print(f"sequence_name--------------------------: {sequence_name}\n"
        #       f"seq_adj_matrix: {seq_adj_matrix.shape}\n"
        #       f"idseq: {idseq.size}\n"
        #       f"str nodes_num: {nodes_num}\n"
        #       f"seq nodes_num: {kmersCnt}")

        for i in range(idseq.size - 1):
            seq_adj_matrix[idseq[i]][idseq[i + 1]] += 1
        tmp_coo = sp.coo_matrix(seq_adj_matrix)
        edge_attr = tmp_coo.data
        edge_index = np.vstack((tmp_coo.row, tmp_coo.col))
        u, v = torch.tensor(edge_index[0], dtype=torch.int64), torch.tensor(edge_index[1], dtype=torch.int64)
        weight = torch.FloatTensor(edge_attr)
        seq_graph = dgl.graph((u, v), num_nodes=kmersCnt)
        norm = EdgeWeightNorm(norm='both')
        norm_weight = norm(seq_graph, weight)
        seq_graph.edata['weight'] = norm_weight
        seq_graph.ndata['attr'] = torch.tensor(kmer_vectors, dtype=torch.float32)
        seq_graph = dgl.add_self_loop(seq_graph, ['weight'], 1.0)

        return sequence_name, sequence, label, sturct_G, seq_graph, idseq

    def __len__(self):
        return len(self.labels)

    def add_edges_custom(self, G, radius_index_list):
        src, dst = radius_index_list[1], radius_index_list[0]
        if len(src) != len(dst):
            print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
            raise Exception
        G.add_edges(src, dst)
        G.add_edges(dst, src)

    def construct_kmer_vectors(self, sequence, vectors, k, method="mean"):
        """
        Construct k-mer sequences and their corresponding vector representations.

        Args:
            sequence (str): Protein sequence (e.g., "ACDEFGHIKLMNPQRSTVWY").
            vectors (np.ndarray): Protein vector of shape (n, m), where n is the number of residues, m is the feature dimension.
            k (int): Length of each k-mer.
            method (str): How to represent k-mer vectors, "mean" or "concat".

        Returns:
            kmer_list (list of str): List of k-mer sequences.
            kmer_vectors (np.ndarray): K-mer vectors, shape depends on `method`.
        """
        vectors = vectors.tolist()[0]
        if len(sequence) != vectors.shape[0]:
            raise ValueError("Sequence length and vector length must match.")
        if len(sequence) < k:
            raise ValueError("Sequence length must be greater than or equal to k.")

        # Generate k-mer sequences
        kmer_list = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

        # Generate k-mer vectors
        kmer_dict = {}
        for i in range(len(kmer_list)):
            kmer = kmer_list[i]
            if kmer not in kmer_dict:
                kmer_dict[kmer] = vectors[i:i + k].mean(axis=0)
            else:
                kmer_dict[kmer] += vectors[i:i + k].mean(axis=0)
                kmer_dict[kmer] /= 2

        kmer_vectors = np.array(list(kmer_dict.values()))

        return kmer_list, kmer_vectors

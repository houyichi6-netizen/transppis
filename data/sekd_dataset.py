import torch
import numpy as np
from torch.utils.data import Dataset
from data.pdb_adj import StructureDataParser
import warnings

warnings.filterwarnings("ignore")
#  从pdb中构造图
device = f"cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)




def esmAF_feature(seq_name):
    esm_features = np.load('data/ESM/' + seq_name + '.npy')
    # af_features = np.load('../PPIS_GVP/Features/AF/'+seq_name+'.npy')
    af_features = [0]
    return esm_features, af_features


def cal_edges(seq_name):
    # 加载距离矩阵文件，该文件包含了序列中各个元素之间的距离信息
    file_path = 'data/N_new_PDB/' + seq_name + '.pdb'
    struct_parser = StructureDataParser(file_path, seq_name, 'pdb')
    coords = struct_parser.get_residue_atoms_coords()
    coords_CA = torch.as_tensor(coords['CA'], dtype=torch.float32)
    dist_matrix = struct_parser.generate_atom_distance_map('CA')
    # dist_matrix = np.load('Features/Dist_CA/' + seq_name+ ".npy")
    # 创建一个掩码矩阵，用于筛选距离在指定半径范围内的元素之间的连接关系
    mask = ((dist_matrix >= 0) * (dist_matrix <= 14))
    # 将掩码矩阵转换为整数类型，以便后续处理
    adjacency_matrix = mask.astype(np.int_)
    np.fill_diagonal(adjacency_matrix, 0)  # 取消自连接
    # 获取半径范围内元素的索引列表
    radius_index_list = np.where(adjacency_matrix == 1)
    # 将索引列表转换为节点列表的形式
    radius_index_list = [list(nodes) for nodes in radius_index_list]
    radius_index_list = torch.from_numpy(np.array(radius_index_list).astype(dtype="int64"))
    # 返回半径范围内元素的索引列表
    return radius_index_list, coords_CA


class myDatasets(Dataset):
    def __init__(self, dataframe):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.topk = 40
        self.num_rbf = 16
        self.map = 14

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])

        esm_fs, af_fs = esmAF_feature(sequence_name)
        esm_fs = torch.from_numpy(esm_fs).type(torch.FloatTensor)
        edge_index, CA_coords = cal_edges(sequence_name)

        return sequence_name, sequence, label, esm_fs, af_fs, edge_index, CA_coords

    def __len__(self):
        return len(self.labels)


class myDatasets_single(Dataset):
    def __init__(self, dataframe):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        # self.labels = dataframe['label'].values
        self.topk = 40
        self.num_rbf = 16
        self.map = 14

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        # label = np.array(self.labels[index])

        esm_fs, af_fs = esmAF_feature(sequence_name)
        esm_fs = torch.from_numpy(esm_fs).type(torch.FloatTensor)
        edge_index, CA_coords = cal_edges(sequence_name)

        return sequence_name, sequence, esm_fs, af_fs, edge_index, CA_coords

    def __len__(self):
        return len(self.sequences)

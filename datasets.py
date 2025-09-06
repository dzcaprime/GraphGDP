import torch
import json
import os
import numpy as np
import os.path as osp
import pandas as pd
import pickle as pk
from torch.utils.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import from_networkx, degree, to_networkx

import data


class NRISpringsDataset(Dataset):
    def __init__(self, root, suffix="_springs", n_balls=10, pool="mean"):
        """
        NRI 弹簧数据集读取器。

        参数
        ----
        root : str
            数据根目录，包含 feat{suffix}.npy 与 edges{suffix}.npy。
        suffix : str
            数据后缀（默认 "_springs"），会附加球数。
        n_balls : int
            小球数量 N（节点数）。
        pool : str
            预留参数（聚合方式），不在此处使用。
        """
        super().__init__()
        suffix += str(n_balls)

        def _load(name):
            path = os.path.join(root, f"{name}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            return np.load(path)

        feat = _load(f"feat{suffix}")  # [S, N, T, C]
        edges = _load(f"edges{suffix}")  # [S, N, N]
        self.S, self.N, self.T, self.C = feat.shape
        self.edges = edges.astype(np.float32)
        self.feat = feat
        self.pool = pool
        self.L = feat.shape[2]

    class _Subset(Dataset):
        """
        父数据集的轻量子集视图。只保存索引，复用父数据集存取逻辑。
        """
        def __init__(self, parent: "NRISpringsDataset", indices):
            self.parent = parent
            # 规范化索引为 Python int 列表，避免 Tensor/ndarray 造成歧义
            if isinstance(indices, torch.Tensor):
                indices = indices.cpu().numpy().tolist()
            elif isinstance(indices, np.ndarray):
                indices = indices.tolist()
            elif isinstance(indices, range):
                indices = list(indices)
            self.indices = [int(i) for i in indices]

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            # 委托父数据集的整数索引逻辑
            return self.parent[int(self.indices[int(i)])]

        def n_node_pmf(self):
            # N 为定值，PMF 与父数据集一致
            return self.parent.n_node_pmf()

    def __len__(self):
        return self.S

    def __getitem__(self, idx):
        """
        支持：
        - 整数索引：返回单样本字典
        - 切片/list/ndarray/tensor：返回子数据集视图，兼容 DataLoader
        """
        if isinstance(idx, slice):
            return NRISpringsDataset._Subset(self, range(*idx.indices(self.S)))
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            return NRISpringsDataset._Subset(self, idx)

        i = int(idx)
        A = self.edges[i]  # [N,N]
        A = (A + A.T) / 2.0
        A = np.clip(A, 0, 1)
        X = self.feat[i]  # [N,L,C]
        ts = X  # [N,L,C]
        adj = torch.from_numpy(A[None, ...])  # [1,N,N]
        # 将对角置零，避免自环；保持上下三角可用于模型前向
        mask = torch.ones_like(adj)
        eye = torch.eye(self.N, dtype=adj.dtype, device=adj.device)
        mask = mask - eye[None, ...]
        ts = torch.from_numpy(ts.astype(np.float32))  # [N,L,C]
        return {"adj": adj, "mask": mask, "ts": ts, "n": self.N}

    def n_node_pmf(self):
        pmf = np.zeros(self.N + 1, dtype=np.float32)
        pmf[self.N] = 1.0
        return pmf


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""

    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""

    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


def networkx_graphs(dataset):
    return [to_networkx(dataset[i], to_undirected=True, remove_self_loops=True) for i in range(len(dataset))]


class StructureDataset(InMemoryDataset):
    def __init__(self, root, dataset_name, transform=None, pre_transform=None, pre_filter=None):

        self.dataset_name = dataset_name

        super(StructureDataset, self).__init__(root, transform, pre_transform, pre_filter)

        if not os.path.exists(self.raw_paths[0]):
            raise ValueError("Without raw files.")
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()

    @property
    def raw_file_names(self):
        return [self.dataset_name + ".pkl"]

    @property
    def processed_file_names(self):
        return [self.dataset_name + ".pt"]

    @property
    def num_node_features(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1)

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ""
        return f"{self.dataset_name}({arg_repr})"

    def process(self):
        # Read data into 'Data' list
        input_path = self.raw_paths[0]
        with open(input_path, "rb") as f:
            graphs_nx = pk.load(f)
        data_list = [from_networkx(G) for G in graphs_nx]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    @torch.no_grad()
    def max_degree(self):
        data_list = [self.get(i) for i in range(len(self))]

        def graph_max_degree(g_data):
            return max(degree(g_data.edge_index[1], num_nodes=g_data.num_nodes))

        degree_list = [graph_max_degree(data) for data in data_list]
        return int(max(degree_list).item())

    def n_node_pmf(self):
        node_list = [self.get(i).num_nodes for i in range(len(self))]
        n_node_pmf = np.bincount(node_list)
        n_node_pmf = n_node_pmf / n_node_pmf.sum()
        return n_node_pmf


def get_dataset(config):
    """
    创建训练/评估/测试数据集与节点数分布。
    保持原有切分语义不变，避免破坏用户空间。
    """
    # define data transforms
    transform = T.Compose(
        [
            # T.ToDense(config.data.max_node),
            T.ToDevice(config.device)
        ]
    )

    # 在 get_dataset 中加分支：
    if getattr(config.data, "temporal", False) or config.data.name.lower() == "nri_springs10":
        dataset = NRISpringsDataset(config.data.root, "_springs", n_balls=config.data.max_node)
        num_train = int(len(dataset) * config.data.split_ratio)
        num_test = len(dataset) - num_train
        train_ds = dataset[:num_train]  # 现已返回子数据集视图
        eval_ds = dataset[:num_test]
        test_ds = dataset[num_train:]
        return train_ds, eval_ds, test_ds, train_ds.n_node_pmf()

    # Build up data iterators
    dataset = StructureDataset(config.data.root, config.data.name, transform=transform)
    num_train = int(len(dataset) * config.data.split_ratio)
    num_test = len(dataset) - num_train
    train_dataset = dataset[:num_train]
    eval_dataset = dataset[:num_test]
    test_dataset = dataset[num_train:]

    n_node_pmf = train_dataset.n_node_pmf()

    return train_dataset, eval_dataset, test_dataset, n_node_pmf

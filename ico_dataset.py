import os.path as osp
from typing import Callable, Optional

import scipy
import torch

import numpy as np

from torch_geometric.data import InMemoryDataset, download_url, Data
from icosphere import icosphere
from itertools import combinations


class IcoDataset(InMemoryDataset):
    def __init__(self, root: str = ".", transform=None, pre_transform=None, pre_filter=None, nu: int = 2):

        self.nu = nu

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return ['raw_data']

    @property
    def processed_file_names(self):
        return ['data.pt']

    # in case we want to download the data from somewhere eventually

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
    #     ...

    def process(self):
        # data = read_npz(
        #     self.raw_paths[0])  # or another reader, depending on the format in which our icosahedral mesh is stored

        data = IcoSphereWithFeatures(nu=self.nu)

        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])

        torch.save((data, slices), self.processed_paths[0])


class IcoSphereWithFeatures:
    def __init__(self, v_features=None, nu: int = 2, response=None):
        self.vertex_coordinates, self.faces = icosphere(nu=nu)

        self.ncell = len(self.faces)
        self.num_nodes = self.vertex_coordinates.shape[0]

        if v_features is not None:
            self.x = v_features  # a feature matrix of the form [n_features, feature_dim]

        # otherwise populate with random features
        else:
            self.x = torch.rand((self.num_nodes, 4), dtype=torch.float32)  # a feature matrix of the form [n_features, feature_dim]

        self.num_node_features = self.x.shape[1]

        if response is not None:
            self.y = response
        else:
            self.y = torch.randint(size=(self.num_nodes,), low=0, high=4)

        # random train mask using ~0.8 share of the nodes as training data
        self.train_mask = torch.bernoulli(torch.full(size=(self.num_nodes,),
                                                     fill_value=0.9)).to(torch.bool)
        self.half_mask = torch.bernoulli(torch.full(size=(self.num_nodes,), fill_value=0.5)).to(torch.bool)
        self.val_mask = self.half_mask * ~self.train_mask
        self.test_mask = ~self.half_mask * ~self.train_mask


        # adjacency matrix
        self.edges_by_vertex_indices = []
        for i, j in combinations(range(len(self.faces)), 2):
            face_intersection = set([i for i in set.intersection(set(self.faces[i]), set(self.faces[j])) if i!=0])
            if len(face_intersection) == 2:
                   self.edges_by_vertex_indices.append(list(face_intersection))
        self.edge_index = torch.Tensor(self.edges_by_vertex_indices).to(torch.int64).transpose(0,1)
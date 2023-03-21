import os.path as osp
from typing import Callable, Optional

import scipy
import torch

import numpy as np

from torch_geometric.data import InMemoryDataset, download_url, Data
from icosphere import icosphere
from itertools import combinations


class IcoDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, nu: int = 2):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.nu = nu

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
        self.nvert = self.vertex_coordinates.shape[0]

        if v_features is not None:
            self.v_features = v_features  # a feature matrix of the form [n_features, feature_dim]

        # otherwise populate with random features
        else:
            self.v_features = torch.rand((self.nvert, 4), dtype=torch.float32)  # a feature matrix of the form [n_features, feature_dim]

        if response is not None:
            self.y = response
        else:
            self.y = torch.randint(size=(self.nvert, 1), low=0, high=4)

        # adjacency matrix
        self.edges_by_vertex_indices = []
        for i, j in combinations(range(len(self.faces)), 2):
            face_intersection = set([i for i in set.intersection(set(self.faces[i]), set(self.faces[j])) if i!=0])
            if len(face_intersection) == 2:
                   self.edges_by_vertex_indices.append(list(face_intersection))
        self.edge_index = np.array(self.edges_by_vertex_indices)
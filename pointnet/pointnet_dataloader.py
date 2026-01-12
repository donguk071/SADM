import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
import glob
import torch

import h5py

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoints, num_category=10, split='train', process_data=True):
        self.root = root
        self.npoints = npoints
        self.process_data = process_data
        self.num_category = num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
            
        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
                    point_set = farthest_point_sample(point_set, self.npoints)

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            point_set = farthest_point_sample(point_set, self.npoints)
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        point_set = point_set[:, 0:3].T

        return torch.from_numpy(point_set), torch.tensor(label[0])

    def __getitem__(self, index):
        return self._get_item(index)
    
def load_h5(h5_filename):
    with h5py.File(h5_filename, 'r', locking=False) as f:
        data = f['data'][:]
        label = f['label'][:]
    return data, label
    
class ScanObjectNNLoader(Dataset):
    def __init__(self, root, split='train'):

        if split == 'train':
            self.dataset, self.label = load_h5(root + "/sampled_train.h5")
        else:
            self.dataset, self.label = load_h5(root + "/sampled_test.h5")

    def __len__(self):
        return len(self.label)

    def _get_item(self, index):
        point_set = self.dataset[index]
        label = self.label[index]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        point_set = point_set[:, 0:3].T 

        return torch.from_numpy(point_set), torch.tensor(label)

    def __getitem__(self, index):
        return self._get_item(index)

class ShapeNetLoader(Dataset):
    def __init__(self, root, split='train'):

        self.label_list = []

        if split == 'train':
            self.data_list = glob.glob(root + '/*/train/*.npy')
            for d in tqdm(self.data_list):
                self.label_list.append(d.split('/')[-3])
        else:
            self.data_list = glob.glob(root + '/*/test/*.npy')
            for d in tqdm(self.data_list):
                self.label_list.append(d.split('/')[-3])

    def __len__(self):
        return len(self.data_list)

    def _get_item(self, index):
        point_path = self.data_list[index]
        label = self.label_list[index]

        point = np.load(point_path)

        return torch.from_numpy(point).T, torch.tensor(int(label))

    def __getitem__(self, index):
        return self._get_item(index)

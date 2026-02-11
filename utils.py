import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pointnet import pointnet_dataloader, pointnet_model
from tqdm import tqdm
import logging 
import random

class CustomSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def get_dataset(args, dataset, npoints = 1024):
    if dataset == 'CrossDataset': 
        # define shared indices
        MN40_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 25, 30, 33]
        SN_index = [0, 4, 5, 6, 8, 9, 10, 16, 18, 26, 20, 30, 31, 38, 47, 49]
        num_classes = len(MN40_index)
        coord_dim = 3

        mn40_label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(MN40_index)}
        sn_label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(SN_index)}

        raw_mn40_train = pointnet_dataloader.ModelNetDataLoader(
            root='/root/dataset/ModelNet40', split='train', npoints=npoints, num_category=40)
        raw_mn40_test = pointnet_dataloader.ModelNetDataLoader(
            root='/root/dataset/ModelNet40', split='test', npoints=npoints, num_category=40)
        raw_sn_train = pointnet_dataloader.ShapeNetLoader(
            "/root/dataset/ShapeNetv2/PointCloud", split='train')
        raw_sn_test = pointnet_dataloader.ShapeNetLoader(
            "/root/dataset/ShapeNetv2/PointCloud", split='test')


        def filter_and_remap(original_dataset, index_list, label_map):
            label_to_points = {lbl: [] for lbl in index_list}

            for pts, lbl in original_dataset:
                lbl = int(lbl)
                if lbl in index_list:
                    label_to_points[lbl].append(pts)

            filtered_data = []
            filtered_labels = []

            for lbl in index_list:
                for pts in label_to_points[lbl]:
                    filtered_data.append(pts)
                    filtered_labels.append(label_map[lbl])  

            return filtered_data, filtered_labels


        mn40_train_data, mn40_train_labels = filter_and_remap(raw_mn40_train, MN40_index, mn40_label_map)
        mn40_test_data, mn40_test_labels = filter_and_remap(raw_mn40_test, MN40_index, mn40_label_map)
        sn_train_data, sn_train_labels = filter_and_remap(raw_sn_train, SN_index, sn_label_map)
        sn_test_data, sn_test_labels = filter_and_remap(raw_sn_test, SN_index, sn_label_map)

        # pick train/test based on direction
        if args.cross_dataset == "MN40_SN":
            train_data, train_labels = mn40_train_data, mn40_train_labels
            test_data, test_labels = sn_test_data, sn_test_labels
        elif args.cross_dataset == "SN_MN40":
            train_data, train_labels = sn_train_data, sn_train_labels
            test_data, test_labels = mn40_test_data, mn40_test_labels
        elif args.cross_dataset == "MN40_MN40":
            train_data, train_labels = mn40_train_data, mn40_train_labels
            test_data, test_labels = mn40_test_data, mn40_test_labels
        elif args.cross_dataset == "SN_SN":
            train_data, train_labels = sn_train_data, sn_train_labels
            test_data, test_labels = sn_test_data, sn_test_labels

        # wrap into Dataset & DataLoader
        train_dataset = CustomSubsetDataset(train_data, train_labels)
        test_dataset = CustomSubsetDataset(test_data, test_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_real, shuffle=True)
        bs = 16 if args.eval_mode == 'CrossArchi' else 128
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

        return npoints, coord_dim, num_classes, train_dataset, train_loader, test_loader
  
        
    elif dataset == 'MODELNET40': 
        num_classes = 40
        coord_dim = 3
        npoints = npoints
        train_dataset = pointnet_dataloader.ModelNetDataLoader(
           root = '/root/dataset/ModelNet40',
           split='train',
           npoints=npoints,
           num_category=num_classes,
       )
        test_dataset = pointnet_dataloader.ModelNetDataLoader(
           root = '/root/dataset/ModelNet40',
           split='test',
           npoints=npoints,
           num_category=num_classes,
       )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_real, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)    
        return npoints,coord_dim, num_classes, train_dataset, train_loader, test_loader 
    
    elif dataset == 'MODELNET10':
        num_classes = 10
        coord_dim = 3
        npoints = npoints
        train_dataset = pointnet_dataloader.ModelNetDataLoader(
           root = '/root/dataset/ModelNet40',
           split='train',
           npoints=npoints,
           num_category=num_classes,
       )
        test_dataset = pointnet_dataloader.ModelNetDataLoader(
           root = '/root/dataset/ModelNet40',
           split='test',
           npoints=npoints,
           num_category=num_classes,
       )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_real, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)        
        return npoints,coord_dim, num_classes, train_dataset, train_loader, test_loader 
    
    elif dataset == 'scanobjectnn':
        num_classes = 15
        coord_dim = 3
        npoints = npoints
        train_dataset = pointnet_dataloader.ScanObjectNNLoader("/root/dataset/ScanObjectNN/main_split_nobg", split='train')
        test_dataset = pointnet_dataloader.ScanObjectNNLoader("/root/dataset/ScanObjectNN/main_split_nobg", split='test')

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_real, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
        return npoints, coord_dim, num_classes, train_dataset, train_loader, test_loader
    
    elif dataset == 'shapenet':
        num_classes = 55
        coord_dim = 3
        npoints = npoints
        train_dataset = pointnet_dataloader.ShapeNetLoader("/root/dataset/ShapeNetv2/PointCloud", split='train')
        test_dataset = pointnet_dataloader.ShapeNetLoader("/root/dataset/ShapeNetv2/PointCloud", split='test')

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_real, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
        return npoints, coord_dim, num_classes, train_dataset, train_loader, test_loader
    
    else:
        exit('unknown dataset: %s'%dataset)

    print(dataset)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader

class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def get_network(model, channel, num_classes,feature_transform = False):
    if model == "PointNet": 
        net = pointnet_model.PointNetCls(k=num_classes, feature_transform=feature_transform)
    
    # elif model == "PointNetPlusPlus" :
    #     net = pointnetpp_model.PointNet2Cls(num_classes)
        
    # elif model == "DGCNN" :
    #     net = dgcnn.DGCNN(num_classes)
   
    # elif model == "PointConvDensityClsSsg":
    #     net = pointconv.PointConvDensityClsSsg(num_classes)    

    # elif model == "PointTransformerCls":
    #     net = point_transformer.PointTransformerCls(num_classes)    

    else:
        net = None
        exit('unknown model: %s'%model)

    gpu_num = torch.cuda.device_count()

    if gpu_num>0:
        device = 'cuda'
        if gpu_num>1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers.cuda()
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

class M3DLoss(nn.Module):

    def __init__(self, kernel_type):
        super().__init__()
        if kernel_type == 'gaussian':
            self.kernel = RBF()

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def epoch(mode, dataloader, net, optimizer, criterion, args, aug, calc_classwise_acc = False):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if calc_classwise_acc:
        num_classes = args.num_classes
        correct_per_class = [0] * num_classes 
        total_per_class = [0] * num_classes   

    predictions_per_sample = []

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output, feats, _, _, _= net(img)
        loss = criterion(output, lab)
        
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        # Calculate class-wise accuracy
        if calc_classwise_acc:
            _, predicted = torch.max(output, 1)
            for label, prediction in zip(lab, predicted):
                if label == prediction:
                    correct_per_class[label] += 1
                total_per_class[label] += 1
                
                predictions_per_sample.append((label.item(), prediction.item()))

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    acc_test_per_class = None
    if calc_classwise_acc:
        acc_test_per_class = [0.0] * num_classes
        for class_idx in range(num_classes):
            if total_per_class[class_idx] > 0:
                acc_test_per_class[class_idx] = correct_per_class[class_idx] / total_per_class[class_idx]

    return loss_avg, acc_avg,acc_test_per_class,predictions_per_sample
    
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def pc_normalize_batch(pc):
    centroid = torch.mean(pc, dim=2, keepdims=True)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1, keepdims=True)), dim=2, keepdims=True).values
    pc = pc / m
    return pc

def seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)

def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
    net = net.to(args.device)
    images_train = pc_normalize_batch(images_train).to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)
    dst_train = TensorDataset(images_train, labels_train)
    g = torch.Generator()
    g.manual_seed(0)

    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_real, shuffle=True, num_workers=0, worker_init_fn=seed_worker, generator=g)
    start = time.time()

    best_acc = 0

    for ep in tqdm(range(Epoch+1)):
        loss_train, acc_train,_,_ = epoch('train', trainloader, net, optimizer, criterion, args, aug = False)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        
        if ep % 10 == 0 and ep > 499 :

            loss_test, acc_test, acc_test_per_class, predictions_per_sample = epoch('test', testloader, net, optimizer, criterion, args, aug=False, calc_classwise_acc=True)

            if acc_test > best_acc:
                best_acc = acc_test
                best_per_class = acc_test_per_class
                best_prediction = predictions_per_sample
                best_net = net

    time_train = time.time() - start

    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, best_acc))


    return net, acc_train, best_acc, best_per_class, best_prediction


def get_eval_pool(eval_mode, model):
    if eval_mode == 'S':  # itself
        model_eval_pool = [model]
    elif eval_mode == 'SSS': 
        model_eval_pool = [model , 'PointNetPlusPlus'] #ext
    return model_eval_pool


def build_logger(work_dir, cfgname):
    assert cfgname is not None
    log_file = cfgname + '.log'
    log_path = os.path.join(work_dir, log_file)

    logger = logging.getLogger(cfgname)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler1 = logging.FileHandler(log_path)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)

    handler2 = logging.StreamHandler()
    handler2.setFormatter(formatter)
    logger.addHandler(handler2)
    logger.propagate = False

    return logger

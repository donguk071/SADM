import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import math

from utils import *
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='MODELNET40', help='dataset')
    parser.add_argument('--model', type=str, default='PointNet', help='model')
    parser.add_argument('--ppc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=500, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1500, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=10, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_rot', type=float, default=0.01, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=8, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=8, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--addition_setting',  type=str , default='None', help='additional experiment setting')
    parser.add_argument('--mode', type=str, default='result', help='V1/V2')
    parser.add_argument('--feature_transform', type=int, default=0, help='feature transform')
    parser.add_argument('--mmdkernel', type=str, default="gaussian", help='kernel')


    args = parser.parse_args()

    os.makedirs('./'+args.mode, exist_ok=True)

    if args.addition_setting != 'None':
        log_filename = f"{args.mode}/LRimg{args.lr_img}_LRnet{args.lr_net}_ppc{args.ppc}_Model_{args.model}_It{args.Iteration}_Dataset_{args.dataset}_init_{args.init}_{args.addition_setting}"
    else:
        log_filename = f"{args.mode}/LRimg{args.lr_img}_LRnet{args.lr_net}_ppc{args.ppc}_Model_{args.model}_It{args.Iteration}_Dataset_{args.dataset}_init_{args.init}"
    logger = build_logger('.', log_filename)

    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 250).tolist() if args.eval_mode == 'S' or args.eval_mode == "SSS" else [args.Iteration] # The list of iterations when we evaluate models and record results.
    logger.info('eval_it_pool: %s ', eval_it_pool)

    if args.dataset == "MODELNET40" or "MODELNET10" or "scanobjectnn" or "shapenet":
        npoints, coord_dim, num_classes, dst_train, _, testloader = get_dataset(args, args.dataset)

    args.num_classes = num_classes

    model_eval_pool = get_eval_pool(args.eval_mode, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    accs_all = []
    data_save = []

    gf_list = []

    for exp in range(args.num_exp):
        logger.info('\n================== Exp %d ==================\n '%exp)
        logger.info('Hyper-parameters: {args.__dict__} \n')
        logger.info('Evaluation model pool: %s', model_eval_pool)

        ''' organize the real dataset '''
        pointcloud_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        pointcloud_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))] 
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        pointcloud_all = torch.cat(pointcloud_all, dim=0).to(args.device) #torch.Size([50000, 3, 32, 32]) 
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device) #torch.Size([50000]) 

        for c in range(num_classes):
            logger.info('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_pointclouds(c, n):  
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return pointcloud_all[idx_shuffle]#.to(args.device) # N,3,1024
        
        channel = 3
        for ch in range(channel):
            logger.info('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(pointcloud_all[:, ch]), 
                                                                     torch.std(pointcloud_all[:, ch])))


        ''' initialize the synthetic data '''
        pointcloud_tmp = torch.randn(size=(num_classes * args.ppc, coord_dim, npoints), dtype=torch.float,
                                requires_grad=True, device=args.device)

        pointcloud_tmp = torch.tanh(pointcloud_tmp).detach().clone().requires_grad_(True)

        label_syn = torch.tensor([np.ones(args.ppc) * i for i in range(num_classes)], dtype=torch.int,
                                 requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]

        theta_x = torch.zeros(size=(num_classes * args.ppc,), dtype=torch.float, device=args.device).requires_grad_(True)
        theta_y = torch.zeros(size=(num_classes * args.ppc,), dtype=torch.float, device=args.device).requires_grad_(True)
        theta_z = torch.zeros(size=(num_classes * args.ppc,), dtype=torch.float, device=args.device).requires_grad_(True)
        optimizer_theta = torch.optim.SGD([
            {'params': theta_x, 'lr': args.lr_rot / 10},
            {'params': theta_y, 'lr': args.lr_rot},
            {'params': theta_z, 'lr': args.lr_rot / 10}
        ], momentum=0.5)
        
        optimizer_theta.zero_grad()
        
        # Inside training loop
        def create_rotation_matrix(angle, type="x"):
            c = torch.cos(angle)
            s = torch.sin(angle)
            zeros = torch.zeros_like(c)
            ones = torch.ones_like(c)
            if type=="x":
                return torch.stack([
                    torch.stack([ones, zeros, zeros], dim=1),
                    torch.stack([zeros,    c,    -s], dim=1),
                    torch.stack([zeros,    s,     c], dim=1)
                ], dim=2)
            elif type=="y":
                return torch.stack([
                    torch.stack([    c, zeros,    s], dim=1),
                    torch.stack([zeros, ones, zeros], dim=1),
                    torch.stack([   -s, zeros,    c], dim=1)
                ], dim=2)
            else:
                return torch.stack([
                    torch.stack([    c,    -s, zeros], dim=1),
                    torch.stack([    s,     c, zeros], dim=1),
                    torch.stack([zeros, zeros, ones], dim=1)
                ], dim=2)

        if args.init == 'real':
            logger.info('initialize synthetic data from random real images')
            for c in range(num_classes):
                pointcloud_tmp.data[c*args.ppc:(c+1)*args.ppc] = get_pointclouds(c, args.ppc).detach().data
        
        else:
            logger.info('initialize synthetic data from random noise')

        '''visualize and save initial synthetic data'''
        if args.addition_setting != 'None':
            pc_div_name = f"{args.mode}/LRimg{args.lr_img}_LRnet{args.lr_net}_ppc{args.ppc}_Model_{args.model}_It{args.Iteration}_Dataset_{args.dataset}_init_{args.init}_{args.addition_setting}"
        else :
            pc_div_name = f"{args.mode}/LRimg{args.lr_img}_LRnet{args.lr_net}_ppc{args.ppc}_Model_{args.model}_It{args.Iteration}_Dataset_{args.dataset}_init_{args.init}"
        
        os.makedirs(pc_div_name, exist_ok=True)
        for c in range(num_classes): 
            label_folder = os.path.join(pc_div_name, f'class_{c}')
            os.makedirs(label_folder, exist_ok=True)
            pc_per_class = pointcloud_tmp.data[c * args.ppc:(c + 1) * args.ppc].cpu().numpy()
            for i, save_pc in enumerate(pc_per_class):
                file_name = os.path.join(label_folder, f'init_{c}_{i}')
                np.savetxt(f'{file_name}.txt', save_pc.T, delimiter=',')


        ''' training '''
        optimizer_img = torch.optim.SGD([pointcloud_tmp, ], lr=args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        logger.info('%s training begins'%get_time())

        m3d_criterion = M3DLoss(args.mmdkernel)


        gf_list = {}
        acc_list = {}
        total_acc_list = []
        for c in range(num_classes):
            gf_list[c] = []
            acc_list[c] = []

        for it in tqdm(range(args.Iteration+1)):
            with torch.no_grad():
                # Normalize theta to [-pi, pi] range
                theta_x.data = torch.remainder(theta_x.data + math.pi, 2 * math.pi) - math.pi
                theta_y.data = torch.remainder(theta_y.data + math.pi, 2 * math.pi) - math.pi
                theta_z.data = torch.remainder(theta_z.data + math.pi, 2 * math.pi) - math.pi
            rot_matrix_x = create_rotation_matrix(theta_x, 'x')
            rot_matrix_y = create_rotation_matrix(theta_y, 'y')
            rot_matrix_z = create_rotation_matrix(theta_z, 'z')

            trans_matrix = torch.bmm(rot_matrix_z, torch.bmm(rot_matrix_y, rot_matrix_x))
            
            pc_transposed = pointcloud_tmp.permute(0, 2, 1).contiguous()
            pc_rotated = torch.bmm(pc_transposed, trans_matrix)
            pointcloud_syn_new = pc_rotated.permute(0, 2, 1).contiguous()
            pointcloud_syn = pointcloud_syn_new

            ''' Evaluate synthetic data '''
            if it in eval_it_pool : # and it != 0:
                for model_eval in model_eval_pool:

                    args.epoch_eval_train = 500

                    accs = []
                    accs_per_class = [] 
                    all_real_labels = []  
                    all_predicted_labels = []  
                    for it_eval in range(args.num_eval):
    
                        torch.manual_seed(1996+it_eval)
                        
                        net_eval = get_network(model_eval, channel, num_classes, args.feature_transform).to(args.device) # get a random model
                        pointcloud_syn_eval, label_syn_eval = copy.deepcopy(pointcloud_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, _, acc_test, acc_test_per_class, predictions_per_sample = evaluate_synset(it_eval, net_eval, pointcloud_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)

                        if len(accs_per_class) == 0:  # Initialize the per-class accuracy list
                            accs_per_class = [[] for _ in range(num_classes)]
                        
                        for class_idx in range(num_classes):
                            accs_per_class[class_idx].append(acc_test_per_class[class_idx])
                        
                        real_labels = [real for real, _ in predictions_per_sample]
                        predicted_labels = [pred for _, pred in predictions_per_sample]
                        all_real_labels.extend(real_labels)
                        all_predicted_labels.extend(predicted_labels)
                    
                    accs_per_class = [np.mean(accs_per_class[class_idx]) for class_idx in range(num_classes)]
                    logger.info('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))


                    accs_all.append(np.mean(accs))

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs
                        
                    
                ''' visualize and save '''
                pointcloud_syn_vis = copy.deepcopy(pointcloud_syn.detach().cpu().numpy())

                for c in range(num_classes): 
                    label_folder = os.path.join(pc_div_name, f'class_{c}')
                    os.makedirs(label_folder, exist_ok=True)
                    pc_syn_per_class = pointcloud_syn_vis[c * args.ppc:(c + 1) * args.ppc]
                    for i, save_syn_pc in enumerate(pc_syn_per_class): 
                        file_name = os.path.join(label_folder, f'iter_{it}_class_{c}_{i}')
                        np.savetxt(f'{file_name}.txt', save_syn_pc.T, delimiter=',')
           
            ''' Train synthetic data '''
            channel = 3
            net = get_network(args.model, channel, num_classes,args.feature_transform).to(args.device) # get a random model
            net.train()

            for param in list(net.parameters()):
                param.requires_grad = False

            loss_avg = 0

            BN_flag = False
            BNSizePC = 8  # for batch normalization
            for module in net.modules():
                if 'BatchNorm' in module._get_name(): #BatchNorm
                    BN_flag = True
            if BN_flag:
                pc_real = torch.cat([get_pointclouds(c, BNSizePC) for c in range(num_classes)], dim=0)
                net.train() # for updating the mu, sigma of BatchNorm
                output_real = net(pc_real) # get running mu, sigma
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():  #BatchNorm
                        module.eval() # fix mu and sigma of every BatchNorm layer

            ''' update synthetic data '''

            loss = torch.tensor(0.0).to(args.device)

            for c in range(num_classes):

                pc_syn = pointcloud_syn[c * args.ppc:(c + 1) * args.ppc].reshape(
                    (args.ppc, coord_dim, npoints)).to(args.device)

                pc_real = get_pointclouds(c, args.batch_real)


                with torch.no_grad():
                    _, _, _, _, layers_real = net(pc_real)
                
                _, _, _, _, layers_syn = net(pc_syn)

                sorted_real = torch.sort(layers_real["x_m"], dim=2, descending=True)[0].detach()
                sorted_syn = torch.sort(layers_syn["x_m"], dim=2, descending=True)[0]

                real = sorted_real.mean(dim=0)
                syn = sorted_syn.mean(dim=0)

                loss1 = (((real - syn)**2).sum(dim=0)).mean() * 0.2 + m3d_criterion(layers_real['x_gf'], layers_syn['x_gf']) * 0.001
                loss += loss1 * args.ppc



            optimizer_img.zero_grad()
            optimizer_theta.zero_grad()
            loss.backward()
            optimizer_img.step()
            optimizer_theta.step()

            loss_avg += loss.item()

            loss_avg /= (num_classes)

            if it%50 == 0:
                logger.info('%s iter = %04d, loss avg = %.4f, loss = %.4f' % (get_time(), it, loss_avg, loss))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(pointcloud_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dppc.pt'%(args.method, args.dataset, args.model, args.ppc)))


    logger.info('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        logger.info('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))

    logger.info(np.array(accs_all).reshape(-1,len(model_eval_pool))[np.argmax(np.mean(np.array(accs_all).reshape(-1,len(model_eval_pool)), axis=1))])
    logger.info(np.array(accs_all).reshape(-1,len(model_eval_pool)))
    print('\n==================== Final Results ====================\n')
    print(np.array(accs_all).reshape(-1,len(model_eval_pool)))

if __name__ == '__main__':
    random.seed(1999)
    np.random.seed(1999)
    torch.manual_seed(1999)
    torch.cuda.manual_seed(1999)
    torch.cuda.manual_seed_all(1999)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main()
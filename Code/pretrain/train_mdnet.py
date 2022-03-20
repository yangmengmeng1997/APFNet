import os, sys
import pickle
import yaml
import time
import argparse
import numpy as np
import pdb
import torch

sys.path.insert(0,'.')
from data_prov import RegionDataset
sys.path.insert(0,'./modules')
from model import MDNet, set_optimizer, BCELoss, Precision


def train_mdnet(opts):

    # Init dataset
    if opts['dataset'] == 'gtot':
        opts['data_path'] = './pretrain/data/GTOT_ALL.pkl'
    elif opts['dataset'] == 'rgbt234':
        opts['data_path'] = './pretrain/data/RGBT234_ALL.pkl'
    with open(opts['data_path'], 'rb') as fp:
        data = pickle.load(fp)
    K = len(data)
    dataset = [None] * K
    for k, seq in enumerate(data.values()):
        dataset[k] = RegionDataset(seq['images_v'], seq['images_i'], seq['gt'], opts)

    # Init model
    model = MDNet(opts['init_model_path'], K)
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    # Init criterion and optimizer
    criterion = BCELoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'], opts['lr_mult'])
    best_score = 0.
    # Main trainig loop
    for i in range(opts['n_cycles']):
        print('==== Start Cycle {:d}/{:d} ===='.format(i + 1, opts['n_cycles']))

        if i in opts.get('lr_decay', []):
            print('decay learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opts.get('gamma', 0.1)

        # Training
        model.train()
        prec = np.zeros(K)
        k_list = np.random.permutation(K)
        for j, k in enumerate(k_list):
            tic = time.time()
            # training
            pos_regions_v, neg_regions_v, pos_regions_i, neg_regions_i = dataset[k].next()
            if opts['use_gpu']:
                pos_regions_v = pos_regions_v.cuda()
                neg_regions_v = neg_regions_v.cuda()
                pos_regions_i = pos_regions_i.cuda()
                neg_regions_i = neg_regions_i.cuda()
            pos_score = model(pos_regions_v, pos_regions_i, k)
            neg_score = model(neg_regions_v, neg_regions_i, k)

            loss = criterion(pos_score, neg_score)

            batch_accum = opts.get('batch_accum', 1)
            if j % batch_accum == 0:
                model.zero_grad()
            loss.backward()
            if j % batch_accum == batch_accum - 1 or j == len(k_list) - 1:
                if 'grad_clip' in opts:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
                optimizer.step()

            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time()-tic
            print('Cycle {:2d}/{:2d}, Iter {:2d}/{:2d} (Domain {:2d}), Loss {:.3f}, Precision {:.3f}, Time {:.3f}'
                    .format(i, opts['n_cycles'], j, len(k_list), k, loss.item(), prec[k], toc))
        cur_score = prec.mean()
        print('Mean Precision: {:.3f}'.format(cur_score))
        if cur_score > best_score:
            best_score = cur_score
            print('Save model to {:s}'.format(opts['model_path']+str(i)+'.pth')) #only save one
            if opts['use_gpu']:
                model = model.cpu()
            states = {'layers_v': model.layers_v.state_dict(),
                      'layers_i': model.layers_i.state_dict(),
                      'fc': model.fc.state_dict()}
            torch.save(states, opts['model_path']+'.pth')   #only save one
            if opts['use_gpu']:
                model = model.cuda()


if __name__ == "__main__":
    #print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='rgbt234', help='training dataset {gtot, rgbt234}')
    parser.add_argument("-model_path", default ="./models/RGBT", help = "model path")
    parser.add_argument("-init_model_path", default="./models/mdnet_imagenet_vid.pth")
    #parser.add_argument('--seed', type=int, default=123456,help='random seed')
    args = parser.parse_args()
    #seed_torch(args.seed)
    opts = yaml.safe_load(open('./pretrain/options_rgbt.yaml'))
    # pdb.set_trace()
    opts['dataset'] = args.dataset
    opts['model_path'] = args.model_path
    opts['init_model_path'] = args.init_model_path
    train_mdnet(opts)

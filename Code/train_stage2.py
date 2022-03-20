#encoding=utf-8
import os, sys
import pickle
import yaml
import time
import argparse
import numpy as np
import pdb
import torch
import logging

#sys.path.insert(0,'.')
#from data_prov import RegionDataset
sys.path.insert(0,'./modules')
#from model import MDNet, set_optimizer, BCELoss, Precision
from model_stage2 import MDNet, set_optimizer, BCELoss, Precision
sys.path.insert(0,'./pretrain')
#Rememeber to change the pretrain_option for stage2
from pretrain_option import *
from data_prov import RegionDataset

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def train_mdnet(opts):

    # Init dataset
    ## set image directory
    if opts['set_type'] == 'RGBT234_ALL':
        img_home = '/DATA/zhuli/RGBT234/'  
        data_path = './pretrain/data/RGBT234_ALL.pkl'
    #********************************************
    elif opts['set_type'] == 'GTOT_ALL':
        img_home = '/DATA/zhuli/GTOT_withannotation/GTOT/'
        data_path = './pretrain/data/GTOT_ALL.pkl'
    #*********************************************
    elif opts['set_type'] == 'RGBT234_FM':
        img_home = '/DATA/zhuli/RGBT234/'
        data_path = './pretrain/data/RGBT234_FM.pkl'
    elif opts['set_type'] == 'RGBT234_SC':
        img_home = '/DATA/zhuli/RGBT234/'
        data_path = './pretrain/data/RGBT234_SC.pkl'
    elif opts['set_type'] == 'RGBT234_OCC':
        img_home = '/DATA/zhuli/RGBT234/'        
        data_path = './pretrain/data/RGBT234_OCC.pkl'
    elif opts['set_type'] == 'RGBT234_ILL':
        img_home = '/DATA/zhuli/RGBT234/'
        data_path = './pretrain/data/RGBT234_ILL.pkl'
    elif opts['set_type'] == 'RGBT234_TC':
        img_home = '/DATA/zhuli/RGBT234/'   
        data_path = './pretrain/data/RGBT234_TC.pkl'
    #************************************************
    elif opts['set_type'] == 'GTOT_FM':
        img_home = '/DATA/yangmengmeng/GTOT_withannotation/GTOT/'
        data_path = './pretrain/data/GTOT_FM.pkl'
    elif opts['set_type'] == 'GTOT_SC':
        img_home = '/DATA/yangmengmeng/GTOT_withannotation/GTOT/'
        data_path = './pretrain/data/GTOT_SC.pkl'
    elif opts['set_type'] == 'GTOT_OCC':
        img_home = '/DATA/yangmengmeng/GTOT_withannotation/GTOT/'
        data_path = './pretrain/data/GTOT_OCC.pkl'
    elif opts['set_type'] == 'GTOT_ILL':
        img_home = '/DATA/yangmengmeng/GTOT_withannotation/GTOT/'
        data_path = './pretrain/data/GTOT_ILL.pkl'
    elif opts['set_type'] == 'GTOT_TC':
        img_home = '/DATA/yangmengmeng/GTOT_withannotation/GTOT/'
        data_path = './pretrain/data/GTOT_TC.pkl'
    #*****************************************************

    with open(data_path, 'rb') as fp:
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
    model.get_learnable_params()

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
        logger.info('Mean Precision: {:.3f}'.format(cur_score))
        if cur_score > best_score:
            best_score = cur_score
            print('Save model to {:s}'.format(opts['model_path']+str(i)+'.pth')) #only save one
            if opts['use_gpu']:
                model = model.cpu()
            states = {
                      'layers_v': model.layers_v.state_dict(),
                      'layers_i': model.layers_i.state_dict(),
                      'fc': model.fc.state_dict(),
                      'parallel1': model.parallel1.state_dict(),
                      'parallel2': model.parallel2.state_dict(),
                      'parallel3': model.parallel3.state_dict(),
                      'parallel1_skcov': model.parallel1_skconv.state_dict(),
                      'parallel2_skcov': model.parallel2_skconv.state_dict(),
                      'parallel3_skcov': model.parallel3_skconv.state_dict(),
                      'ensemble1_skcov': model.ensemble1_skconv.state_dict(),
                      'ensemble2_skcov': model.ensemble2_skconv.state_dict(),
                      'ensemble3_skcov': model.ensemble3_skconv.state_dict(),
                      }
            torch.save(states, opts['model_path']+'.pth')   #only save the best one
            if cur_score>0.95:      #we also save some good model
                torch.save(states, opts['model_path']+str(i)+'.pth')
                logger.info('Save model to {:s}'.format(opts['model_path']+str(i)+'.pth'))
            if opts['use_gpu']:
                model = model.cuda()


#We save the attributes ,the backbone ,the aggregation in one model
if __name__ == "__main__":
    logger = get_logger('./log/trainGTOTSKALL.log')
    parser = argparse.ArgumentParser()
    #set yourdataset
    parser.add_argument("-set_type", default = 'GTOT_ALL')
    #Saving model path
    parser.add_argument("-model_path", default ="/home/zhuli/xuewanlin/cat/MDNet_CAT_SK_Trans/models/GTOT_ALL", help = "model path")
    #your backbone model 
    parser.add_argument("-init_model_path", default="/home/zhuli/xuewanlin/cat/MDNet_CAT_SK_Trans/models/GTOT.pth")
    parser.add_argument("-batch_frames", default = 8, type = int)
    parser.add_argument("-lr", default=0.0001, type = float)     #set it bysourself
    parser.add_argument("-batch_pos",default = 32, type = int)
    parser.add_argument("-batch_neg", default = 96, type = int)
    parser.add_argument("-n_cycles", default = 500, type = int ) #set it bysourself
    args = parser.parse_args()

    ##option setting
    opts['set_type'] = args.set_type
    opts['model_path']=args.model_path
    opts['init_model_path'] = args.init_model_path
    opts['batch_frames'] = args.batch_frames
    opts['lr'] = args.lr
    opts['batch_pos'] = args.batch_pos 
    opts['batch_neg'] = args.batch_neg 
    opts['n_cycles'] = args.n_cycles
    print(opts)

    train_mdnet(opts)

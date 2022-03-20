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
from model_stage3 import MDNet, set_optimizer, BCELoss, Precision
sys.path.insert(0,'./pretrain')
#Rememeber to change the pretrain_option for stage3
from pretrain_option import *
from data_prov import RegionDataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
        img_home = '/DATA/liulei/RGBT234/'  
        data_path = './pretrain/data/RGBT234_ALL.pkl'
    #********************************************
    elif opts['set_type'] == 'GTOT_ALL':
        img_home = '/DATA/zhuli/GTOT_withannotation/GTOT/'
        data_path = './pretrain/data/GTOT_ALL.pkl'
    #*********************************************
    elif opts['set_type'] == 'RGBT234_FM':
        img_home = '/DATA/liulei/RGBT234/'
        data_path = './pretrain/data/RGBT234_FM.pkl'
    elif opts['set_type'] == 'RGBT234_SC':
        img_home = '/DATA/liulei/RGBT234/'
        data_path = './pretrain/data/RGBT234_SC.pkl'
    elif opts['set_type'] == 'RGBT234_OCC':
        img_home = '/DATA/liulei/RGBT234/'        
        data_path = './pretrain/data/RGBT234_OCC.pkl'
    elif opts['set_type'] == 'RGBT234_ILL':
        img_home = '/DATA/liulei/RGBT234/'
        data_path = './pretrain/data/RGBT234_ILL.pkl'
    elif opts['set_type'] == 'RGBT234_TC':
        img_home = '/DATA/liulei/RGBT234/'   
        data_path = './pretrain/data/RGBT234_TC.pkl'
    #************************************************
    elif opts['set_type'] == 'GTOT_FM':
        img_home = '/DATA/zhuli/GTOT_withannotation/GTOT/'
        data_path = './pretrain/data/GTOT_FM.pkl'
    elif opts['set_type'] == 'GTOT_SC':
        img_home = '/DATA/zhuli/GTOT_withannotation/GTOT/'
        data_path = './pretrain/data/GTOT_SC.pkl'
    elif opts['set_type'] == 'GTOT_OCC':
        img_home = '/DATA/zhuli/GTOT_withannotation/GTOT/'
        data_path = './pretrain/data/GTOT_OCC.pkl'
    elif opts['set_type'] == 'GTOT_ILL':
        img_home = '/DATA/zhuli/GTOT_withannotation/GTOT/'
        data_path = './pretrain/data/GTOT_ILL.pkl'
    elif opts['set_type'] == 'GTOT_TC':
        img_home = '/DATA/zhuli/GTOT_withannotation/GTOT/'
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
            logger.info('Save model to {:s}'.format(opts['model_path']+str(i)+'.pth'))
            if opts['use_gpu']:
                model = model.cpu()
            states = {
                      'layers_v': model.layers_v.state_dict(),
                      'layers_i': model.layers_i.state_dict(),
                      'fc': model.fc.state_dict(),
                      'parallel1': model.parallel1.state_dict(),
                      'parallel2': model.parallel2.state_dict(),
                      'parallel3': model.parallel3.state_dict(),
                      'parallel1_skconv': model.parallel1_skconv.state_dict(),
                      'parallel2_skconv': model.parallel2_skconv.state_dict(),
                      'parallel3_skconv': model.parallel3_skconv.state_dict(),
                      'ensemble1_skconv': model.ensemble1_skconv.state_dict(),
                      'ensemble2_skconv': model.ensemble2_skconv.state_dict(),
                      'ensemble3_skconv': model.ensemble3_skconv.state_dict(),
                      'transformer1_encoder1': model.transformer1_encoder1.state_dict(),
                      'transformer1_encoder2': model.transformer1_encoder2.state_dict(),
                      'transformer1_encoder3': model.transformer1_encoder3.state_dict(),
                      'transformer1_decoder1': model.transformer1_decoder1.state_dict(),
                      'transformer1_decoder2': model.transformer1_decoder2.state_dict(),
                      'transformer2_encoder1': model.transformer2_encoder1.state_dict(),
                      'transformer2_encoder2': model.transformer2_encoder2.state_dict(),
                      'transformer2_encoder3': model.transformer2_encoder3.state_dict(),
                      'transformer2_decoder1': model.transformer2_decoder1.state_dict(),
                      'transformer2_decoder2': model.transformer2_decoder2.state_dict(),
                      'transformer3_encoder1': model.transformer3_encoder1.state_dict(),
                      'transformer3_encoder2': model.transformer3_encoder2.state_dict(),
                      'transformer3_encoder3': model.transformer3_encoder3.state_dict(),
                      'transformer3_decoder1': model.transformer3_decoder1.state_dict(),
                      'transformer3_decoder2': model.transformer3_decoder2.state_dict(),
                    }
            torch.save(states, opts['model_path']+'.pth')   #only save the best one
            if cur_score>0.95:      #we also save some good model
                torch.save(states, opts['model_path']+str(i)+'.pth')
                logger.info('Save model to {:s}'.format(opts['model_path']+str(i)+'.pth'))
            if opts['use_gpu']:
                model = model.cuda()


if __name__ == "__main__":
    logger = get_logger('./log/Train_RGBT234_ALL_Transformer.log')
    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default = 'RGBT234_ALL')
    parser.add_argument("-model_path", default ="/DATA/yangmengmeng/MyCode/MDNet_CAT_SK_Transformer/models/RGBT234_ALL_Transformer", help = "model path")
    parser.add_argument("-init_model_path", default="/DATA/yangmengmeng/MyCode/MDNet_CAT_SK_Transformer/models/RGBT234_ALL.pth")
    parser.add_argument("-batch_frames", default = 8, type = int)
    parser.add_argument("-lr", default=0.0001, type = float)
    parser.add_argument("-batch_pos",default = 32, type = int)
    parser.add_argument("-batch_neg", default = 96, type = int)
    parser.add_argument("-n_cycles", default = 1000, type = int )
    args = parser.parse_args()

    ##option setting
    opts['set_type'] = args.set_type
    opts['model_path']=args.model_path
    opts['init_model_path'] = args.init_model_path
    opts['batch_frames'] = args.batch_frames
    opts['lr'] = args.lr
    opts['batch_pos'] = args.batch_pos  # original = 64
    opts['batch_neg'] = args.batch_neg  # original = 192
    opts['n_cycles'] = args.n_cycles
    print(opts)

    train_mdnet(opts)

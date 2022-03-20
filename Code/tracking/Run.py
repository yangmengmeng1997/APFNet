import numpy as np
import os
import sys
import time
import argparse
import yaml, json
from PIL import Image
import logging

import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.optim as optim

sys.path.insert(0, './modules')
from model_tracking import MDNet, BCELoss, set_optimizer
from sample_generator import SampleGenerator
from utils import overlap_ratio
from data_prov import RegionExtractor
from bbreg import BBRegressor
from gen_config import gen_config
import pdb
sys.path.insert(0, './pretrain')
from option import *

os.environ["CUDA_VISIBLE_DEVICES"] = '5'

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


def forward_samples(model, image_v, image_i, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image_v, image_i, samples, opts)
    for i, regions in enumerate(extractor):
        regions_v = regions[0]
        regions_i = regions[1]
        if opts['use_gpu']:
            regions_v = regions_v.cuda()
            regions_i = regions_i.cuda()
        with torch.no_grad():
            feat = model(regions_v, regions_i,out_layer=out_layer)    
        if i==0:
            feats = feat.detach().clone()
        else:
            feats = torch.cat((feats, feat.detach().clone()), dim=0)
    return feats


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for i in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = pos_feats[pos_cur_idx]
        batch_neg_feats = neg_feats[neg_cur_idx]

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                with torch.no_grad():
                    score = model(batch_neg_feats[start:end], batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.detach()[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats[top_idx]
            model.train()

        # forward
        pos_score = model(batch_pos_feats, batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        if 'grad_clip' in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()


def run_mdnet(img_list_visible, img_list_infrared, init_bbox, gt, args, savefig_dir='', display=False):

    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list_visible), 4))
    result_bb = np.zeros((len(img_list_visible), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    if gt is not None:
        overlap = np.zeros(len(img_list_visible))
        overlap[0] = 1

    # Init model
    model = MDNet(args.model_path)
    if opts['use_gpu']:
        model = model.cuda()

    # Init criterion and optimizer 
    criterion = BCELoss()
    model.set_learnable_params(opts['ft_layers'])
    model.get_learnable_params()
    init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_mult'])
    update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_mult'])

    tic = time.time()
    # Load first image
    image_v = Image.open(img_list_visible[0]).convert('RGB')
    image_i = Image.open(img_list_infrared[0]).convert('RGB')

    # Draw pos/neg samples
    pos_examples = SampleGenerator('gaussian', image_v.size, opts['trans_pos'], opts['scale_pos'])(
                        target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
                    SampleGenerator('uniform', image_v.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
                    SampleGenerator('whole', image_v.size)(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats = forward_samples(model, image_v, image_i, pos_examples)
    neg_feats = forward_samples(model, image_v, image_i, neg_examples)

    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
    del init_optimizer, neg_feats
    torch.cuda.empty_cache()

    # Train bbox regressor
    bbreg_examples = SampleGenerator('uniform', image_v.size, opts['trans_bbreg'], opts['scale_bbreg'], opts['aspect_bbreg'])(
                        target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
    bbreg_feats = forward_samples(model, image_v, image_i, bbreg_examples)
    bbreg = BBRegressor(image_v.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
    del bbreg_feats
    torch.cuda.empty_cache()

    # Init sample generators for update
    sample_generator = SampleGenerator('gaussian', image_v.size, opts['trans'], opts['scale'])
    pos_generator = SampleGenerator('gaussian', image_v.size, opts['trans_pos'], opts['scale_pos'])
    neg_generator = SampleGenerator('uniform', image_v.size, opts['trans_neg'], opts['scale_neg'])

    # Init pos/neg features for update
    neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
    neg_feats = forward_samples(model, image_v, image_i, neg_examples)
    pos_feats_all = [pos_feats]
    neg_feats_all = [neg_feats]

    spf_total = time.time() - tic

    # Display
    savefig = savefig_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (image_v.size[0] / dpi, image_v.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image_v, aspect='auto')

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)

    # Main loop
    for i in range(1, len(img_list_visible)):

        tic = time.time()
        # Load image
        image_v = Image.open(img_list_visible[i]).convert('RGB')
        image_i = Image.open(img_list_infrared[i]).convert('RGB')

        # Estimate target bbox
        samples = sample_generator(target_bbox, opts['n_samples'])
        
        sample_scores=forward_samples(model,image_v,image_i,samples,out_layer='fc6')

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
        success = target_score > 0
        
        # Expand search area at failure
        if success:
            sample_generator.set_trans(opts['trans'])
        else:
            sample_generator.expand_trans(opts['trans_limit'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None,:]
            bbreg_feats = forward_samples(model, image_v, image_i, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
            pos_feats = forward_samples(model, image_v, image_i, pos_examples)
            pos_feats_all.append(pos_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]

            neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
            neg_feats = forward_samples(model, image_v, image_i, neg_examples)
            neg_feats_all.append(neg_feats)
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.cat(pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            #we update the new templeate 
            first_samples=np.expand_dims(target_bbox,axis=0)
            first_samples=first_samples.repeat(256, axis=0)  
            #first image extractor
            first_extractor = RegionExtractor(image_v, image_i, first_samples, opts)
            pos_data = torch.cat(pos_feats_all, 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        torch.cuda.empty_cache()
        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image)

            if gt is not None:
                gt_rect.set_xy(gt[i, :2])
                gt_rect.set_width(gt[i, 2])
                gt_rect.set_height(gt[i, 3])

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir, '{:04d}.jpg'.format(i)), dpi=dpi)

        if gt is None:
            print('Frame {:d}/{:d}, Score {:.3f}, Time {:.3f}'
                .format(i, len(img_list_visible), target_score, spf))
        else:
            overlap[i] = overlap_ratio(gt[i], result_bb[i])[0]
            print('Frame {:d}/{:d}, Overlap {:.3f}, Score {:.3f}, Time {:.3f}'
                .format(i, len(img_list_visible), overlap[i], target_score, spf))

    if gt is not None:
        print('meanIOU: {:.3f}'.format(overlap.mean()))
    fps = len(img_list_visible) / spf_total
    return overlap, result, result_bb, fps


if __name__ == "__main__":
    logger = get_logger('./log/GTOT_ALL_Transformer.log')    #Record your log files
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')
    parser.add_argument("-dataset", default = 'RGBT234' )             #testing dataset
    parser.add_argument("-model_path", default ='./models/GTOT_ALL_Transformer.pth')   # your model
    parser.add_argument("-result_path", default ='/DATA/yangmengmeng/MyCode/MDNet_CAT_SK_Transformer/results/') #your result path
    args = parser.parse_args()
    ##your result path
    args.result_path = '/DATA/yangmengmeng/MyCode/MDNet_CAT_SK_Transformer/results/'+args.dataset+'/'+args.model_path.split('/')[-1].split('.')[0]+'/'
    #assert args.seq != '' or args.json != ''
    print(opts)

    dataset_path = os.path.join('/DATA/', args.dataset)     #dataset path
    mylist='/DATA/RGBT234.txt'    #dataset list path
    mypath=open(mylist) 
    seq_list=[]
    while True:
        line=mypath.readline().strip()
        if line:
            seq_list.append(line)
        else:
            break
    mypath.close()
    seq_list = [f for f in os.listdir(dataset_path)]
    iou_list=[]
    fps_list=dict()
    bb_result = dict()
    result = dict()
    bb_result_nobb=dict()


    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
    for num,seq in enumerate(seq_list):
        torch.cuda.empty_cache()
        np.random.seed(123)
        torch.manual_seed(456)
        torch.cuda.manual_seed(789)
        seq_path = dataset_path + '/' + seq
        if 'txt' in seq or args.model_path.split('/')[-1].split('.')[0]+'_'+seq+'.txt' in os.listdir(args.result_path) or num<-1:
            continue
        # Generate sequence config
        img_list_v,img_list_i,gt=gen_config(seq_path,args.dataset)
    
        # Run tracker
        iou_result, result, result_bb, fps = run_mdnet(img_list_v, img_list_i, gt[0], gt, args)
    
        # Save result
        iou_list.append(iou_result.sum()/len(iou_result))
        bb_result[seq] = result_bb
        fps_list[seq]=fps

        bb_result_nobb[seq] = result
        print('{} {} : {} , total mIoU:{}, fps:{}'.format(num,seq,iou_result.mean(), sum(iou_list)/len(iou_list),sum(fps_list.values())/len(fps_list)))
        logger.info('{} {} : {} , total mIoU:{}, fps:{}'.format(num,seq,iou_result.mean(), sum(iou_list)/len(iou_list),sum(fps_list.values())/len(fps_list)))
        for i in range(len(result_bb)):
            with open(args.result_path+args.model_path.split('/')[-1].split('.')[0]+'_'+seq+'.txt', 'a') as f:
                res='{} {} {} {} {} {} {} {}'.format(result_bb[i][0],result_bb[i][1],result_bb[i][0]+result_bb[i][2],result_bb[i][1],result_bb[i][0]+result_bb[i][2],result_bb[i][1]+result_bb[i][3],result_bb[i][0],result_bb[i][1]+result_bb[i][3]) 
                f.write(res)
                f.write('\n')

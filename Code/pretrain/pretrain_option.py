from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = True

opts['init_model_path'] = './models/imagenet-vgg-m.mat'
opts['model_path'] = './models'

opts['batch_frames'] = 8
opts['batch_pos'] = 32
opts['batch_neg'] = 96

opts['overlap_pos'] = [0.7, 1]
opts['overlap_neg'] = [0, 0.5]

opts['img_size'] = 107
opts['padding']=16
opts['batch_accum']=50

# training examples sampling
opts['trans_pos']=0.1
opts['scale_pos']=1.3
opts['trans_neg']=2
opts['scale_neg']=1.6

# augmentation
opts['flip']=True
opts['rotate']=30
opts['blur']=7

opts['lr'] = 0.0001
opts['grad_clip'] = 10
#stage1 common
# opts['ft_layers'] = ['parallel','fc']
# opts['lr_mult'] = {'parallel':10,'fc':5}
# opts['n_cycles'] = 500

##stage2 ensemble sknet
# opts['ft_layers'] = ['ensemble','fc6']
# opts['lr_mult'] = {'ensemble':10,'fc6':5}
# opts['n_cycles'] = 500

#stage3 ensemble Transformer fine-tune all
opts['ft_layers'] = ['transformer','fc','layer','parallel','ensemble']
opts['lr_mult'] = {'transformer':10,'fc':1,'fc6':5,'layer':1,'parallel':1,'ensemble':1}
opts['n_cycles'] = 500

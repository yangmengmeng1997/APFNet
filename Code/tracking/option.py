from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = True

opts['init_model_path'] = './models/imagenet-vgg-m.mat'
opts['model_path'] = './models'

# batch size
opts['batch_pos'] = 32
opts['batch_neg'] = 96
opts['batch_neg_cand']= 1024
opts['batch_test']= 256

# candidates sampling
opts['n_samples'] = 256
opts['trans'] = 0.6    #the orignal is 0.6
opts['scale'] = 1.05   #the original is  1.05
opts['trans_limit'] = 1.5 # the original is 1.5

# input size
opts['img_size'] = 107
opts['padding'] = 16

# training examples sampling
opts['trans_pos'] = 0.1
opts['scale_pos'] = 1.3
opts['trans_neg_init'] = 1
opts['scale_neg_init'] = 1.6
opts['trans_neg'] = 2
opts['scale_neg'] = 1.3

# bounding box regression
opts['n_bbreg']=1000
opts['overlap_bbreg']=[0.6, 1]
opts['trans_bbreg']=0.3
opts['scale_bbreg']=1.6
opts['aspect_bbreg']=1.1

# initial training
opts['lr_init']=0.0005
opts['maxiter_init']=50
opts['n_pos_init']=500
opts['n_neg_init']=5000
opts['overlap_pos_init']=[0.7, 1]
opts['overlap_neg_init']=[0, 0.5]

# online training
opts['lr_update']=0.001
opts['maxiter_update']=15
opts['n_pos_update']=50
opts['n_neg_update']=200
opts['overlap_pos_update']=[0.7, 1]
opts['overlap_neg_update']=[0, 0.3]

# update criteria
opts['long_interval']=10
opts['n_frames_long']=100
opts['n_frames_short']=30

opts['lr'] = 0.0001
opts['grad_clip'] = 10

opts['lr_mult'] = {'fc4':5,'fc5':5,'fc6':10}
opts['ft_layers'] = ['fc']

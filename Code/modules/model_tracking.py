#encoding=utf-8
import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce
import math

'''
  Complete network structure: Encoder and Decoder were added on a two-phase basis
'''

def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.items():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError('Duplicated param name: {:s}'.format(name))


def set_optimizer(model, lr_base, lr_mult, train_all=False, momentum=0.9, w_decay=0.0005):
    if train_all:
        params = model.get_all_params()
    else:
        params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer

#we add Transformer encoder and decoder 

class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        #backbone
        self.layers_v = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),nn.ReLU(inplace=True),nn.LocalResponseNorm(2),nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),nn.ReLU(inplace=True),nn.LocalResponseNorm(2),nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),nn.ReLU()))]))
        self.layers_i = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),nn.ReLU(inplace=True),nn.LocalResponseNorm(2),nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),nn.ReLU(inplace=True),nn.LocalResponseNorm(2),nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),nn.ReLU()))]))
        self.fc = nn.Sequential(OrderedDict([
                ('fc4',   nn.Sequential(nn.Linear(512 * 3 * 3 * 2, 512),nn.ReLU(inplace=True))),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),nn.Linear(512, 512),nn.ReLU(inplace=True)))]))
              
        self.parallel1 = nn.ModuleList([nn.Sequential(OrderedDict([ #0:FM 1:OCC 2:SC 3:TC 4:ILL
                ('parallel1_conv1',nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=2),nn.ReLU())),
                ('parallel1_conv2',nn.Sequential(nn.Conv2d(32, 96, kernel_size=4, stride=2)))])) for _ in range(5)])
        
        self.parallel1_skconv=nn.ModuleList([nn.Sequential(OrderedDict([
                ('parallel1_skconv_global_pool',nn.AdaptiveAvgPool2d(1)),    
                ('parallel1_skconv_fc1',nn.Sequential(nn.Conv2d(96,32,1,bias=False),   
                                       nn.ReLU(inplace=True))),      
                ('parallel1_skconv_fc2',nn.Sequential(nn.Conv2d(32,96*2,1,1,bias=False)))])) for _ in range(5)])
                
        self.parallel2 = nn.ModuleList([nn.Sequential(OrderedDict([
                ('parallel2_conv1',nn.Sequential(nn.Conv2d(96, 256, kernel_size=3, stride=2),nn.MaxPool2d(kernel_size=8, stride=1)))])) for _ in range(5)])
        
        self.parallel2_skconv=nn.ModuleList([nn.Sequential(OrderedDict([
                ('parallel2_skconv_global_pool',nn.AdaptiveAvgPool2d(1)),    
                ('parallel2_skconv_fc1',nn.Sequential(nn.Conv2d(256,32,1,bias=False),   
                                       nn.ReLU(inplace=True))),       
                ('parallel2_skconv_fc2',nn.Sequential(nn.Conv2d(32,256*2,1,1,bias=False)))])) for _ in range(5)])     
        self.parallel3 = nn.ModuleList([nn.Sequential(OrderedDict([
                ('parallel3_conv1',nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=1),nn.MaxPool2d(kernel_size=3, stride=1)))])) for _ in range(5)])
        
        self.parallel3_skconv=nn.ModuleList([nn.Sequential(OrderedDict([
                ('parallel3_skconv_global_pool',nn.AdaptiveAvgPool2d(1)),    
                ('parallel3_skconv_fc1',nn.Sequential(nn.Conv2d(512,64,1,bias=False),   
                                       nn.ReLU(inplace=True))),   
                ('parallel3_skconv_fc2',nn.Sequential(nn.Conv2d(64,512*2,1,1,bias=False)))])) for _ in range(5)])
        
        self.ensemble1_skconv=nn.Sequential(OrderedDict([
                ('ensemble1_skconv_global_pool',nn.AdaptiveAvgPool2d(1)),    
                ('ensemble1_skconv_fc1',nn.Sequential(nn.Conv2d(96,32*5,1,bias=False),   
                                       nn.ReLU(inplace=True))),      
                ('ensemble1_skconv_fc2',nn.Sequential(nn.Conv2d(32*5,96*5,1,1,bias=False)))]))

        self.ensemble2_skconv=nn.Sequential(OrderedDict([
                ('ensemble2_skconv_global_pool',nn.AdaptiveAvgPool2d(1)),    
                ('ensemble2_skconv_fc1',nn.Sequential(nn.Conv2d(256,64*5,1,bias=False),   
                                       nn.ReLU(inplace=True))),      
                ('ensemble2_skconv_fc2',nn.Sequential(nn.Conv2d(64*5,256*5,1,1,bias=False)))]))
        
        self.ensemble3_skconv=nn.Sequential(OrderedDict([
                ('ensemble3_skconv_global_pool',nn.AdaptiveAvgPool2d(1)),    
                ('ensemble3_skconv_fc1',nn.Sequential(nn.Conv2d(512,128*5,1,bias=False),   
                                       nn.ReLU(inplace=True))),     
                ('ensemble3_skconv_fc2',nn.Sequential(nn.Conv2d(128*5,512*5,1,1,bias=False)))]))

        #We add Encoders and Decoders here.
        #a layer has there encoders and decoders.
        #####################################
        self.transformer1_encoder1=nn.Sequential(OrderedDict([
                ('transformer1_encoder1_WK',nn.Sequential(nn.Linear(32,32))),
                ('transformer1_encoder1_WV',nn.Sequential(nn.Linear(32,32))),  
                ('transformer1_encoder1_fc_reduce',nn.Sequential(nn.Conv2d(96,32,1,1,bias=False))),
                ('transformer1_encoder1_fc_rise',nn.Sequential(nn.Conv2d(32,96,1)))
                ]))
        self.transformer1_encoder2=nn.Sequential(OrderedDict([
                ('transformer1_encoder2_WK',nn.Sequential(nn.Linear(32,32))),
                ('transformer1_encoder2_WV',nn.Sequential(nn.Linear(32,32))),  
                ('transformer1_encoder2_fc_reduce',nn.Sequential(nn.Conv2d(96,32,1,1,bias=False))),
                ('transformer1_encoder2_fc_rise',nn.Sequential(nn.Conv2d(32,96,1)))
                ]))
        self.transformer1_encoder3=nn.Sequential(OrderedDict([
                ('transformer1_encoder3_WK',nn.Sequential(nn.Linear(32,32))),
                ('transformer1_encoder3_WV',nn.Sequential(nn.Linear(32,32))),  
                ('transformer1_encoder3_fc_reduce',nn.Sequential(nn.Conv2d(96,32,1,1,bias=False))),
                ('transformer1_encoder3_fc_rise',nn.Sequential(nn.Conv2d(32,96,1)))
                ]))
        ###################################################
        self.transformer1_decoder1=nn.Sequential(OrderedDict([
                ('transformer1_decoder1_WK',nn.Sequential(nn.Linear(32,32))),
                ('transformer1_decoder1_WV',nn.Sequential(nn.Linear(32,32))),  
                ('transformer1_decoder1_fc_reduce',nn.Sequential(nn.Conv2d(96,32,1,1,bias=False))),
                ('transformer1_decoder1_fc_rise',nn.Sequential(nn.Conv2d(32,96,1)))
                ]))
        self.transformer1_decoder2=nn.Sequential(OrderedDict([
                ('transformer1_decoder2_WK',nn.Sequential(nn.Linear(32,32))),
                ('transformer1_decoder2_WV',nn.Sequential(nn.Linear(32,32))),  
                ('transformer1_decoder2_fc_reduce',nn.Sequential(nn.Conv2d(96,32,1,1,bias=False))),
                ('transformer1_decoder2_fc_rise',nn.Sequential(nn.Conv2d(32,96,1)))
                ]))
        ############################################################
        
        #############################
        self.transformer2_encoder1=nn.Sequential(OrderedDict([
                ('transformer2_encoder1_WK',nn.Sequential(nn.Linear(64,64))),
                ('transformer2_encoder1_WV',nn.Sequential(nn.Linear(64,64))),  
                ('transformer2_encoder1_fc_reduce',nn.Sequential(nn.Conv2d(256,64,1,1,bias=False))),
                ('transformer2_encoder1_fc_rise',nn.Sequential(nn.Conv2d(64,256,1)))
                ]))
        self.transformer2_encoder2=nn.Sequential(OrderedDict([
                ('transformer2_encoder2_WK',nn.Sequential(nn.Linear(64,64))),
                ('transformer2_encoder2_WV',nn.Sequential(nn.Linear(64,64))),  
                ('transformer2_encoder2_fc_reduce',nn.Sequential(nn.Conv2d(256,64,1,1,bias=False))),
                ('transformer2_encoder2_fc_rise',nn.Sequential(nn.Conv2d(64,256,1)))
                ]))
        self.transformer2_encoder3=nn.Sequential(OrderedDict([
                ('transformer2_encoder3_WK',nn.Sequential(nn.Linear(64,64))),
                ('transformer2_encoder3_WV',nn.Sequential(nn.Linear(64,64))),  
                ('transformer2_encoder3_fc_reduce',nn.Sequential(nn.Conv2d(256,64,1,1,bias=False))),
                ('transformer2_encoder3_fc_rise',nn.Sequential(nn.Conv2d(64,256,1)))
                ]))
        
        self.transformer2_decoder1=nn.Sequential(OrderedDict([
                ('transformer2_decoder1_WK',nn.Sequential(nn.Linear(64,64))),
                ('transformer2_decoder1_WV',nn.Sequential(nn.Linear(64,64))),  
                ('transformer2_decoder1_fc_reduce',nn.Sequential(nn.Conv2d(256,64,1,1,bias=False))),
                ('transformer2_decoder1_fc_rise',nn.Sequential(nn.Conv2d(64,256,1)))
                ]))
        self.transformer2_decoder2=nn.Sequential(OrderedDict([
                ('transformer2_decoder2_WK',nn.Sequential(nn.Linear(64,64))),
                ('transformer2_decoder2_WV',nn.Sequential(nn.Linear(64,64))),  
                ('transformer2_decoder2_fc_reduce',nn.Sequential(nn.Conv2d(256,64,1,1,bias=False))),
                ('transformer2_decoder2_fc_rise',nn.Sequential(nn.Conv2d(64,256,1)))
                ]))

        ###########################
        self.transformer3_encoder1=nn.Sequential(OrderedDict([
                ('transformer3_encoder1_WK',nn.Sequential(nn.Linear(128,128))),
                ('transformer3_encoder1_WV',nn.Sequential(nn.Linear(128,128))),  
                ('transformer3_encoder1_fc_reduce',nn.Sequential(nn.Conv2d(512,128,1,1,bias=False))),
                ('transformer3_encoder1_fc_rise',nn.Sequential(nn.Conv2d(128,512,1)))
                ]))
        self.transformer3_encoder2=nn.Sequential(OrderedDict([
                ('transformer3_encoder2_WK',nn.Sequential(nn.Linear(128,128))),
                ('transformer3_encoder2_WV',nn.Sequential(nn.Linear(128,128))),  
                ('transformer3_encoder2_fc_reduce',nn.Sequential(nn.Conv2d(512,128,1,1,bias=False))),
                ('transformer3_encoder2_fc_rise',nn.Sequential(nn.Conv2d(128,512,1)))
                ]))
        self.transformer3_encoder3=nn.Sequential(OrderedDict([
                ('transformer3_encoder3_WK',nn.Sequential(nn.Linear(128,128))),
                ('transformer3_encoder3_WV',nn.Sequential(nn.Linear(128,128))),  
                ('transformer3_encoder3_fc_reduce',nn.Sequential(nn.Conv2d(512,128,1,1,bias=False))),
                ('transformer3_encoder3_fc_rise',nn.Sequential(nn.Conv2d(128,512,1)))
                ]))
        
        self.transformer3_decoder1=nn.Sequential(OrderedDict([
                ('transformer3_decoder1_WK',nn.Sequential(nn.Linear(128,128))),
                ('transformer3_decoder1_WV',nn.Sequential(nn.Linear(128,128))),  
                ('transformer3_decoder1_fc_reduce',nn.Sequential(nn.Conv2d(512,128,1,1,bias=False))),
                ('transformer3_decoder1_fc_rise',nn.Sequential(nn.Conv2d(128,512,1)))
                ]))
        self.transformer3_decoder2=nn.Sequential(OrderedDict([
                ('transformer3_decoder2_WK',nn.Sequential(nn.Linear(128,128))),
                ('transformer3_decoder2_WV',nn.Sequential(nn.Linear(128,128))),  
                ('transformer3_decoder2_fc_reduce',nn.Sequential(nn.Conv2d(512,128,1,1,bias=False))),
                ('transformer3_decoder2_fc_rise',nn.Sequential(nn.Conv2d(128,512,1)))
                ]))
        
        #multi branch
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),nn.Linear(512, 2)) for _ in range(K)])
       
        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError('Unkown model format: {:s}'.format(model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers_v.named_children():
            append_params(self.params, module, 'layers_v'+name)
        for name, module in self.layers_i.named_children():
            append_params(self.params, module, 'layers_i'+name)
        for name, module in self.fc.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d'%(k))

        for k, module in enumerate(self.parallel1):
            for name, model in module.named_children():
                if 'pool' in name:
                    continue
                append_params(self.params, model, name+'_%d'%(k))
        for k, module in enumerate(self.parallel2):
            for name, model in module.named_children():
                if 'pool' in name:
                    continue
                append_params(self.params, model, name+'_%d'%(k))
        for k, module in enumerate(self.parallel3):
            for name, model in module.named_children():
                if 'pool' in name:
                    continue
                append_params(self.params, model, name+'_%d'%(k))
        
        for k, module in enumerate(self.parallel1_skconv):
            for name, model in module.named_children():
                if 'pool' in name:
                    continue
                append_params(self.params, model, name+'_%d'%(k))
            
        for k, module in enumerate(self.parallel2_skconv):
            for name, model in module.named_children():
                if 'pool' in name:
                    continue
                append_params(self.params, model, name+'_%d'%(k))

        for k, module in enumerate(self.parallel3_skconv):
            for name, model in module.named_children():
                if 'pool' in name:
                    continue
                append_params(self.params, model, name+'_%d'%(k))
        #############################################################
        # the last ALL modeles
        for name, module in self.ensemble1_skconv.named_children():
            if 'pool' in name:
                continue
            append_params(self.params, module, name)
        for name, module in self.ensemble2_skconv.named_children():
            if 'pool' in name:
                continue
            append_params(self.params, module, name)
        for name, module in self.ensemble3_skconv.named_children():
            if 'pool' in name:
                continue
            append_params(self.params, module, name)

        #add Trans 
        for name, module in self.transformer1_encoder1.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer1_encoder2.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer1_encoder3.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer1_decoder1.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer1_decoder2.named_children():
            append_params(self.params, module, name)
        # for name, module in self.transformer1_FFN.named_children():
        #     append_params(self.params, module, name)
        for name, module in self.transformer2_encoder1.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer2_encoder2.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer2_encoder3.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer2_decoder1.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer2_decoder2.named_children():
            append_params(self.params, module, name)
        # for name, module in self.transformer2_FFN.named_children():
        #     append_params(self.params, module, name)
        for name, module in self.transformer3_encoder1.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer3_encoder2.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer3_encoder3.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer3_decoder1.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer3_decoder2.named_children():
            append_params(self.params, module, name)
        # for name, module in self.transformer3_FFN.named_children():
        #     append_params(self.params, module, name)
       
        
    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        print('get_learnable_params',params.keys())
        return params
    
    def get_all_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            params[k] = p
        return params

    #self-atttention
    def Transformer_feature_layer1_vis(self,x):
        x_1=self.transformer1_encoder1[2](x)   
        batch,dim,w,h = x_1.shape 
        x_1=x_1.permute(0,2,3,1)   
        x_k=x_1.reshape(batch,w*h,dim)  
        x_v=x_1.reshape(batch,w*h,dim)  
        x_q=x_1.reshape(batch,w*h,dim)

        w_k=self.transformer1_encoder1[0](x_k)   
        w_k = F.normalize(w_k, p=2, dim=-1)     
        w_k=w_k.permute(0,1,2)    

        w_q = self.transformer1_encoder1[0](x_q)  
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(0,2,1)   
        
        dot_prod = torch.bmm(w_q, w_k) 
        affinity = F.softmax(dot_prod*30, dim=-1) 

        w_v=self.transformer1_encoder1[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v=w_v.permute(0,2,1)   
    
        output = torch.bmm(affinity, w_v)  
        output=output.reshape(batch,dim,w,h)  
        output=self.transformer1_encoder1[3](output)  
        x=x+output   
        return x
        

    def Transformer_feature_layer1_inf(self,x):
        x_1=self.transformer1_encoder2[2](x)   
        batch,dim,w,h = x_1.shape 
        x_1=x_1.permute(0,2,3,1)   
        x_k=x_1.reshape(batch,w*h,dim)  
        x_v=x_1.reshape(batch,w*h,dim)  
        x_q=x_1.reshape(batch,w*h,dim)

        w_k=self.transformer1_encoder2[0](x_k)   
        w_k = F.normalize(w_k, p=2, dim=-1)     
        w_k=w_k.permute(0,1,2)  

        w_q = self.transformer1_encoder2[0](x_q)  
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(0,2,1)   
        
        dot_prod = torch.bmm(w_q, w_k) 
        affinity = F.softmax(dot_prod*30, dim=-1) 

        w_v=self.transformer1_encoder2[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v=w_v.permute(0,2,1)   
    
        output = torch.bmm(affinity, w_v) 
        output=output.reshape(batch,dim,w,h)  
        output=self.transformer1_encoder2[3](output)  
        x=x+output   
        return x

    def Transformer_feature_layer1_agg(self,x):
        x_1=self.transformer1_encoder3[2](x)   
        batch,dim,w,h = x_1.shape 
        x_1=x_1.permute(0,2,3,1)   
        x_k=x_1.reshape(batch,w*h,dim)  
        x_v=x_1.reshape(batch,w*h,dim)  
        x_q=x_1.reshape(batch,w*h,dim)

        w_k=self.transformer1_encoder3[0](x_k)   
        w_k = F.normalize(w_k, p=2, dim=-1)     
        w_k=w_k.permute(0,1,2)    

        w_q = self.transformer1_encoder3[0](x_q)  
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(0,2,1)   
        
        dot_prod = torch.bmm(w_q, w_k) 
        affinity = F.softmax(dot_prod*30, dim=-1) 

        w_v=self.transformer1_encoder3[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v=w_v.permute(0,2,1)    
    
        output = torch.bmm(affinity, w_v)  
        output=output.reshape(batch,dim,w,h)  
        output=self.transformer1_encoder3[3](output)  
        x=x+output   
        return x
    
    def Transformer_feature_layer2_vis(self,x):
        x_1=self.transformer2_encoder1[2](x)   
        batch,dim,w,h = x_1.shape    
        x_1=x_1.permute(0,2,3,1)
        x_k=x_1.reshape(batch,w*h,dim)  
        x_v=x_1.reshape(batch,w*h,dim)  
        x_q=x_1.reshape(batch,w*h,dim)

        w_k=self.transformer2_encoder1[0](x_k)   
        w_k = F.normalize(w_k, p=2, dim=-1)     
        w_k=w_k.permute(0,1,2)    

        w_q = self.transformer2_encoder1[0](x_q)  
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(0,2,1)   
        
        dot_prod = torch.bmm(w_q, w_k) 
        affinity = F.softmax(dot_prod*30, dim=-1) 

        w_v=self.transformer2_encoder1[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v=w_v.permute(0,2,1)    
    
        output = torch.bmm(affinity, w_v)   
        output=output.reshape(batch,dim,w,h)  
        output=self.transformer2_encoder1[3](output)     
        x=x+output
        return x

    def Transformer_feature_layer2_inf(self,x):
        x_1=self.transformer2_encoder2[2](x)   
        batch,dim,w,h = x_1.shape    
        x_1=x_1.permute(0,2,3,1)
        x_k=x_1.reshape(batch,w*h,dim)  
        x_v=x_1.reshape(batch,w*h,dim)  
        x_q=x_1.reshape(batch,w*h,dim)

        w_k=self.transformer2_encoder2[0](x_k)   
        w_k = F.normalize(w_k, p=2, dim=-1)     
        w_k=w_k.permute(0,1,2)    

        w_q = self.transformer2_encoder2[0](x_q)  
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(0,2,1)   
        
        dot_prod = torch.bmm(w_q, w_k) 
        affinity = F.softmax(dot_prod*30, dim=-1) 

        w_v=self.transformer2_encoder2[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v=w_v.permute(0,2,1)    
    
        output = torch.bmm(affinity, w_v)   
        output=output.reshape(batch,dim,w,h)  
        output=self.transformer2_encoder2[3](output)     
        x=x+output
        return x

    def Transformer_feature_layer2_agg(self,x):
        x_1=self.transformer2_encoder3[2](x)   
        batch,dim,w,h = x_1.shape    
        x_1=x_1.permute(0,2,3,1)
        x_k=x_1.reshape(batch,w*h,dim)  
        x_v=x_1.reshape(batch,w*h,dim)  
        x_q=x_1.reshape(batch,w*h,dim)

        w_k=self.transformer2_encoder3[0](x_k)   
        w_k = F.normalize(w_k, p=2, dim=-1)     
        w_k=w_k.permute(0,1,2)    

        w_q = self.transformer2_encoder3[0](x_q)  
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(0,2,1)   
        
        dot_prod = torch.bmm(w_q, w_k) 
        affinity = F.softmax(dot_prod*30, dim=-1) 

        w_v=self.transformer2_encoder3[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v=w_v.permute(0,2,1)    
    
        output = torch.bmm(affinity, w_v)   
        output=output.reshape(batch,dim,w,h)  
        output=self.transformer2_encoder3[3](output)     
        x=x+output
        return x
        

    def Transformer_feature_layer3_vis(self,x):
        x_1=self.transformer3_encoder1[2](x)   
        batch,dim,w,h = x_1.shape    
        x_1=x_1.permute(0,2,3,1)
        x_k=x_1.reshape(batch,w*h,dim)  
        x_v=x_1.reshape(batch,w*h,dim)  
        x_q=x_1.reshape(batch,w*h,dim)

        w_k=self.transformer3_encoder1[0](x_k)   
        w_k = F.normalize(w_k, p=2, dim=-1)     
        w_k=w_k.permute(0,1,2)    

        w_q = self.transformer3_encoder1[0](x_q)  
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(0,2,1)   
        
        dot_prod = torch.bmm(w_q, w_k) 
        affinity = F.softmax(dot_prod*30, dim=-1) 

        w_v=self.transformer3_encoder1[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v=w_v.permute(0,2,1)    
    
        output = torch.bmm(affinity, w_v)   
        output=output.reshape(batch,dim,w,h)  
        output=self.transformer3_encoder1[3](output)     
        x=x+output  
        return x

    def Transformer_feature_layer3_inf(self,x):
        x_1=self.transformer3_encoder2[2](x)   
        batch,dim,w,h = x_1.shape    
        x_1=x_1.permute(0,2,3,1)
        x_k=x_1.reshape(batch,w*h,dim)  
        x_v=x_1.reshape(batch,w*h,dim)  
        x_q=x_1.reshape(batch,w*h,dim)

        w_k=self.transformer3_encoder2[0](x_k)   
        w_k = F.normalize(w_k, p=2, dim=-1)     
        w_k=w_k.permute(0,1,2)    

        w_q = self.transformer3_encoder2[0](x_q)  
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(0,2,1)   
        
        dot_prod = torch.bmm(w_q, w_k) 
        affinity = F.softmax(dot_prod*30, dim=-1) 

        w_v=self.transformer3_encoder2[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v=w_v.permute(0,2,1)    
    
        output = torch.bmm(affinity, w_v)   
        output=output.reshape(batch,dim,w,h)  
        output=self.transformer3_encoder2[3](output)     
        x=x+output  
        return x

    def Transformer_feature_layer3_agg(self,x):
        x_1=self.transformer3_encoder3[2](x)   
        batch,dim,w,h = x_1.shape    
        x_1=x_1.permute(0,2,3,1)
        x_k=x_1.reshape(batch,w*h,dim)  
        x_v=x_1.reshape(batch,w*h,dim)  
        x_q=x_1.reshape(batch,w*h,dim)

        w_k=self.transformer3_encoder3[0](x_k)   
        w_k = F.normalize(w_k, p=2, dim=-1)     
        w_k=w_k.permute(0,1,2)    

        w_q = self.transformer3_encoder3[0](x_q)  
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(0,2,1)   
        
        dot_prod = torch.bmm(w_q, w_k) 
        affinity = F.softmax(dot_prod*30, dim=-1) 

        w_v=self.transformer3_encoder3[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v=w_v.permute(0,2,1)    
    
        output = torch.bmm(affinity, w_v)   
        output=output.reshape(batch,dim,w,h)  
        output=self.transformer3_encoder3[3](output)     
        x=x+output  
        return x
       
        
    #Cross Attention
    def CrossAttention_layer1_visagg(self,x,V):
        x_1=self.transformer1_decoder1[2](x)   
        batch,dim,w,h = x_1.shape    
        V_1=self.transformer1_decoder1[2](V) 
        x_1=x_1.permute(0,2,3,1)
        V_1=V_1.permute(0,2,3,1)
        x_k=x_1.reshape(batch,w*h,dim)  
        x_v=x_1.reshape(batch,w*h,dim)  
        V_q=V_1.reshape(batch,w*h,dim)

        w_k=self.transformer1_decoder1[0](x_k)   
        w_k = F.normalize(w_k, p=2, dim=-1) 
        w_k=w_k.permute(0,1,2)    

        w_q = self.transformer1_decoder1[0](V_q)  
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q=w_q.permute(0,2,1)   

        dot_prod = torch.bmm(w_q, w_k) 
        affinity = F.softmax(dot_prod*30, dim=-1) 

        w_v=self.transformer1_decoder1[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v=w_v.permute(0,2,1)    
    
        output = torch.bmm(affinity, w_v) 
        output=output.reshape(batch,dim,w,h)  
        output=self.transformer1_decoder1[3](output)    
        x=x+output
        return x

    def CrossAttention_layer1_infagg(self,x,V):
        x_1=self.transformer1_decoder2[2](x)   
        batch,dim,w,h = x_1.shape    
        V_1=self.transformer1_decoder2[2](V) 
        x_1=x_1.permute(0,2,3,1)
        V_1=V_1.permute(0,2,3,1)
        x_k=x_1.reshape(batch,w*h,dim)  
        x_v=x_1.reshape(batch,w*h,dim)  
        V_q=V_1.reshape(batch,w*h,dim)

        w_k=self.transformer1_decoder2[0](x_k)   
        w_k = F.normalize(w_k, p=2, dim=-1) 
        w_k=w_k.permute(0,1,2)    

        w_q = self.transformer1_decoder2[0](V_q)  
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q=w_q.permute(0,2,1)   

        dot_prod = torch.bmm(w_q, w_k) 
        affinity = F.softmax(dot_prod*30, dim=-1) 

        w_v=self.transformer1_decoder2[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v=w_v.permute(0,2,1)    
    
        output = torch.bmm(affinity, w_v) 
        output=output.reshape(batch,dim,w,h)  
        output=self.transformer1_decoder2[3](output)    
        x=x+output
        return x
        
    
    def CrossAttention_layer2_visagg(self,x,V):
        x_1=self.transformer2_decoder1[2](x)   
        batch,dim,w,h = x_1.shape    
        V_1=self.transformer2_decoder1[2](V) 
        x_1=x_1.permute(0,2,3,1)
        V_1=V_1.permute(0,2,3,1)
        x_k=x_1.reshape(batch,w*h,dim)  
        x_v=x_1.reshape(batch,w*h,dim)  
        V_q=V_1.reshape(batch,w*h,dim)

        w_k=self.transformer2_decoder1[0](x_k)   
        w_k = F.normalize(w_k, p=2, dim=-1) 
        w_k=w_k.permute(0,1,2)    

        w_q = self.transformer2_decoder1[0](V_q)  
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q=w_q.permute(0,2,1)   

        dot_prod = torch.bmm(w_q, w_k) 
        affinity = F.softmax(dot_prod*30, dim=-1) 

        w_v=self.transformer2_decoder1[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v=w_v.permute(0,2,1)    
    
        output = torch.bmm(affinity, w_v) 
        output=output.reshape(batch,dim,w,h)  
        output=self.transformer2_decoder1[3](output)    
        x=x+output
        return x

    def CrossAttention_layer2_infagg(self,x,V):
        x_1=self.transformer2_decoder2[2](x)   
        batch,dim,w,h = x_1.shape    
        V_1=self.transformer2_decoder2[2](V) 
        x_1=x_1.permute(0,2,3,1)
        V_1=V_1.permute(0,2,3,1)
        x_k=x_1.reshape(batch,w*h,dim)  
        x_v=x_1.reshape(batch,w*h,dim)  
        V_q=V_1.reshape(batch,w*h,dim)

        w_k=self.transformer2_decoder2[0](x_k)   
        w_k = F.normalize(w_k, p=2, dim=-1) 
        w_k=w_k.permute(0,1,2)    

        w_q = self.transformer2_decoder2[0](V_q)  
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q=w_q.permute(0,2,1)   

        dot_prod = torch.bmm(w_q, w_k) 
        affinity = F.softmax(dot_prod*30, dim=-1) 

        w_v=self.transformer2_decoder2[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v=w_v.permute(0,2,1)    
    
        output = torch.bmm(affinity, w_v) 
        output=output.reshape(batch,dim,w,h)  
        output=self.transformer2_decoder2[3](output)    
        x=x+output
        return x
       
       
    def CrossAttention_layer3_visagg(self,x,V):
        x_1=self.transformer3_decoder1[2](x)   
        batch,dim,w,h = x_1.shape    
        V_1=self.transformer3_decoder1[2](V) 
        x_1=x_1.permute(0,2,3,1)
        V_1=V_1.permute(0,2,3,1)
        x_k=x_1.reshape(batch,w*h,dim)  
        x_v=x_1.reshape(batch,w*h,dim)  
        V_q=V_1.reshape(batch,w*h,dim)

        w_k=self.transformer3_decoder1[0](x_k)   
        w_k = F.normalize(w_k, p=2, dim=-1) 
        w_k=w_k.permute(0,1,2)    

        w_q = self.transformer3_decoder1[0](V_q)  
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q=w_q.permute(0,2,1)   

        dot_prod = torch.bmm(w_q, w_k) 
        affinity = F.softmax(dot_prod*30, dim=-1) 

        w_v=self.transformer3_decoder1[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v=w_v.permute(0,2,1)    
    
        output = torch.bmm(affinity, w_v) 
        output=output.reshape(batch,dim,w,h)  
        output=self.transformer3_decoder1[3](output)    
        x=x+output
        return x
        
    def CrossAttention_layer3_infagg(self,x,V):
        x_1=self.transformer3_decoder2[2](x)   
        batch,dim,w,h = x_1.shape    
        V_1=self.transformer3_decoder2[2](V) 
        x_1=x_1.permute(0,2,3,1)
        V_1=V_1.permute(0,2,3,1)
        x_k=x_1.reshape(batch,w*h,dim)  
        x_v=x_1.reshape(batch,w*h,dim)  
        V_q=V_1.reshape(batch,w*h,dim)

        w_k=self.transformer3_decoder2[0](x_k)   
        w_k = F.normalize(w_k, p=2, dim=-1) 
        w_k=w_k.permute(0,1,2)    

        w_q = self.transformer3_decoder2[0](V_q)  
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q=w_q.permute(0,2,1)   

        dot_prod = torch.bmm(w_q, w_k) 
        affinity = F.softmax(dot_prod*30, dim=-1) 

        w_v=self.transformer3_decoder2[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v=w_v.permute(0,2,1)    
    
        output = torch.bmm(affinity, w_v) 
        output=output.reshape(batch,dim,w,h)  
        output=self.transformer3_decoder2[3](output)    
        x=x+output
        return x
    
    def forward(self, x1, x2, k=0, in_layer='conv1', out_layer='fc6'):
        # forward model from in_layer to out_layer
        run = False
        x = x1
        for (name_v, module_v),(name_i, module_i) in zip(self.layers_v.named_children(),self.layers_i.named_children()):
            output=[] 
            if name_v == in_layer:
                run = True
            if run:
                if name_v in ['conv1','conv2','conv3']:
                    if name_v == 'conv1':
                        #fast_motion
                        output=[]
                        x1_fm = self.parallel1[0](x1)
                        x2_fm = self.parallel1[0](x2)
                        batch_size=x1.size(0)
                        output.append(x1_fm)    
                        output.append(x2_fm)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.parallel1_skconv[0](U)
                        a_b=a_b.reshape(batch_size,2,96,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)     
                        a_b=list(a_b.chunk(2,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,96,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))  
                        V_fm=reduce(lambda x,y:x+y,V)              
                        del x1_fm,x2_fm
                        output=[]    #you should clear your list

                        #occlusion
                        x1_occ = self.parallel1[1](x1)
                        x2_occ = self.parallel1[1](x2)
                        batch_size=x1.size(0)
                        output.append(x1_occ)    #the part of fusion
                        output.append(x2_occ)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.parallel1_skconv[1](U)
                        a_b=a_b.reshape(batch_size,2,96,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)    
                     
                        a_b=list(a_b.chunk(2,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,96,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))   
                        V_occ=reduce(lambda x,y:x+y,V)              
                        del x1_occ,x2_occ
                        output=[]

                        #size change
                        x1_sc = self.parallel1[2](x1)
                        x2_sc = self.parallel1[2](x2)
                        batch_size=x1.size(0)
                        output.append(x1_sc)    
                        output.append(x2_sc)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.parallel1_skconv[2](U)
                        a_b=a_b.reshape(batch_size,2,96,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)     
                        a_b=list(a_b.chunk(2,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,96,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))   
                        V_sc=reduce(lambda x,y:x+y,V)            
                        del x1_sc,x2_sc
                        output=[]

                        #tc
                        x1_tc = self.parallel1[3](x1)
                        x2_tc = self.parallel1[3](x2)
                        batch_size=x1.size(0)
                        output.append(x1_tc)    
                        output.append(x2_tc)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.parallel1_skconv[3](U)
                        a_b=a_b.reshape(batch_size,2,96,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)    
                        a_b=list(a_b.chunk(2,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,96,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))   
                        V_tc=reduce(lambda x,y:x+y,V)              
                        del x1_tc,x2_tc
                        output=[]
                        
                        #ill
                        x1_ill = self.parallel1[4](x1)
                        x2_ill = self.parallel1[4](x2)
                        batch_size=x1.size(0)
                        output.append(x1_ill)    
                        output.append(x2_ill)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.parallel1_skconv[4](U)
                        a_b=a_b.reshape(batch_size,2,96,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)    
                        a_b=list(a_b.chunk(2,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,96,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))   
                        V_ill=reduce(lambda x,y:x+y,V)              
                        del x1_ill,x2_ill
                        output=[]

                        #input to ensemble for x1: ALL1
                        batch_size=x1.size(0)
                        output.append(V_fm)    
                        output.append(V_occ)
                        output.append(V_sc)
                        output.append(V_tc)
                        output.append(V_ill)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.ensemble1_skconv(U)
                        a_b=a_b.reshape(batch_size,5,96,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)    
                        a_b=list(a_b.chunk(5,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,96,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))   
                        V=reduce(lambda x,y:x+y,V)              
                        del V_fm,V_occ,V_sc,V_tc,V_ill,a_b
                        output=[]
                        torch.cuda.empty_cache()
   
                    elif name_v == 'conv2':
                        #fast_motion
                        output=[]
                        x1_fm = self.parallel2[0](x1)
                        x2_fm = self.parallel2[0](x2)
                        batch_size=x1.size(0)
                        output.append(x1_fm)   
                        output.append(x2_fm)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.parallel2_skconv[0](U)
                        a_b=a_b.reshape(batch_size,2,256,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)    
                        a_b=list(a_b.chunk(2,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,256,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))  
                        V_fm=reduce(lambda x,y:x+y,V)         
                        del x1_fm,x2_fm
                        output=[]

                        #occlusion
                        x1_occ = self.parallel2[1](x1)
                        x2_occ = self.parallel2[1](x2)
                        batch_size=x1.size(0)
                        output.append(x1_occ)   
                        output.append(x2_occ)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.parallel2_skconv[1](U)
                        a_b=a_b.reshape(batch_size,2,256,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)     
                        a_b=list(a_b.chunk(2,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,256,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))   
                        V_occ=reduce(lambda x,y:x+y,V)              
                        del x1_occ,x2_occ
                        output=[]

                        #size change
                        x1_sc = self.parallel2[2](x1)
                        x2_sc = self.parallel2[2](x2)
                        batch_size=x1.size(0)
                        output.append(x1_sc)    
                        output.append(x2_sc)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.parallel2_skconv[2](U)
                        a_b=a_b.reshape(batch_size,2,256,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)     
                        a_b=list(a_b.chunk(2,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,256,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))  
                        V_sc=reduce(lambda x,y:x+y,V)              
                        del x1_sc,x2_sc
                        output=[]

                        #tc
                        x1_tc = self.parallel2[3](x1)
                        x2_tc = self.parallel2[3](x2)
                        batch_size=x1.size(0)
                        output.append(x1_tc)   
                        output.append(x2_tc)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.parallel2_skconv[3](U)
                        a_b=a_b.reshape(batch_size,2,256,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)     
                        a_b=list(a_b.chunk(2,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,256,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))   
                        V_tc=reduce(lambda x,y:x+y,V)            
                        del x1_tc,x2_tc
                        output=[]
                        
                        #ill
                        x1_ill = self.parallel2[4](x1)
                        x2_ill = self.parallel2[4](x2)
                        batch_size=x1.size(0)
                        output.append(x1_ill)   
                        output.append(x2_ill)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.parallel2_skconv[4](U)
                        a_b=a_b.reshape(batch_size,2,256,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)     
                        a_b=list(a_b.chunk(2,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,256,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))   
                        V_ill=reduce(lambda x,y:x+y,V)             
                        del x1_ill,x2_ill
                        output=[]

                        #input to ensemble for x1: ALL1
                        batch_size=x1.size(0)
                        output.append(V_fm)   
                        output.append(V_occ)
                        output.append(V_sc)
                        output.append(V_tc)
                        output.append(V_ill)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.ensemble2_skconv(U)
                        a_b=a_b.reshape(batch_size,5,256,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)     
                        a_b=list(a_b.chunk(5,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,256,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))   
                        V=reduce(lambda x,y:x+y,V)             
                        del V_fm,V_occ,V_sc,V_tc,V_ill,a_b
                        output=[]
                        torch.cuda.empty_cache()
                       
                    else:
                        output=[]
                        x1_fm = self.parallel3[0](x1)
                        x2_fm = self.parallel3[0](x2)
                        batch_size=x1.size(0)
                        output.append(x1_fm)    
                        output.append(x2_fm)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.parallel3_skconv[0](U)
                        a_b=a_b.reshape(batch_size,2,512,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)    
                        a_b=list(a_b.chunk(2,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,512,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))   
                        V_fm=reduce(lambda x,y:x+y,V)            
                        del x1_fm,x2_fm
                        output=[]

                        #occlusion
                        x1_occ = self.parallel3[1](x1)
                        x2_occ = self.parallel3[1](x2)
                        batch_size=x1.size(0)
                        output.append(x1_occ)   
                        output.append(x2_occ)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.parallel3_skconv[1](U)
                        a_b=a_b.reshape(batch_size,2,512,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)     
                        a_b=list(a_b.chunk(2,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,512,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))   
                        V_occ=reduce(lambda x,y:x+y,V)            
                        del x1_occ,x2_occ
                        output=[]

                        #size change
                        x1_sc = self.parallel3[2](x1)
                        x2_sc = self.parallel3[2](x2)
                        batch_size=x1.size(0)
                        output.append(x1_sc)    
                        output.append(x2_sc)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.parallel3_skconv[2](U)
                        a_b=a_b.reshape(batch_size,2,512,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)    
                        a_b=list(a_b.chunk(2,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,512,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))   
                        V_sc=reduce(lambda x,y:x+y,V)              
                        del x1_sc,x2_sc
                        output=[]

                        #tc
                        x1_tc = self.parallel3[3](x1)
                        x2_tc = self.parallel3[3](x2)
                        batch_size=x1.size(0)
                        output.append(x1_tc)    
                        output.append(x2_tc)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.parallel3_skconv[3](U)
                        a_b=a_b.reshape(batch_size,2,512,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)     
                        a_b=list(a_b.chunk(2,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,512,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))  
                        V_tc=reduce(lambda x,y:x+y,V)              
                        del x1_tc,x2_tc
                        output=[]
                        
                        #ill
                        x1_ill = self.parallel3[4](x1)
                        x2_ill = self.parallel3[4](x2)
                        batch_size=x1.size(0)
                        output.append(x1_ill)    
                        output.append(x2_ill)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.parallel3_skconv[4](U)
                        a_b=a_b.reshape(batch_size,2,512,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)    
                        a_b=list(a_b.chunk(2,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,512,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))   
                        V_ill=reduce(lambda x,y:x+y,V)             
                        del x1_ill,x2_ill
                        output=[]

                        #input to ensemble for x1: ALL1
                        batch_size=x1.size(0)
                        output.append(V_fm)    
                        output.append(V_occ)
                        output.append(V_sc)
                        output.append(V_tc)
                        output.append(V_ill)
                        U=reduce(lambda x,y:x+y,output)
                        a_b=self.ensemble3_skconv(U)
                        a_b=a_b.reshape(batch_size,5,512,-1) 
                        a_b=nn.Softmax(dim=1)(a_b)     
                      
                        a_b=list(a_b.chunk(5,dim=1))
                        a_b=list(map(lambda x:x.reshape(batch_size,512,1,1),a_b)) 
                        V=list(map(lambda x,y:x*y,output,a_b))   
                        V=reduce(lambda x,y:x+y,V)              
                        del V_fm,V_occ,V_sc,V_tc,V_ill,a_b
                        output=[]
                        torch.cuda.empty_cache()
                    if name_v == 'conv1':
                        x1=module_v(x1)
                        x2=module_i(x2)
                        x1=self.Transformer_feature_layer1_vis(x1)
                        V=self.Transformer_feature_layer1_agg(V)
                        x2=self.Transformer_feature_layer1_inf(x2)
                        x1=self.CrossAttention_layer1_visagg(x1, V)
                        x2=self.CrossAttention_layer1_infagg(x2,V)
                    elif name_v=='conv2':
                        x1=module_v(x1)
                        x2=module_i(x2)
                        x1=self.Transformer_feature_layer2_vis(x1)
                        V=self.Transformer_feature_layer2_agg(V)
                        x2=self.Transformer_feature_layer2_inf(x2)
                        x1=self.CrossAttention_layer2_visagg(x1, V)
                        x2=self.CrossAttention_layer2_infagg(x2,V)
                    else:
                        x1=module_v(x1)
                        x2=module_i(x2)
                        x1=self.Transformer_feature_layer3_vis(x1)
                        V=self.Transformer_feature_layer3_agg(V)
                        x2=self.Transformer_feature_layer3_inf(x2)
                        x1=self.CrossAttention_layer3_visagg(x1, V)
                        x2=self.CrossAttention_layer3_infagg(x2,V)
                        x = torch.cat((x1,x2),1)
                        x = x.contiguous().view(x.size(0), -1)
                
                if name_v == out_layer:
                    return x
        x = self.fc(x)
        x = self.branches[k](x)
        if out_layer=='fc6':
            return x
        elif out_layer=='fc6_softmax':
            return F.softmax(x, dim=1)

    def load_model(self, model_path):
        states = torch.load(model_path)
        try:
            print('load LanYangYang model.')
            self.layers_v.load_state_dict(states['layers_v'])
            self.layers_i.load_state_dict(states['layers_i'])
            self.fc.load_state_dict(states['fc'])
            self.parallel1.load_state_dict(states['parallel1'])
            self.parallel2.load_state_dict(states['parallel2'])
            self.parallel3.load_state_dict(states['parallel3'])
            self.parallel1_skconv.load_state_dict(states['parallel1_skconv'])
            self.parallel2_skconv.load_state_dict(states['parallel2_skconv'])
            self.parallel3_skconv.load_state_dict(states['parallel3_skconv'])
            self.ensemble1_skconv.load_state_dict(states['ensemble1_skconv'])
            self.ensemble2_skconv.load_state_dict(states['ensemble2_skconv'])
            self.ensemble3_skconv.load_state_dict(states['ensemble3_skconv'])
            
            self.transformer1_encoder1.load_state_dict(states['transformer1_encoder1'])
            self.transformer1_encoder2.load_state_dict(states['transformer1_encoder2'])
            self.transformer1_encoder3.load_state_dict(states['transformer1_encoder3'])
            self.transformer1_decoder1.load_state_dict(states['transformer1_decoder1'])
            self.transformer1_decoder2.load_state_dict(states['transformer1_decoder2'])

            self.transformer2_encoder1.load_state_dict(states['transformer2_encoder1'])
            self.transformer2_encoder2.load_state_dict(states['transformer2_encoder2'])
            self.transformer2_encoder3.load_state_dict(states['transformer2_encoder3'])
            self.transformer2_decoder1.load_state_dict(states['transformer2_decoder1'])
            self.transformer2_decoder2.load_state_dict(states['transformer2_decoder2'])

            self.transformer3_encoder1.load_state_dict(states['transformer3_encoder1'])
            self.transformer3_encoder2.load_state_dict(states['transformer3_encoder2'])
            self.transformer3_encoder3.load_state_dict(states['transformer3_encoder3'])
            self.transformer3_decoder1.load_state_dict(states['transformer3_decoder1'])
            self.transformer3_decoder2.load_state_dict(states['transformer3_decoder2'])
            print('load LanYangYang model end.')
        except:
            print('load LanYangYang model error!')
            print('load VID model.')
            shared_layers = states['shared_layers']
            pretrain_parm = OrderedDict()
            pretrain_parm['layers_v'] = OrderedDict()
            pretrain_parm['layers_i'] = OrderedDict()
            pretrain_parm['fc'] = OrderedDict()
            for k,v in shared_layers.items():
                if 'conv' in k:
                    pretrain_parm['layers_v'][k] = v
                    pretrain_parm['layers_i'][k] = v
                elif k == 'fc4.0.weight':
                    pretrain_parm['fc'][k] = torch.cat((v,v),1)
                else:
                    pretrain_parm['fc'][k] = v
            self.layers_v.load_state_dict(pretrain_parm['layers_v'])
            self.layers_i.load_state_dict(pretrain_parm['layers_i'])
            self.fc.load_state_dict(pretrain_parm['fc'])
            print('load VID model end.')

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers_v[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers_v[i][0].bias.data = torch.from_numpy(bias[:, 0])
            self.layers_i[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers_i[i][0].bias.data = torch.from_numpy(bias[:, 0])


class BCELoss(nn.Module):
    def forward(self, pos_score, neg_score, average=True):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]

        loss = pos_loss.sum() + neg_loss.sum()
        if average:
            loss /= (pos_loss.size(0) + neg_loss.size(0))
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()
        acc = (pos_correct + neg_correct) / (pos_score.size(0) + neg_score.size(0) + 1e-8)
        return acc.item()


class Precision():
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)
        return prec.item()

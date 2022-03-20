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
'''
   model stage2 is MDNet(backbone)+ (Five challenge branches + SKNet ensemble)  
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
        #the first branch to fuse
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
        
        #filter the five challenge information 
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

        #fc6
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),nn.Linear(512, 2)) for _ in range(K)])
        #initial parameters
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        #add new branch
        for m in self.parallel1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in self.parallel2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in self.parallel3.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in self.parallel1_skconv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.constant_(m.bias, 0.1)
        for m in self.parallel2_skconv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.constant_(m.bias, 0.1)
        for m in self.parallel3_skconv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.constant_(m.bias, 0.1)
        
        # the last SK_ALL_filter modeles
        for m in self.ensemble1_skconv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.constant_(m.bias, 0.1)
        for m in self.ensemble2_skconv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.constant_(m.bias, 0.1)
        for m in self.ensemble3_skconv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.constant_(m.bias, 0.1)
        
        #end the model and load the parameters
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
        #add new architecher
        for name, module in self.parallel1.named_children():
            append_params(self.params, module, name)
        for name,module in self.parallel1_skconv.named_children():
            if 'pool' in name:
                continue
            append_params(self.params,module,name)
       
        for name, module in self.parallel2.named_children():
            append_params(self.params, module, name)
    
        for name,module in self.parallel2_skconv.named_children():
            if 'pool' in name:
                continue
            append_params(self.params,module,name)
        
        for name, module in self.parallel3.named_children():
            append_params(self.params, module, name)
    
        for name,module in self.parallel3_skconv.named_children():
            if 'pool' in name:
                continue
            append_params(self.params,module,name)
        
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
                        output.append(x1_occ)    
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
                       
                x1 = module_v(x1)
                x1=x1+V
                x2 = module_i(x2)
                x2=x2+V
                if name_v == 'conv3':
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
        print('load LanYangYang model.')
        self.layers_v.load_state_dict(states['layers_v'])
        self.layers_i.load_state_dict(states['layers_i'])
        self.fc.load_state_dict(states['fc'])
        #load parallele1 branches
        self.parallel1[0].load_state_dict(torch.load('./models/GTOT_FM.pth')['parallel1'])
        self.parallel1_skconv[0].load_state_dict(torch.load('./models/GTOT_FM.pth')['paralle1_skconv'])
        self.parallel1[1].load_state_dict(torch.load('./models/GTOT_OCC.pth')['parallel1']) 
        self.parallel1_skconv[1].load_state_dict(torch.load('./models/GTOT_OCC.pth')['paralle1_skconv']) 
        self.parallel1[2].load_state_dict(torch.load('./models/GTOT_SC.pth')['parallel1'])
        self.parallel1_skconv[2].load_state_dict(torch.load('./models/GTOT_SC.pth')['paralle1_skconv'])
        self.parallel1[3].load_state_dict(torch.load('./models/GTOT_TC.pth')['parallel1'])
        self.parallel1_skconv[3].load_state_dict(torch.load('./models/GTOT_TC.pth')['paralle1_skconv'])
        self.parallel1[4].load_state_dict(torch.load('./models/GTOT_ILL.pth')['parallel1'])
        self.parallel1_skconv[4].load_state_dict(torch.load('./models/GTOT_ILL.pth')['paralle1_skconv'])
        # parallel2 branches   
        self.parallel2[0].load_state_dict(torch.load('./models/GTOT_FM.pth')['parallel2'])
        self.parallel2_skconv[0].load_state_dict(torch.load('./models/GTOT_FM.pth')['paralle2_skconv'])
        self.parallel2[1].load_state_dict(torch.load('./models/GTOT_OCC.pth')['parallel2']) 
        self.parallel2_skconv[1].load_state_dict(torch.load('./models/GTOT_OCC.pth')['paralle2_skconv']) 
        self.parallel2[2].load_state_dict(torch.load('./models/GTOT_SC.pth')['parallel2'])
        self.parallel2_skconv[2].load_state_dict(torch.load('./models/GTOT_SC.pth')['paralle2_skconv'])
        self.parallel2[3].load_state_dict(torch.load('./models/GTOT_TC.pth')['parallel2'])
        self.parallel2_skconv[3].load_state_dict(torch.load('./models/GTOT_TC.pth')['paralle2_skconv'])
        self.parallel2[4].load_state_dict(torch.load('./models/GTOT_ILL.pth')['parallel2'])
        self.parallel2_skconv[4].load_state_dict(torch.load('./models/GTOT_ILL.pth')['paralle2_skconv'])
        # parallel3 branches
        self.parallel3[0].load_state_dict(torch.load('./models/GTOT_FM.pth')['parallel3'])
        self.parallel3_skconv[0].load_state_dict(torch.load('./models/GTOT_FM.pth')['paralle3_skconv'])
        self.parallel3[1].load_state_dict(torch.load('./models/GTOT_OCC.pth')['parallel3']) 
        self.parallel3_skconv[1].load_state_dict(torch.load('./models/GTOT_OCC.pth')['paralle3_skconv']) 
        self.parallel3[2].load_state_dict(torch.load('./models/GTOT_SC.pth')['parallel3'])
        self.parallel3_skconv[2].load_state_dict(torch.load('./models/GTOT_SC.pth')['paralle3_skconv'])
        self.parallel3[3].load_state_dict(torch.load('./models/GTOT_TC.pth')['parallel3'])
        self.parallel3_skconv[3].load_state_dict(torch.load('./models/GTOT_TC.pth')['paralle3_skconv'])
        self.parallel3[4].load_state_dict(torch.load('./models/GTOT_ILL.pth')['parallel3'])
        self.parallel3_skconv[4].load_state_dict(torch.load('./models/GTOT_ILL.pth')['paralle3_skconv'])
        print('load LanYangYang model end.')
        

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
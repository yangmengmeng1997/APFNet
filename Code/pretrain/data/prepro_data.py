import os
import numpy as np
import pickle
from collections import OrderedDict

set_type = 'GTOT' # set datasets GTOT or RGBT234.modify it to yourself
seq_home = '/DATA/'+set_type +'/' #modify it to yourself
challenge_type = 'ALL' # set challenge type. FM~Fast	Motion; OCC~Occlusion; SC~Scale; Variation; ILL~Illumination Variation; TC~Thermal Crossover; ALL~Whole dataset
if set_type=='GTOT':  #modify it to yourself
    seqlist_path = '/DATA/gtot.txt'     #modify it to yourself
    output_path = '/DATA/yangmengmeng/MyCode/MDNet_CAT_SK_Transformer/pretrain/data/GTOT_'+challenge_type+'.pkl'   #modify it to yourself
elif set_type == 'RGBT234':
    seqlist_path = '/DATA/RGBT234.txt'
    output_path = '/DATA/yangmengmeng/MyCode/MDNet_CAT_SK_Transformer/pretrain/data/RGBT234_'+challenge_type+'.pkl'

with open(seqlist_path,'r') as fp:
    seq_list = fp.read().splitlines() 
# Construct db
data = OrderedDict()
for i, seqname in enumerate(seq_list):
    if set_type=='GTOT':
        seq_path = seq_home+seqname
        img_list_v = sorted([p for p in os.listdir(seq_path+'/v') if os.path.splitext(p)[1] in ['.jpg','.bmp','.png']])
        img_list_i = sorted([p for p in os.listdir(seq_path+'/i') if os.path.splitext(p)[1] in ['.jpg','.bmp','.png']])
        img_list_v = [os.path.join(seq_home, seqname, 'v', img) for img in img_list_v]
        img_list_i = [os.path.join(seq_home, seqname, 'i', img) for img in img_list_i]
        gt = np.loadtxt(seq_path + '/init.txt')
    elif set_type=='RGBT234':
        seq_path = seq_home+seqname
        img_list_v = sorted([p for p in os.listdir(seq_path+'/visible') if os.path.splitext(p)[1] in ['.jpg','.bmp','.png']])
        img_list_i = sorted([p for p in os.listdir(seq_path+'/infrared') if os.path.splitext(p)[1] in ['.jpg','.bmp','.png']])
        img_list_v = [os.path.join(seq_home, seqname, 'visible', img) for img in img_list_v]
        img_list_i = [os.path.join(seq_home, seqname, 'infrared', img) for img in img_list_i]
        gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')
    assert len(img_list_v) == len(gt) == len(img_list_i), "Lengths do not match!!"
    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
    if challenge_type == 'FM':
        challenge_inf = 'fast_motion.tag'
    elif challenge_type == 'OCC':
        challenge_inf = 'occlusion.tag'
    elif challenge_type == 'SC':
        challenge_inf = 'size_change.tag'
    elif challenge_type == 'ILL':
        challenge_inf = 'illum_change.tag'
    elif challenge_type == 'TC':
        challenge_inf = 'thermal_crossover.tag'
    elif challenge_type == 'ALL':
        data[seqname] = {'images_v':img_list_v, 'images_i':img_list_i, 'gt':gt}
    if challenge_type!='ALL':
        try:
            challenge_label = np.loadtxt(os.path.join(seq_home, seqname, challenge_inf))
            challenge_label = challenge_label.tolist()
            assert len(challenge_label) == len(img_list_v), 'len(challenge_label)!=len(img_list_v):'
            challenge_label = np.array(challenge_label)
            idx = np.ones(len(img_list_v), dtype=bool)
            idx*=(challenge_label>0)
            img_list_v = np.array(img_list_v)
            img_list_i = np.array(img_list_i)
            gt = gt[idx,:]
            img_list_v = img_list_v[idx] 
            img_list_v = img_list_v.tolist()
            img_list_i = img_list_i[idx] 
            img_list_i = img_list_i.tolist()
            print(seqname,challenge_type,len(img_list_v),len(gt)) # modify py3 to py2
            if len(img_list_v)>0:
                data[seqname] = {'images_v': img_list_v, 'images_i': img_list_i,'gt': gt}
            else:
                print (seqname,'length not enough!')   # modify py3 to py2
        except:
            print (seqname,'no',challenge_type)   # modify py3 to py2
# Save db
with open(output_path, 'wb') as fp:
    print ('output_path',output_path)    # modify py3 to py2
    pickle.dump(data, fp)

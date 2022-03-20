import cv2
import numpy as np
import math


def search_iter_sample_x_axis(image_v, model_bbox,x_axis):
    
    sample = model_bbox
    iter_samples = np.tile(sample[None, :], (17 ,1))#11
    if x_axis > 0:
        dx = 1
    else:
        dx = -1
 
    j = 0
    min_x, min_y, w, h = model_bbox

    for i in range(1,17):
        i = dx*i
        iter_samples[j] = np.array([min_x+w*i*0.25, min_y, w, h])    # 1<x>1 

        # iter_samples[j+1] = np.array([min_x+i*w, min_y+h*0.5, w, h]) 
        # iter_samples[j+2] = np.array([min_x+i*w, min_y-h*0.5, w, h])

        # iter_samples[j+3] = np.array([min_x+i*w, min_y-h, w, h])
        # iter_samples[j+4] = np.array([min_x+i*w, min_y+h, w, h])

        j = j + 1
    #iter_samples[j+1] = np.array([min_x-w*0.5, min_y, w, h]) 
    #iter_samples[j+1] =  model_bbox
    # iter_samples[j] = np.array([min_x, min_y-h*0.5, w, h]) 
    # iter_samples[j+1] = np.array([min_x, min_y+h*0.5, w, h]) 

    # adjust bbox range
    # iter_samples[:, 2:] = np.clip(iter_samples[:, 2:], 10, self.img_size - 10)
    # if self.valid:
    #     iter_samples[:, :2] = np.clip(iter_samples[:, :2], iter_samples[:, 2:] / 2, self.img_size - iter_samples[:, 2:] / 2 - 1)
    # else:
    iter_samples[:, :1] = np.clip(iter_samples[:, :1], 1, image_v.size[0]-1)
    iter_samples[:, 1:2] = np.clip(iter_samples[:, 1:2], 1, image_v.size[1]-1)   

    return iter_samples 


def search_iter_sample_y_axis(image_v, model_bbox,y_axis):
    
    sample = model_bbox
    iter_samples = np.tile(sample[None, :], (17 ,1))
    if y_axis > 0:
        dy = 1
    else:
        dy = -1
 
    j = 0
    min_x, min_y, w, h = model_bbox

    for i in range(1,17):
        i = dy*i
        iter_samples[j] = np.array([min_x, min_y+h*i*0.25, w, h])    # 1<x>1 

        # iter_samples[j+1] = np.array([min_x+w*0.5, min_y+h*i, w, h]) 
        # iter_samples[j+2] = np.array([min_x-w*0.5, min_y+h*i, w, h])

        # iter_samples[j+3] = np.array([min_x-w, min_y-h*i, w, h])
        # iter_samples[j+4] = np.array([min_x+w, min_y+h*i, w, h])

        j = j + 1
    #iter_samples[j+1] = np.array([min_x, min_y-h*0.5, w, h]) 
    # iter_samples[j] = np.array([min_x+w*0.5, min_y, w, h]) 
    # iter_samples[j+1] = np.array([min_x-w*0.5, min_y, w, h]) 
    #iter_samples[j+1] =  model_bbox
    # adjust bbox range
    # iter_samples[:, 2:] = np.clip(iter_samples[:, 2:], 10, self.img_size - 10)
    # if self.valid:
    #     iter_samples[:, :2] = np.clip(iter_samples[:, :2], iter_samples[:, 2:] / 2, self.img_size - iter_samples[:, 2:] / 2 - 1)
    # else:
    iter_samples[:, :1] = np.clip(iter_samples[:, :1], 1, image_v.size[0]-1)
    iter_samples[:, 1:2] = np.clip(iter_samples[:, 1:2], 1, image_v.size[1]-1)   

    return iter_samples 





def grid_global_search(image_v,model_bbox):

    w_a = image_v.size[0]//model_bbox[2]
    h_a = image_v.size[1]//model_bbox[3]

    w_a = int(w_a)
    h_a = int(h_a)
    print(w_a,h_a)
    
    count = w_a*h_a

    sample = model_bbox
    iter_samples = np.tile(sample[None, :], (count,1))
    min_x, min_y, w, h = model_bbox
    k=0
    for i in range(w_a):
        for j in range(h_a):
            iter_samples[k] = np.array([i*w,j*h,w,h])
            k = k+1


    iter_samples[:, :1] = np.clip(iter_samples[:, :1], 1, image_v.size[0]-1)
    iter_samples[:, 1:2] = np.clip(iter_samples[:, 1:2], 1, image_v.size[1]-1)   

    return iter_samples 






def search_iter_sample_old(image_v, model_bbox):
    
    
    w_a = image_v.size[0]//model_bbox[2]
    h_a = image_v.size[1]//model_bbox[3]

    a =int(max(w_a,h_a))
    print(a)
    sample = model_bbox
    iter_samples = np.tile(sample[None, :], (a*8 ,1))
    j = 0
    min_x, min_y, w, h = model_bbox
    for i in range(1,a+1):
        for j in range(1,a+1):        
            iter_samples[j] = np.array([min_x, min_y-h*j, w, h]) 
            iter_samples[j+1] = np.array([min_x, min_y+h*j, w, h])
            
            iter_samples[j+2] = np.array([min_x-w*i, min_y, w, h])
            iter_samples[j+3] = np.array([min_x+w*i, min_y, w, h])
            
            iter_samples[j+4] = np.array([min_x-w*i, min_y-h*j, w, h])
            iter_samples[j+5] = np.array([min_x+w*i, min_y-h*j, w, h])

            iter_samples[j+6] = np.array([min_x-w*i, min_y+h*j, w, h])
            iter_samples[j+7] = np.array([min_x+w*i, min_y+h*j, w, h])

            iter_samples[j+8] = np.array([min_x+w*i, min_y+h*j, w, h])
            iter_samples[j+9] = np.array([min_x+w*i, min_y-h*j, w, h])

            iter_samples[j+10] = np.array([min_x-w*i, min_y+h*j, w, h])
            iter_samples[j+11] = np.array([min_x-w*i, min_y-h*j, w, h])


            j = j + 12

    # iter_samples[j] = np.array([min_x-w*1, min_y-h*2, w, h])
    # iter_samples[j+1] = np.array([min_x+w*1, min_y-h*2, w, h])

    # iter_samples[j+2] = np.array([min_x-w*1, min_y+h*2, w, h])
    # iter_samples[j+3] = np.array([min_x+w*1, min_y+h*2, w, h])

    # iter_samples[j+4] = np.array([min_x-w*2, min_y-h*1, w, h])
    # iter_samples[j+5] = np.array([min_x+w*2, min_y-h*1, w, h])

    # iter_samples[j+6] = np.array([min_x-w*2, min_y+h*1, w, h])
    # iter_samples[j+7] = np.array([min_x+w*2, min_y+h*1, w, h])

    # adjust bbox range
    # iter_samples[:, 2:] = np.clip(iter_samples[:, 2:], 10, self.img_size - 10)
    # if self.valid:
    #     iter_samples[:, :2] = np.clip(iter_samples[:, :2], iter_samples[:, 2:] / 2, self.img_size - iter_samples[:, 2:] / 2 - 1)
    # else:
    iter_samples[:, :1] = np.clip(iter_samples[:, :1], 1, image_v.size[0]-1)
    iter_samples[:, 1:2] = np.clip(iter_samples[:, 1:2], 1, image_v.size[1]-1)   

    return iter_samples 

def search_iter_sample_long(image_v, model_bbox):
    
    sample = model_bbox
    w_a = image_v.size[0]//model_bbox[2]
    h_a = image_v.size[1]//model_bbox[3]

    a = max(w_a,h_a)
    print(a)

    iter_samples = np.tile(sample[None, :], (int(a)*4,1))
    j = 0
    for i in range(1,int(a)+1):
        min_x, min_y, w, h = model_bbox

        iter_samples[j] = np.array([min_x, min_y-h*i, w, h]) 
        iter_samples[j+1] = np.array([min_x, min_y+h*i, w, h])
        
        iter_samples[j+2] = np.array([min_x-w*i, min_y, w, h])
        iter_samples[j+3] = np.array([min_x+w*i, min_y, w, h])
        
        # iter_samples[j+4] = np.array([min_x-w*i, min_y-h*i, w, h])
        # iter_samples[j+5] = np.array([min_x+w*i, min_y-h*i, w, h])

        # iter_samples[j+6] = np.array([min_x-w*i, min_y+h*i, w, h])
        # iter_samples[j+7] = np.array([min_x+w*i, min_y+h*i, w, h])

        j = j + 4

    # adjust bbox range
    # iter_samples[:, 2:] = np.clip(iter_samples[:, 2:], 10, self.img_size - 10)
    # if self.valid:
    #     iter_samples[:, :2] = np.clip(iter_samples[:, :2], iter_samples[:, 2:] / 2, self.img_size - iter_samples[:, 2:] / 2 - 1)
    # else:
    iter_samples[:, :1] = np.clip(iter_samples[:, :1], 1, image_v.size[0]-1)
    iter_samples[:, 1:2] = np.clip(iter_samples[:, 1:2], 1, image_v.size[1]-1)   

    return iter_samples


def match_img4(image_v, target_bbox, template_all_img, template_all_box):
    
    # serach image(current image)
    image_v = cv2.cvtColor(np.array(image_v),cv2.COLOR_RGB2BGR)
    image_v_gray = cv2.cvtColor(image_v,cv2.COLOR_BGR2GRAY)

    sample = template_all_box[0]
    match_samples = np.tile(sample[None, :], (len(template_all_box) ,1))
    # template image
    for i , template_img in enumerate(template_all_img):
        
        template_img = cv2.cvtColor(np.array(template_img),cv2.COLOR_RGB2BGR)
        template_img_gray = cv2.cvtColor(template_img,cv2.COLOR_BGR2GRAY)
        template_box = template_all_box[i]
        tlbr_box = coner2tlbr(template_box)
        tl_x,tl_y,br_x,br_y = tlbr_box
        #print(math.ceil(tl_x))
        template = template_img_gray[math.ceil(tl_y):math.ceil(br_y),math.ceil(tl_x):math.ceil(br_x)] 
        
        res = cv2.matchTemplate(image_v_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        match_samples[i] = np.array([max_loc[0],max_loc[1],template_box[2],template_box[3]])

    return match_samples    





def coner2tlbr(coner_box):

    tlbr_box = coner_box

    br_x = coner_box[0] + coner_box[2]*0.5
    br_y = coner_box[1] + coner_box[3]*0.5

    tlbr_box[2] = br_x
    tlbr_box[3] = br_y

    # upper 
    # tlbr_box[0] = np.trunc(tlbr_box)
    # tlbr_box = np.array(tlbr_box)
    return tlbr_box


def coner2center(coner_box):

    center_box = coner_box
    minx = coner_box[0]
    miny = coner_box[1]
    w = coner_box[2]
    h = coner_box[3]

    cx = minx + w*0.5
    cy = miny + h*0.5

    center_box[0] = cx
    center_box[1] = cy

    return center_box
def match_img3(image, predict_box, reliable_img_path, reliable_box, target_score,count):
    
    if target_score < 0 and count ==1 :
        image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # template = cv2.cvtColor(np.array(target),cv2.COLOR_RGB2BGR) 
        reliable_image = cv2.imread(reliable_img_path,0)
        h,w = reliable_image.shape

        search_cx = reliable_box[0] + reliable_box[2]*0.5
        search_cy = reliable_box[1] + reliable_box[3]*0.5

        search_w = 2*reliable_box[2]
        search_h = 2*reliable_box[3]
        #print(search_cx,search_cy,search_w,search_h)

        tl_x = search_cx - search_w*0.5
        tl_y = search_cy - search_h*0.5
        
        br_x = search_cx + search_w*0.5
        br_y = search_cy + search_h*0.5
        #print(int(tl_x),int(tl_y),int(br_x),int(br_y))
        if tl_x <= 0:
            tl_x = 1
        if tl_y <= 0:
            tl_y = 1
        if br_x >= w:
            br_x = w - 1
        if br_y >= h:
            br_y = h - 1

        match_box = predict_box
        template = reliable_image[math.ceil(tl_y):math.ceil(br_y), math.ceil(tl_x):math.ceil(br_x)]
        # if template is None:
        #     print('None')
        #print (template.shape)
        tw, th = template.shape[::-1]


        res = cv2.matchTemplate(image_gray,template,cv2.TM_CCOEFF_NORMED)
        # if res is None:
        #     print ('res_none')
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        #print('min_val:', min_val)
        #print('max_val:', max_val)
        tl = max_loc
      

        br = (tl[0] + w, tl[1] + h)


        template_cx = tl[0] + w*0.5
        template_cy = tl[1] + h*0.5

        cx = predict_box[0] + predict_box[2]*0.5
        cy = predict_box[1] + predict_box[3]*0.5

        x_d = abs(template_cx-cx)
        y_d = abs(template_cy-cy)
        # print(x_d,y_d)
        # print(reliable_box[2],reliable_box[3])
        if x_d > reliable_box[2] or y_d > reliable_box[3]:
            
            match_box[0] = template_cx - tw*0.5
            match_box[1] = template_cy - th*0.5
            match_box[2] = predict_box[2]
            match_box[3] = predict_box[3]

            return match_box
            print('use_match')
           
        else:
            return predict_box
    else:
        return predict_box

def match_img2(image, predict_box , reliable_img_path, reliable_box, target_score):
    
    # img_rgb = cv2.imread(image)
    # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)


    image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # template = cv2.cvtColor(np.array(target),cv2.COLOR_RGB2BGR) 
    reliable_image = cv2.imread(reliable_img_path,0)
    h,w = reliable_image.shape
    #print (h,w)
    # 
    search_cx = reliable_box[0] + reliable_box[2]*0.5
    search_cy = reliable_box[1] + reliable_box[3]*0.5

    # fw = image.shape[0]//reliable_box[2]
    # fh = image.shape[1]//reliable_box[3]
    # #print(fw,fh)
    # beta_w = 2 + fw//reliable_box[2]
    # beta_h = 2 + fh//reliable_box[3]
    # print(beta_w,beta_h)
    #search_w, search_h  = reliable_box[2:]
    search_w = 2*reliable_box[2]
    search_h = 2*reliable_box[3]
    #print(search_cx,search_cy,search_w,search_h)

    tl_x = search_cx - search_w*0.5
    tl_y = search_cy - search_h*0.5
    
    br_x = search_cx + search_w*0.5
    br_y = search_cy + search_h*0.5

    #print(int(tl_x),int(tl_y),int(br_x),int(br_y))
    
    if tl_x <= 0:
        tl_x = 1
    if tl_y <= 0:
        tl_y = 1
    if br_x >= w:
        br_x = w - 1
    if br_y >= h:
        br_y = h - 1

    match_box = predict_box
    template = reliable_image[math.ceil(tl_y):math.ceil(br_y), math.ceil(tl_x):math.ceil(br_x)]
    # if template is None:
    #     print('None')
    #print (template.shape)
    w, h = template.shape[::-1]


    res = cv2.matchTemplate(image_gray,template,cv2.TM_CCOEFF_NORMED)
    # if res is None:
    #     print ('res_none')
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    #print('min_val:', min_val)
    #print('max_val:', max_val)
    tl = max_loc
  

    br = (tl[0] + w, tl[1] + h)


    template_cx = tl[0] + w*0.5
    template_cy = tl[1] + h*0.5

    cx = predict_box[0] + predict_box[2]*0.5
    cy = predict_box[1] + predict_box[3]*0.5

    x_d = abs(template_cx-cx)
    y_d = abs(template_cy-cy)
    # print(x_d,y_d)
    # print(reliable_box[2],reliable_box[3])
    if x_d > reliable_box[2] or y_d > reliable_box[3]:
        if target_score < 0:

            match_box[0] = template_cx - w*0.5
            match_box[1] = template_cy -h*0.5
            match_box[2] = predict_box[2]
            match_box[3] = predict_box[3]

            return match_box
            print('use_match')
        else:

            return predict_box
    else:
        return predict_box


   
def match_img(image, predict_box , reliable_img_path, reliable_box):
    

    image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    reliable_image = cv2.imread(reliable_img_path,0)

    # 
    search_cx = reliable_box[0] + reliable_box[2]*0.5
    search_cy = reliable_box[1] + reliable_box[3]*0.5

    search_w = reliable_box[2]
    search_h = reliable_box[3]

    tl_x = search_cx - search_w*0.5
    tl_y = search_cy - search_h*0.5
    
    br_x = search_cx + search_w*0.5
    br_y = search_cy + search_h*0.5
    if tl_x < 0:
        tl_x = 1
    if tl_y < 0:
        tl_y = 1
    if br_x > image.shape[0]:
        br_x = image.shape[0] - 1
    if br_y > image.shape[1]:
        br_y = image.shape[1] - 1

    template = reliable_image[int(tl_y):int(br_y), int(tl_x):int(br_x)]
    w, h = template.shape[::-1]


    res = cv2.matchTemplate(image_gray,template,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    #print('min_val:', min_val)
    #print('max_val:', max_val)
    tl = max_loc
    br = (tl[0] + w, tl[1] + h)
    
    re_cx = tl[0] + w*0.5
    re_cy = tl[1] + h*0.5

    predict_box[0] = re_cx -w*0.5
    predict_box[1] = re_cy - h*0.5
    predict_box[2] = predict_box[2]
    predict_box[3] = predict_box[3]
    re_box = predict_box
    #print('use re_local')
    return re_box




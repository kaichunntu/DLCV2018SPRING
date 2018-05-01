import os

import skimage.io as img_io
import numpy as np
import scipy.misc
import tensorflow as tf

from keras.models import load_model
from keras import backend as K

from multiprocessing import Manager , Process

def load_img(fpath , fname):
    img_source = []
    img_mask = []

    for name in fname:
        path = os.path.join(fpath , name)

        img_mask.append( img_io.imread( path+"_mask.png" ) )
        img_source.append( img_io.imread(  path+"_sat.jpg") )
    return np.array(img_source) , np.array(img_mask) 

def augment(data,method):
    if method ==0:
        img = np.rot90(data , k = 1)
        
    elif method ==1:
        img = np.rot90(data , k = 2)
        
    elif method ==2:
        img = np.rot90(data , k = 2)
        img = np.flip(data , axis=1)
        
    else:
        img = np.flip(data , axis=0)
        
    return img

# def batch_augment( data , label , data_arr , label_arr , i):
#     data_arr[i] = []
#     label_arr[i] = []
#     for k ,(d,l) in enumerate(zip(data , label)):
#         data_arr[i].append(augment(d , k%4))
#         label_arr[i].append(augment(l , k%4))

def produce_augment(data , label , count=1000):
    r_idx = np.arange(data.shape[0])
    np.random.shuffle(r_idx)

    augment_data = data[r_idx[0:count]]
    augment_label = label[r_idx[0:count]]
    
#     manager = Manager()
#     job_list=[]
#     data_arr = manager.list(range(4))
#     label_arr = manager.list(range(4))
#     batch = count//4
    
#     for i in range(4):
#         start = i*batch
#         end = start+batch
#         if end > count:
#             end=count
#         p = Process(target=batch_augment , kwargs={"data":augment_data[start:end]
#                                                   ,"label":augment_label[start:end]
#                                                   ,"data_arr":data_arr
#                                                   ,"label_arr":label_arr
#                                                   ,"i":i})
#         job_list.append(p)
#         p.start()
#     for i , p in enumerate(job_list):
#         p.join()
#         print(data_arr[i].shape)
#     del augment_data , augment_label
#     augment_data , augment_label = [],[]
#     for i in range(4):
#         augment_data.extend(data_arr[i])
#         augment_label.extend(label_arr[i])
    
    for i , (d , m) in enumerate( zip( augment_data , augment_label ) ):
        
        augment_data[i] = augment(d , i%4)
        augment_label[i] = augment(m , i%4)

    data = np.concatenate([data , np.array(augment_data)] , axis=0)
    label = np.concatenate([label , np.array(augment_label)] , axis=0)
    return data , label


def norm(x):
    l = np.sqrt(np.sum((x+[1,2,3])**2))
    return (x+[1,2,3])/l
transform_arr = np.array([[0,255,255],
                          [255,255,0],
                          [255,0,255],
                          [0,255,0]  ,
                          [0,0,255]  ,
                          [255,255,255],
                          [0,0,0]])

transform_matrix = np.apply_along_axis(norm , arr=transform_arr , axis=1)

def produce_label(x):
    global transform_matrix
    x = x.reshape(-1,3)
    l = np.sqrt(np.sum((x+[1,2,3])**2 , axis=1)).reshape(-1,1)
    x = x/np.repeat(l,3,axis=1)
    return np.argmax(x@transform_matrix.T , axis=1)


def from_label_to_RGB(x):
    global trasform_arr
    RGB_seq=[]
    for i in x:
        RGB_seq.append( transform_arr[i] )
    return np.array(RGB_seq).reshape(512,512,3)



## keras function
def my_loss(y_true,y_pred):
        y_true = tf.cast(y_true , dtype=tf.int32)
        y_true = tf.reshape( y_true,[-1,512,512] )
        dum = K.one_hot(y_true,7)
        y_pred_log = K.log(  K.clip(y_pred ,1e-12 , 1) )
        return K.mean( -K.sum( dum*y_pred_log ,axis=-1 ) )

def my_acc(y_true,y_pred):
    y_true = tf.cast(y_true , dtype=tf.int32)
    y_true = tf.reshape( y_true,[-1,512,512] )
    pred = tf.cast(tf.argmax(y_pred,axis=-1), dtype=tf.int32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true , pred) , dtype=tf.float32) )


def pred_mask( data , label , model_name , use_last=False):
    
    model_path = "model_para/{}_structure.h5".format(model_name)
    model_weights_path = "model_para/{}.h5".format(model_name) 
    
    model = load_model( model_path , custom_objects={"my_loss":my_loss , "my_acc":my_acc})
    if not use_last:
        model.load_weights( model_weights_path )
    print("Check valid evaluation")
    print(model.evaluate(data, label , batch_size=8))
    print("\nPredict label")
    
    pred_valid_value = np.argmax(model.predict(data,batch_size=4,verbose=1) , axis=-1).reshape(257,-1)
    valid_pred_label.append(pred_valid_value)


    
    
    
## evaluate
def read_masks(img_mask):
    '''
    Read masks from inputs and tranform to categorical
    '''
#     file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
#     file_list.sort()
    n_masks = len(img_mask)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(img_mask):
#         mask = scipy.misc.imread(os.path.join(filepath, file))
        mask = (file >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 

    return masks

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou
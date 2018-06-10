import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys

import tensorflow as tf
import numpy as np

ly = tf.contrib.layers

from multiprocessing import Process , Manager

from keras.applications import ResNet50
from keras import backend as K

gpu_opt = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95 , allow_growth=True) 
                         ,device_count={'GPU': 1})

logdir = "RNN-ver0"


def down_sample_50(x):
    _l = len(x)
    if (_l > 50) and(_l<=100):
        i = np.random.choice([0,1] , 1)[0]
        idx = [j+i for j in range(0,_l,2) if (i+j)<_l]
        return x[idx]
    if (_l > 100) and (_l<=200) :
        
        i = np.random.choice([0,1,2,3] , 1)[0]
        idx = [j+i for j in range(0,_l,4) if (i+j)<_l]
        return x[idx]
    elif (_l > 200) and (_l<=300) :
        i = np.random.choice([0,1,2,3,4,5] , 1)[0]
        idx = [j+i for j in range(0,_l,6) if (i+j)<_l]
        return x[idx]
    elif (_l > 300) and (_l<=400) :

        i = np.random.choice([0,1,2,3,4,5,6,7] , 1)[0]
        idx = [j+i for j in range(0,_l,8) if (i+j)<_l]
        return x[idx]
        

def load_data(label_path , video_root_path ,down_factor=12 , th_count = 8,video_root="data/TrimmedVideos/"):
    
    from reader import readShortVideo , getVideoList
    
    def load_label(label_path):
        data = np.genfromtxt(label_path , dtype="str")
        label = {}
        keys = data[0].split(",")[0:6]
        for k in keys:
            label[k]=[]
        for r in data[1:]:
            tmp = r.split(",")[0:6]
            for i,k in enumerate(keys):
                label[k].append(tmp[i])
        return label

    gt_train_dict = load_label(label_path=label_path)

    gt_train_dict.keys()

    tmp_video_idx = gt_train_dict["Video_index"]
    tmp_video_cate = gt_train_dict["Video_category"]
    tmp_video_name = gt_train_dict["Video_name"]
    tmp_video_Action_labels = np.array(gt_train_dict["Action_labels"]).astype("int")

    
    #--------------------------------------------------------------------------------------------------
    def multi_load(my_id , shared_img_array , shared_label_array , shared_length_array ):
        print("#{} loading process".format(my_id))
        start_time = time.time()

        count = len(tmp_video_idx)//th_count
        start = my_id*count
        train_image = []
        train_label = []
        train_length = []
        
    #     end = start+count//100
        if my_id == th_count-1:
            end = len(tmp_video_idx)
        else:
            end = start+count
        c = 0
        for idx in tmp_video_idx[start:end]:
            c+=1
            if c == (count//2):
                print("#{} process finishes half of job. time : {:.2f}".format(my_id,time.time()-start_time))

            idx = int(idx)-1
            tmp_image = readShortVideo(video_root_path , tmp_video_cate[idx] 
                                       , tmp_video_name[idx] , downsample_factor=down_factor)
            train_image.append(tmp_image)
            tmp_label = tmp_video_Action_labels[idx]
            l = len(tmp_image)
            train_label.extend([tmp_label]*l)
            train_length.append(l)

        shared_img_array[my_id] = np.concatenate(train_image , axis=0)
        shared_label_array[my_id] = np.array(train_label)
        shared_length_array[my_id] = np.array(train_length)
    
        print("#{} process finishes job. time : {:.2f}".format(my_id,time.time()-start_time))
    #---------------------------------------------END-----------------------------------------------------
    manager = Manager()
    shared_img_arr = manager.list([i for i in range(th_count)])
    shared_label_arr = manager.list([i for i in range(th_count)])
    shared_length_arr = manager.list([i for i in range(th_count)])
    
    
    print("Multi-process : loading image...\n")
    start_time = time.time()
    job_list = []
    for i in range(th_count):
        config_dict = {"my_id":i ,  "shared_img_array":shared_img_arr 
                       ,"shared_label_array":shared_label_arr , "shared_length_array":shared_length_arr}
        p = Process(target=multi_load , kwargs=config_dict)
        p.start()
        job_list.append(p)

    for i in range(th_count):
        job_list[i].join()

    print()
    print("Finish loading. Consume time : {}".format(time.time()-start_time))

    train_image = shared_img_arr[0]
    train_label = shared_label_arr[0]
    train_length = shared_length_arr[0]
    for i in range(1,th_count):
        train_image = np.concatenate([train_image , shared_img_arr[i]] , axis=0)
        train_label = np.concatenate([train_label , shared_label_arr[i]] , axis=-1)
        train_length = np.concatenate([train_length , shared_length_arr[i]] , axis=-1)
    print("Shape of image :" , train_image.shape)
    print("Shape of label :" , train_label.shape)
    print("Count of video :" , train_length.shape)
    

    del tmp_video_idx, tmp_video_cate , tmp_video_name ,tmp_video_Action_labels
    return train_image , train_label , train_length


##---------------------------------------get feature----------------------------------------------------
def get_video_feature(label_path , video_root_path, share_video_arr , share_video_label , share_video_sep_count ):
    
    from tensorflow.python.keras.applications.resnet50 import preprocess_input
    
    _image , _label , _length = load_data(label_path , video_root_path, down_factor=12 , th_count=4 , video_root="data/TrimmedVideos/")
    
    print("Extract feature\n")
    sess = tf.Session(config=gpu_opt)
    K.set_session(sess)

    my_res = ResNet50(include_top=False , weights="model_para/resnet.h5",input_shape=[240,320,3])

    _feature_arr = []
    batch_size = 200
    start = 0
    
    for i in range(_image.shape[0]//batch_size):
        _feature_arr.append(my_res.predict(preprocess_input(_image[start:start+batch_size].astype("float"))))
        start+=batch_size
    _feature_arr.append(my_res.predict(preprocess_input(_image[start:start+batch_size].astype("float"))))
    
    _feature_arr = np.concatenate(_feature_arr , axis=0)
    print("Finish extraction")
    sess.close()
    
    video_feature=[]
    video_label=[]
    video_sep_count = []
    start=0

    
    for i,_l in enumerate(_length):
        if _l >50:
            tmp = down_sample_50(_feature_arr[start:start+_l])
            video_feature.append(tmp)
            video_label.append(_label[start])
            video_sep_count.append(1)
        else:
            video_feature.append(_feature_arr[start:start+_l])
            video_label.append(_label[start])
            video_sep_count.append(1)
        start+=_l

    print("Concatenate each video to list.\nEach dimension is [ video , time , feature ].")
    share_video_arr["hi"] = video_feature
    share_video_label["hi"] = video_label
    share_video_sep_count["hi"] = video_sep_count
##---------------------------------------get feature END-------------------------------------------------


manager = Manager()
share_video = manager.dict()
share_label = manager.dict()
share_seperate_count = manager.dict()

config = {"label_path":sys.argv[2] , "video_root_path":sys.argv[1] , "share_video_arr":share_video , "share_video_label":share_label 
          ,"share_video_sep_count":share_seperate_count}
p = Process(target=get_video_feature , kwargs=config)

p.start()
p.join()

valid_feat = share_video["hi"]
valid_label = np.array(share_label["hi"])
valid_sep_count = share_seperate_count["hi"]


MAX_SEQ = 50
def pad(x):
    for idx , _data in enumerate(x):
        if len(_data)<MAX_SEQ:
            zeros = np.zeros(shape=[MAX_SEQ - len(_data) , *input_dim[1:] ])
            _data = np.concatenate([zeros , _data] , axis=0)
            x[idx] = _data
    x = np.array(x)
    assert x.shape[1] == 50
    return x

input_dim = valid_feat[0].shape
f_dim = 1
for d in input_dim[1::]:
    f_dim*=d
    
print("feature dimension :",f_dim)


rnn_action = tf.Graph()

with rnn_action.as_default() as g:
    num_class = 11
    with tf.name_scope("Input"):
        feat = tf.placeholder( dtype=tf.float32 , shape=[None , MAX_SEQ , *input_dim[1::]] , name="Video_feature")
        ys = tf.placeholder( dtype=tf.int32 , shape=[None] , name="action_label")
        dum_ys = tf.one_hot( ys , depth=num_class)
    
    _feat = tf.reshape(feat,[-1,MAX_SEQ , f_dim])
    
    with tf.variable_scope("cell_1"):
        state_1 = [tf.zeros_like(_feat[:,0,0:512]) , 
                   tf.zeros_like(_feat[:,0,0:512])]
        
        lstm_cell_1 = tf.contrib.rnn.LSTMCell(512)
    
    with tf.variable_scope("cell_2"):
        state_2 = [tf.zeros_like(_feat[:,0,0:512]) , 
                   tf.zeros_like(_feat[:,0,0:512])]
        lstm_cell_2 = tf.contrib.rnn.LSTMCell(512)
    
    
    with tf.name_scope("RNN"):
        for i in range(MAX_SEQ):
            inputs = _feat[:,i,:]
            out , state_1 = lstm_cell_1(inputs , state_1)
            out , state_2 = lstm_cell_2(out , state_2)
        
    with tf.name_scope("classifier"):
#         concat_state = tf.concat( [state_1[1] , state_2[1]] , axis=-1 )
#         score_state = ly.fully_connected(concat_state , 64 , activation_fn=tf.nn.leaky_relu , scope="score_1")
#         score_state = ly.fully_connected(score_state , 1 , activation_fn=tf.nn.sigmoid , scope="score_2")
        
#         state_sum = score_state*state_1[1] + (1-score_state)*state_2[1]
        
        fc_1 = ly.fully_connected(out , 256 , activation_fn=tf.nn.leaky_relu)
        fc_1 = ly.fully_connected(fc_1 , 128 , activation_fn=tf.nn.leaky_relu)
        prediction = ly.fully_connected(fc_1 , num_class , activation_fn=tf.nn.softmax)
        
    with tf.name_scope("Loss"):
        loss = -tf.reduce_sum( dum_ys*tf.log( tf.clip_by_value(prediction , 1e-10 , 1) ) , axis=-1 )
        loss = tf.reduce_mean( loss , axis=-1 )
    
    with tf.name_scope("Acc"):
        pred_cate = tf.reshape(tf.cast(tf.argmax(prediction , axis=-1) , dtype=tf.int32),[-1])
        acc = tf.reduce_mean(tf.cast(tf.equal(pred_cate , ys) , dtype=tf.float32))
    
    with tf.name_scope("training_strategy"):
        opt = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    summary = tf.summary.merge( [ tf.summary.scalar("loss" , loss),tf.summary.scalar("acc" , acc) ] )
    saver = tf.train.Saver()
#     writer = tf.summary.FileWriter("tb_logs/{}".format(logdir) , graph=g)
    init = tf.global_variables_initializer()
    
print("build model : {}".format(logdir))



sess = tf.Session(graph=rnn_action , config=gpu_opt)



model_path = "model_para/{}".format(logdir)
model_path = os.path.join(model_path , "resnet_avg_2_lstm.ckpt")


saver.restore(sess , model_path)

record = []
for _f in valid_feat:
    tmp = []
    tmp.append(_f)
    tmp = pad(tmp)
    tmp_cate = sess.run( pred_cate , feed_dict={feat:tmp} )
    record.append(tmp_cate[0])

result_path = sys.argv[3]
with open(os.path.join(result_path , "p2_result.txt") , "w") as f:
    for tmp in record:
        f.write(str(tmp)+"\n")
        









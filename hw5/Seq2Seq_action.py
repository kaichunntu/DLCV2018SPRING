
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys

import tensorflow as tf
import numpy as np
import skimage.io as img_io
ly = tf.contrib.layers

from multiprocessing import Process , Manager
from matplotlib import pyplot as plt


from keras.applications import ResNet50
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences


gpu_opt = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95 , allow_growth=True) 
                         ,device_count={'GPU': 1})

logdir = "Seq2Seq_video_action-ver0"


def check(data):
    data = [ int(i.split(".")[0]) for i in data ]
    for i in range(1 , len(data)):
        if (data[i]-data[i-1]) <= 0:
            return False
    return True


data_root = sys.argv[1]

def load_data(data_folder_path):
    
    data_folder = os.listdir(data_folder_path)

    data = []
    length = []
    for i in range(len(data_folder)):
        print("{}".format(data_folder[i]) , end="\r")

        ## get image
        each_video_path = os.path.join(data_folder_path , data_folder[i])
        img_name = os.listdir(each_video_path)
        img_name.sort()
        assert check(img_name)
        tmp_img = []
        for fname in img_name:
            tmp_img_path = os.path.join(each_video_path , fname)
            tmp_img.append(img_io.imread(tmp_img_path))
        tmp_img = np.stack(tmp_img,axis=0)
        length.append(len(tmp_img))
        data.append(tmp_img)
        print("{}. Count : {}".format(data_folder[i] , tmp_img.shape[0]))

    print("\n-----finish loading-----\n")
    return data , length , data_folder



def get_feature(data , share_feature_dict , mode):
    
    batch_size = 50
    sess = tf.Session(config=gpu_opt)
    K.set_session(sess)
    
    my_res = ResNet50(include_top=False , weights="model_para/resnet.h5",input_shape=[240,320,3])
    
    video_feat = []
    for _video in data:
        video_feat.append(my_res.predict(_video,batch_size=batch_size).reshape(-1,2048))
    share_feature_dict[mode] = video_feat

def seperate_video(video):
    """
    Arg:
        video : dimension is 2 , shape is [time , feature]
    """
    _l = len(video)
    interval = 160
    interval_count = _l // interval
    mod = _l % interval
    add_mod_interval = mod // interval_count
    interval+=add_mod_interval
    
    sep_video = []
    start = 0
    for i in range(interval_count-1):
        sep_video.append(video[start:start+interval])
        start += interval
    
    sep_video.append(video[start:])
    
    return sep_video

# def random_sample_video(video , label):
    
#     tmp_video = []
#     tmp_label = []
#     for i in range(3):
#         sample_length = np.random.choice( np.arange(150,180) , 1)[0]
#         start = np.random.choice(np.arange(len(video)-sample_length) , 1)[0]
#         tmp_video.append(video[start:start+sample_length])
#         tmp_label.append(label[start:start+sample_length])
#     return  tmp_video , tmp_label 
    

valid_img , valid_length , valid_file_name = load_data(data_root)    

manager = Manager()
share_feature_dict = manager.dict()

print("Extract feature of valid data\n")
config = {"mode":"valid" , "share_feature_dict":share_feature_dict , "data":valid_img }
p = Process(target=get_feature , kwargs=config)
p.start()
p.join()

print("\n-----Finish Extracting-----\n")

valid_feat = share_feature_dict["valid"]



print("Slice valid data...")
slice_valid_feat=[]
for tmp_f in valid_feat :
    tmp = seperate_video(tmp_f )
    slice_valid_feat.extend(tmp)

slice_valid_length = []
for tmp in slice_valid_feat:
    _l = len(tmp)
    slice_valid_length.append(_l)

slice_valid_length = np.array(slice_valid_length)
print("Count of valid data :" , len(slice_valid_feat) , "\n")


MAX_SEQ = 192

# slice_valid_feat = pad_sequences(slice_valid_feat,maxlen=MAX_SEQ)
slice_valid_feat = pad_sequences(slice_valid_feat,maxlen=MAX_SEQ,padding="post")

seq_action = tf.Graph()

with seq_action.as_default() as g:
    num_class = 11
    with tf.name_scope("Input"):
        feat = tf.placeholder( dtype=tf.float32 , shape=[None , MAX_SEQ , 2048] , name="Video_feature")
        ys = tf.placeholder( dtype=tf.int32 , shape=[None,MAX_SEQ] , name="action_label")
        dum_ys = tf.one_hot( ys , depth=num_class)
        _feat = feat
        
    with tf.name_scope("mask"):
        mask = tf.placeholder(dtype=tf.int32 , shape=[None])
        mask_seq = tf.sequence_mask(mask , maxlen=MAX_SEQ , dtype=tf.float32)
    
    with tf.variable_scope("cell_1"):
        state_1 = [tf.zeros_like(_feat[:,0,0:512]) , 
                   tf.zeros_like(_feat[:,0,0:512])]
        
        lstm_cell_1 = tf.contrib.rnn.LSTMCell(512)
    
    with tf.variable_scope("cell_2"):
        state_2 = [tf.zeros_like(_feat[:,0,0:512]) , 
                   tf.zeros_like(_feat[:,0,0:512])]
        lstm_cell_2 = tf.contrib.rnn.LSTMCell(512)
    
    
    concat_output = []
    with tf.name_scope("RNN"):
        for i in range(MAX_SEQ):
            inputs = _feat[:,i,:]
            out , state_1 = lstm_cell_1(inputs , state_1)
            out , state_2 = lstm_cell_2(out , state_2)
            concat_output.append(out)
        concat_output = tf.stack(concat_output , axis=1)
        
    with tf.name_scope("classifier"):
        
        fc_1 = ly.fully_connected(concat_output , 256 , activation_fn=tf.nn.leaky_relu)
        fc_1 = ly.fully_connected(fc_1 , 128 , activation_fn=tf.nn.leaky_relu)
        prediction = ly.fully_connected(fc_1 , num_class , activation_fn=tf.nn.softmax)
        
    with tf.name_scope("Loss"):
        each_frame_loss = -tf.reduce_sum( dum_ys*tf.log( tf.clip_by_value(prediction , 1e-10 , 1) ) , axis=-1 )*mask_seq
        each_video_loss = tf.reduce_sum( each_frame_loss , axis=-1 )/tf.cast(mask , tf.float32)
        loss = tf.reduce_mean( each_video_loss , axis=-1 )
    
    with tf.name_scope("Acc"):
        pred_cate = tf.reshape(tf.cast(tf.argmax(prediction , axis=-1) , dtype=tf.int32),[-1 , MAX_SEQ])
        each_frame_acc = tf.cast(tf.equal(pred_cate , ys) , dtype=tf.float32)*mask_seq
        each_video_acc = tf.reduce_sum(each_frame_acc , axis=-1)/tf.cast(mask , dtype=tf.float32)
        acc = tf.reduce_mean(each_video_acc)
    
    with tf.name_scope("training_strategy"):
        opt = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    global_step = tf.Variable(0 , dtype=tf.int32)
    add_global = global_step.assign_add(1)
    with tf.control_dependencies([add_global]):
        summary = tf.summary.merge( [ tf.summary.scalar("loss" , loss),tf.summary.scalar("acc" , acc) ] )
    saver = tf.train.Saver()
#     writer = tf.summary.FileWriter("tb_logs/{}".format(logdir) , graph=g)
    init = tf.global_variables_initializer()
    
print("build model : {}".format(logdir))



sess = tf.Session(graph=seq_action , config=gpu_opt)

model_path = "model_para/{}".format(logdir)
model_path = os.path.join(model_path , "seq2seq-ver1.ckpt")

saver.restore(sess , model_path)


record_slice_video_action = []
for f , mask_len in zip(slice_valid_feat , slice_valid_length ):
    tmp_cate = sess.run(pred_cate, feed_dict={feat:[f] , mask:[mask_len]})
    record_slice_video_action.append(tmp_cate[0,0:mask_len])

start = 0
i=0
record = []
for _l in valid_length:
    tmp_length = 0
    tmp_cate_arr = []
    while(tmp_length != _l):
        tmp_cate_arr.extend(record_slice_video_action[i])
        tmp_length += len(record_slice_video_action[i])
        i+=1
        assert tmp_length <= _l
    record.append(tmp_cate_arr)

assert i == len(slice_valid_feat)
assert len(record) == len(valid_file_name)



for tmp_record , tmp_file_name in zip(record , valid_file_name):
    p = os.path.join(sys.argv[2] , tmp_file_name)
    p+=".txt"
    print( "Write file :" , p )
    with open(p,"w") as f:
        for i in tmp_record:
            f.write(str(i)+"\n")











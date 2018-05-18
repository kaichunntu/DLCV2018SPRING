
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time


import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import skimage.io as img_io
import numpy as np

from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.contrib import layers as ly

gpu_opt = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95 , allow_growth=True) 
                             ,device_count={'GPU': 1})
logdir="VAE-ver0"
processing_input = lambda x : (x/255-0.5)*2
inverse_processing = lambda x : (x/2+0.5)*255

l_file_name = ("KL_loss_1.csv" , "MSE_loss_1.csv")

fig = plt.figure(figsize=[12,4])
for i , f in enumerate(l_file_name):
    ax = fig.add_subplot(1,2,i+1)
    ax.set_title(f[0:-4])
    d = np.genfromtxt(os.path.join( "tmp/{}".format(logdir) ,  f),skip_header=1 , delimiter=",")
    l = d[:,2]
    ax.plot( range(0,200000 , 200) , l , c="orange")
    ax.grid()
plt.savefig("{}/fig1_2.png".format(sys.argv[2]))
plt.show()


## load data
data_root = sys.argv[1]
train_path = os.path.join( data_root , "train" )
test_path = os.path.join( data_root , "test" )

train_attribute = np.genfromtxt( train_path+".csv" , delimiter="," , skip_header=0 , dtype="str")
test_attribute = np.genfromtxt( test_path+".csv" , delimiter="," , skip_header=0 , dtype="str")

attr = train_attribute[0,1::]
print("Attribute : ")
print(" , ".join(attr[1:8]))
print(" , ".join(attr[8::]))

train_id = train_attribute[1:,0]
train_attribute = train_attribute[1:,1::].astype("float")

test_id = test_attribute[1:,0]
test_attribute = test_attribute[1:,1::].astype("float")

## load image
train_img = []
for f in train_id:
    train_img.append( img_io.imread( os.path.join( train_path , f ) ) )
    

test_img = []
for f in test_id:
    test_img.append( img_io.imread( os.path.join( test_path , f ) ) )

train_img = np.array(train_img).astype("float")
test_img = np.array(test_img).astype("float")
    
print("Shape of training data :" , train_img.shape)
print("Shape of testing data  :" , test_img.shape)


train_data = processing_input(train_img)
test_data = processing_input(test_img)


VAE_graph = tf.Graph()

def reconstruct_layer(inputs , tr=True):
    x = ly.fully_connected(inputs , 2048 , trainable=tr 
                            , scope="fc_1")
    x = tf.reshape( x , shape=[-1,1,1,2048] )

    x = ly.conv2d_transpose(x , 512 , [2,2] , padding="VALID" , trainable=tr
                             , scope="Conv2DT_1")
    x = ly.conv2d_transpose(x , 256 , [5,5] , padding="VALID" , trainable=tr
                             , scope="Conv2DT_2")
    x = ly.batch_norm(x  , trainable=tr , scope="bc_1")
    x = ly.conv2d_transpose(x , 128 , [6,6] , stride=[2,2] , padding="VALID" 
                            , trainable=tr , scope="Conv2DT_3")

    x = ly.batch_norm(x , reuse=tf.AUTO_REUSE , scope="bc_2")
    x = ly.conv2d_transpose(x , 64 , [9,9] , stride=[4,4] , padding="SAME" 
                            , trainable=tr  , scope="Conv2DT_4")

    out = ly.conv2d(x , 3 , [1,1] , padding="VALID" , activation_fn=tf.nn.tanh 
                    , trainable=tr  , scope="To_rgb")
    return out
        


with VAE_graph.as_default():
    latent_dim = 1024
    
    with tf.name_scope("Input"):
        img = tf.placeholder( shape=[None , 64,64,3] , dtype=tf.float32 )
    
    with tf.name_scope("Encode"):
        with tf.name_scope("block1"):
            _ = ly.conv2d( img , 64 , [5,5] , padding="VALID" )
            _ = ly.conv2d( _ , 96 , [3,3] , padding="VALID" )
            block_1 = ly.avg_pool2d( _ , [3,3] , stride=[2,2] , padding="VALID" )
            
        with tf.name_scope("block2"):
            _ = ly.batch_norm(block_1)
            _ = ly.conv2d( _ , 96 , [3,3] , padding="VALID" )
            _ = ly.conv2d( _ , 128 , [3,3] , padding="VALID" )
            block_2 = ly.max_pool2d( _ , [3,3] , stride=[2,2] , padding="VALID" )
            
        with tf.name_scope("block3"):
            _ = ly.batch_norm(block_2)
            _ = ly.conv2d( _ , 128 , [3,3] , padding="VALID" )
            _ = ly.conv2d( _ , 256 , [1,1] , padding="VALID" )
            _ = ly.conv2d( _ , 256 , [3,3] , padding="VALID" )
            block_3 = ly.avg_pool2d( _ , [2,2] , stride=[1,1] , padding="VALID" )
        
        with tf.name_scope("block4"):
            _ = ly.batch_norm(block_3)
            _ = ly.conv2d( _ , 512 , [3,3] , stride=[2,2] , padding="VALID" )
            block_4 = ly.conv2d( _ , 2048 , [2,2] , padding="VALID" )
        
        flat_code = tf.reshape( block_4 , shape=[-1,2048] , name="flat")
        
    with tf.name_scope("z_code"):
        z_mean = ly.fully_connected(flat_code , latent_dim , activation_fn=tf.nn.tanh , scope="z_mean")
        z_log_var = ly.fully_connected(flat_code , latent_dim , activation_fn=tf.nn.leaky_relu , scope="z_log_var")
    
    with tf.name_scope("Sampling"):
        epsi = tf.random_normal( shape=tf.shape(z_log_var) )
        z = z_mean + epsi*tf.exp(z_log_var/2)

    with tf.name_scope("Decode"):
        with tf.variable_scope("shared_decoder"):
            img_reconstruct = reconstruct_layer(z)
    
    
    with tf.name_scope("Infer"):
        with tf.variable_scope("shared_decoder" , reuse=True):
            Infer_img = reconstruct_layer(z_mean , tr=False )
            Infer_loss = tf.reduce_mean( tf.square(Infer_img-img ))
    ## Assign random vector to z 
    with tf.name_scope("random_infer"):
        with tf.variable_scope("shared_decoder" , reuse=True):
            random_z = tf.placeholder(shape=[None,latent_dim] , dtype=tf.float32)
            Infer_by_random = reconstruct_layer(random_z , tr=False)
            
    
    with tf.name_scope("Loss"):
        with tf.name_scope("KL_loss"):
            kl_loss = -0.5*tf.reduce_sum( 1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var) ,axis=-1 )
            
            
        with tf.name_scope("MSE_loss"):
            dim = 1
            for d in img.shape[1::]:
                dim *= int(d)
            mse_loss = tf.reduce_mean(  tf.reshape(tf.square(img - img_reconstruct) , [-1,dim]),axis=-1)
            
    
        loss = tf.reduce_mean(mse_loss+5e-5*kl_loss)
    tf.summary.scalar("Loss/KL_loss" , tf.reduce_mean(kl_loss))  
    tf.summary.scalar("Loss/MSE_loss" , tf.reduce_mean(mse_loss))
    tf.summary.scalar("Total_loss" , loss)
    
    with tf.name_scope("Train_strategy"):
        decay_policy = tf.train.exponential_decay(2e-4 , decay_rate=0.9 , decay_steps=4000 , global_step=1000)
        opt_operation = tf.train.AdamOptimizer(learning_rate=decay_policy).minimize(loss)
        
        second_opt_operation = tf.train.AdamOptimizer(1e-6).minimize(loss)
    
    init = tf.global_variables_initializer()
    
#     merged_log = tf.summary.merge_all()
#     VAE_writer = tf.summary.FileWriter("tb_logs/VAE_ver0" , graph=VAE_graph)
    
    saver = tf.train.Saver()


sess = tf.Session(graph=VAE_graph , config=gpu_opt)

saver.restore(sess,"model_para/VAE-ver0")

def concat_img(g , col=10):
    concat_all_img = []
    img_count = g.shape[0]
    row_padding = np.zeros(shape=[1 , g.shape[2]*col+col-1,3])
    col_padding = np.zeros(shape=[g.shape[1],1,3])
    for i in range(img_count//col):
        a = g[i*col]
        for j in range(1,col):
            a = np.concatenate( [ a , col_padding , g[ i*col+j ]  ] , axis=1  )
        concat_all_img.append(a)
        if i == (img_count//col-1):
            break
        concat_all_img.append(row_padding)
    return np.concatenate( concat_all_img , axis=0 )



pr_img_path = "tmp/{}".format("VAE-ver0")
if not os.path.exists(pr_img_path):
    os.mkdir(pr_img_path)
print(pr_img_path)


idx = [2314,402,1111,399,41,2169,432,683,218,2455]

plt.style.use("classic")

test_reconstruct = sess.run(Infer_img , feed_dict={img:test_data[idx]})

test_reconstruct = np.concatenate([test_img[idx] , inverse_processing(test_reconstruct)] , axis=0)

plt.figure(figsize=(15,4))
plt.imshow(concat_img(test_reconstruct).astype("int"))
plt.axis("off")
plt.savefig("{}/fig1_3.png".format(sys.argv[2]))


batch_z = np.load("{}/random_z.npy".format(pr_img_path))
random_reconstruct = sess.run( Infer_by_random , feed_dict={random_z:batch_z} )
random_reconstruct = inverse_processing(random_reconstruct)
random_reconstruct = concat_img(random_reconstruct , col=8)

plt.figure(figsize=[15,10])
plt.imshow(random_reconstruct.astype("int"))
plt.axis("off")
plt.savefig("{}/fig1_4.png".format(sys.argv[2]))
plt.show()

from sklearn.manifold import TSNE
print("TSNE")
start_time = time.time()


start = 0
test_latent_space = []

# idx = np.random.choice(np.arange(test_data.shape[0]) , size=1000 , replace=False)
idx = np.load("{}/sample_test_data_idx.npy".format("tmp/VAE-ver0"))
test_data = test_data[idx]

batch_size=100
for l in range(test_data.shape[0]//batch_size):
    test_latent_space.append( sess.run(z_mean , feed_dict={img:test_data[start:start+batch_size]}))
    start+=batch_size
# test_latent_space.append(sess.run(z_mean , feed_dict={img:test_data[start:start+batch_size]}))

test_latent_space = np.concatenate(test_latent_space , axis=0)


reduction_z = TSNE(random_state=1).fit_transform(test_latent_space)


i = 2
test_attr = test_attribute[:,i:i+1][idx]
plt.style.use("ggplot")

fig, ax = plt.subplots(figsize=[8,6])
plt.title(attr[i])
label = ["non-black" , "black"]
color = ["navy" , "orange" ]
for i in range(2):
    idx = test_attr.reshape(-1)==i
    ax.scatter(reduction_z[idx,0],reduction_z[idx,1] , c=color[i] , label=label[i])

plt.legend()
plt.savefig("{}/fig1_5.png".format(sys.argv[2]))
plt.show()
print("Finish tSNE transform. Time :" , time.time()-start_time)


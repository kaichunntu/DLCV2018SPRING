
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


logdir="InfoGAN-ver0"
print("file name : "+logdir+"\n")


processing_input = lambda x : (x/255-0.5)*2
inverse_processing = lambda x : (x/2+0.5)*255

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

def Mix_attr( attr_inputs , z_inputs ):
    """
    Mixed z_inputs with attr vector
    """
#     with tf.variable_scope("g_embedding" , reuse=True):
#         emb = tf.get_variable("attr")
#         transform_weight = tf.get_variable("transform_w")
#     zero_vec = tf.zeros_like(emb[0:1,:])
#     emb = tf.concat([zero_vec,emb] , axis=0)
    
#     z = attr_inputs@emb
    
#     z = tf.nn.selu(attr_vector)
    
    return tf.concat([z_inputs , attr_inputs] , axis=-1)



def Generate( z ):
    """
    Defined by GAN.
    The model is to generate image to cheat descrinator.
    Design with Conv2DTranspose and add batch normalization.
    
    Args:
        inputs : a 2-D tensor , shape is [batch size , z_dim]. Dtype must be float.
                Notice : z_dim nust be multiple of 4
    
    return:
        4-D tensor : [batch size , 64 , 64 , 3].
    """
    bias_regular = ly.l2_regularizer(0.2)
    print("Build generator")
    ## z is 512
    x = tf.reshape(z , [-1,1,1,int(z.shape[1])])
    x = ly.conv2d( x , 2048 , [1,1] , stride=[1,1] , activation_fn=tf.nn.selu 
                  , biases_regularizer=bias_regular 
                  , scope="g_conv_0")
    x = tf.reshape( x , [ -1,2,2,2048//4 ] )
    x = ly.conv2d_transpose( x , 512 , [4,4] , stride=[2,2] , activation_fn=tf.nn.selu 
                            , biases_regularizer=bias_regular
                            , padding="SAME" , scope="g_0")
    print(x.shape)
    ## 4x4x128
    x = ly.batch_norm( x , scope="g_bn_1")
    x = ly.conv2d_transpose( x , 256 , [4,4] , stride=[2,2] , activation_fn=tf.nn.selu
                            , biases_regularizer=bias_regular
                            , padding="SAME" , scope="g_1")
    
#     print(x.shape)
    ## 4x4  why should kernel size be 6x6 not 5x5 ? 
    x = ly.conv2d_transpose( x , 128 , [5,5] , stride=[2,2] , activation_fn=tf.nn.selu
                            , biases_regularizer=bias_regular
                            , padding="SAME" , scope="g_2")
    x = ly.batch_norm( x , scope="g_bn_2")
    
    x = ly.conv2d_transpose( x , 96 , [5,5] , stride=[2,2] , activation_fn=tf.nn.selu
                            , biases_regularizer=bias_regular
                            , padding="SAME" , scope="g_3")
#     x = tf.add(x , x_1 )
    print(x.shape)
    x = ly.conv2d_transpose( x , 64 , [3,3] , stride=[2,2] , activation_fn=tf.nn.selu
                            , biases_regularizer=bias_regular
                            , padding="SAME" , scope="g_4")
    x = ly.conv2d( x , 3 , [1,1] , stride=[1,1] , activation_fn=tf.nn.tanh
                            , biases_regularizer=bias_regular
                            , padding="SAME" , scope="g_5")
    print(x.shape)
    
    return x


InfoGAN_graph = tf.Graph()


print(logdir+"\n")
with InfoGAN_graph.as_default():
    latent_dim=128
    num_class=7
    with tf.name_scope("Input"):
        z_input = tf.placeholder( shape=[None,latent_dim] , dtype=tf.float32 , name="latent_space")
        clf_tag = tf.placeholder( shape=[None,num_class] , dtype=tf.float32 , name="class" )
    
    with tf.name_scope("Mix_z"):
        z = Mix_attr(clf_tag , z_input)
    
    with tf.name_scope("Generator"):
        g_img = Generate(z)
    saver = tf.train.Saver(var_list=tf.trainable_variables())



sess = tf.Session( graph=InfoGAN_graph )

model_path = "model_para/{}".format(logdir)
saver.restore( sess , os.path.join(model_path , "generator_"+str(70000)+".ckpt") )


test_z = np.load("tmp/{}/z_vector.npy".format(logdir))
test_attr = []

for i in range(num_class):
    tmp_a = np.zeros(shape=[5,num_class])
    tmp_a[:,i] += 1
    test_attr.append(tmp_a)
    del tmp_a

def print_generate_img():
    g = sess.run( g_img , feed_dict={z_input:test_z , clf_tag:test_attr[0]} )
    for i in range(1,num_class):
        tmp_g = sess.run( g_img , feed_dict={z_input:test_z , clf_tag:test_attr[i]} )
        g = np.concatenate( [g , tmp_g] , axis=0 )
    g = inverse_processing(g)
    g = concat_img(g , col=5).astype("int")
    
    plt.figure(figsize=(8,12)) 
    plt.imshow(g)
    plt.axis("off")
    path = os.path.join( sys.argv[1] , "bonus_1.png" )
    plt.savefig(path)
#     plt.show()

print_generate_img()


tmp_attr = np.zeros([5,num_class])
tmp_attr = np.zeros([5 , num_class])
tmp_attr[:,0]+=1
tmp_attr[:,6]+=1
tmp_g = sess.run( g_img , feed_dict={z_input:test_z , clf_tag:tmp_attr} )
g = inverse_processing(tmp_g)
g = concat_img(g,col=5).astype("int")

plt.figure(figsize=(8,4)) 
plt.imshow(g)
plt.axis("off")
path = os.path.join( sys.argv[1] , "bonus_2.png" )
plt.savefig(path)



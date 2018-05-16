
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
logdir="GAN-ver2"


processing_input = lambda x : (x/255-0.5)*2
inverse_processing = lambda x : (x/2+0.5)*255

def concat_img(g , col=8):
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


print("Build model.")
print()


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
    x = tf.reshape(z , [-1,1,1,latent_dim])
    x = ly.conv2d( x , 4096 , [1,1] , stride=[1,1] , activation_fn=tf.nn.selu , biases_regularizer=bias_regular , scope="g_conv_0")
    x = tf.reshape( x , [ -1,2,2,4096//4 ] )
    x = ly.conv2d_transpose( x , 512 , [4,4] , stride=[2,2] , activation_fn=tf.nn.selu 
                            , biases_regularizer=bias_regular
                            , padding="SAME" , scope="g_0")
    print(x.shape)
    x_1 = ly.conv2d_transpose( x , 128 , [16,16] , stride=[8,8] , activation_fn=tf.nn.selu
                            , biases_regularizer=bias_regular
                            , padding="SAME" , scope="g_1_1")
    x_1 = ly.batch_norm( x_1 , scope="g_bn_1_1")
    ## 4x4x128
    x = ly.batch_norm( x , scope="g_bn_1")
    x = ly.conv2d_transpose( x , 256 , [4,4] , stride=[2,2] , activation_fn=tf.nn.selu
                            , biases_regularizer=bias_regular
                            , padding="SAME" , scope="g_1")
    
#     print(x.shape)
    ## 4x4  why should kernel size be 6x6 not 5x5 ? 
    x = ly.conv2d_transpose( x , 128 , [9,9] , stride=[4,4] , activation_fn=tf.nn.selu
                            , biases_regularizer=bias_regular
                            , padding="SAME" , scope="g_2")
    x = ly.batch_norm( x , scope="g_bn_2")
    
#     x = tf.concat([x , x_1] , axis=-1)
    x = tf.add(x , x_1 )
    x = ly.conv2d_transpose( x , 96 , [5,5] , stride=[2,2] , activation_fn=tf.nn.selu
                            , biases_regularizer=bias_regular
                            , padding="SAME" , scope="g_3")
#     x = tf.add(x , x_1 )
    print(x.shape)
    x = ly.conv2d( x , 3 , [1,1] , stride=[1,1] , activation_fn=tf.nn.tanh
                            , biases_regularizer=bias_regular
                            , padding="SAME" , scope="g_4")
    
    
    return x

GAN_graph = tf.Graph()

with GAN_graph.as_default():
    latent_dim=256
    
    z_input = tf.placeholder( shape=[None,latent_dim] , dtype=tf.float32 , name="latent_space")
    
    with tf.name_scope("Generator"):
        g_img = Generate(z_input)
    
    
    saver = tf.train.Saver(var_list=tf.trainable_variables())


sess = tf.Session( graph=GAN_graph )

model_path = "model_para/{}".format(logdir)
saver.restore( sess , os.path.join(model_path , "generator_"+str(95000)+".ckpt") )

pr_img_path = "tmp/{}".format(logdir)
test_z = np.load(os.path.join(pr_img_path , "z_vector.npy"))

def print_generate_img():
    g = inverse_processing(sess.run( g_img , feed_dict={z_input:test_z} ))
    fig = plt.figure(figsize=(12,8))
    
    g = concat_img(g).astype("int")
    plt.imshow(g)
    plt.axis("off")
    path = os.path.join( sys.argv[1], "fig2_3.png" )
    plt.savefig(path)
#     plt.show()
    
    
print_generate_img()
    

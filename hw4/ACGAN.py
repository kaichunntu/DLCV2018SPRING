
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
logdir="ACGAN-ver2"

l_file_name = (( "real-desc_loss.csv" , "fake-desc_loss.csv" ) , ( "real-clf_loss.csv" ,"fake-clf_loss.csv" ))
color = [ "firebrick" ,"royalblue"  ]

fig = plt.figure(figsize=[12,4])
for i , f_group in enumerate(l_file_name):
    ax = fig.add_subplot(1,2,i+1)
    ax.set_title(f_group[0][5:-4])
    
    for j , f in enumerate(f_group):
        d = np.genfromtxt(os.path.join( "tmp/{}".format(logdir) ,  f),skip_header=1 , delimiter=",")
        l = d[:,2]
        ax.plot( range(0,200000 , 200) , l , c=color[j] , label=f[0:4])
    plt.xticks(rotation=45)
    ax.grid()
    ax.legend()
plt.savefig("{}/fig3_2.png".format(sys.argv[1]))
# plt.show()




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


print("Build model.")
print()
def Mix_attr( attr_inputs , z_inputs ):
    """
    Mixed z_inputs with attr vector
    """
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
    x = tf.reshape(z , [-1,1,1,int(z.shape[-1])])
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
    x = ly.conv2d_transpose( x , 64 , [5,5] , stride=[2,2] , activation_fn=tf.nn.selu
                            , biases_regularizer=bias_regular
                            , padding="SAME" , scope="g_3")
#     x = tf.add(x , x_1 )
    print(x.shape)
    x = ly.conv2d( x , 3 , [1,1] , stride=[1,1] , activation_fn=tf.nn.tanh
                            , biases_regularizer=bias_regular
                            , padding="SAME" , scope="g_4")
    
    
    return x



ACGAN_graph = tf.Graph()


print(logdir+"\n")
with ACGAN_graph.as_default():
    latent_dim=256
    num_class=1
    with tf.name_scope("Input"):
        z_input = tf.placeholder( shape=[None,latent_dim] , dtype=tf.float32 , name="latent_space")
        clf_tag = tf.placeholder( shape=[None,num_class] , dtype=tf.float32 , name="class" )
    
    with tf.name_scope("Mix_z"):
        z = Mix_attr(clf_tag , z_input)
    
    with tf.name_scope("Generator"):
        g_img = Generate(z)
    
    tvars = tf.trainable_variables()

    for v in tvars:
        if "d_" in v.name:
            pass
        elif "g_" in v.name:
            tf.add_to_collection( "gene" , v )
    saver = tf.train.Saver(var_list=tf.get_collection("gene") , max_to_keep=10)
    
    
    
sess = tf.Session( graph=ACGAN_graph )

model_path = "model_para/{}".format(logdir)
saver.restore( sess , os.path.join(model_path , "generator_"+str(100000)+".ckpt") )

pr_img_path = "tmp/{}".format(logdir)
test_z = np.load(os.path.join(pr_img_path , "z_vector.npy"))
test_zero = np.zeros([test_z.shape[0] , 1])
test_one = np.ones( [test_z.shape[0] , 1] )

def print_generate_img():
    g = inverse_processing(sess.run( g_img 
                                    , feed_dict={
                                        z_input:np.concatenate([test_z,test_z] , axis=0)
                                        ,clf_tag:np.concatenate([test_zero,test_one] , axis=0)
                                    } ))
    fig = plt.figure(figsize=(12,4))
    
    g = concat_img(g).astype("int")
    plt.imshow(g)
    plt.axis("off")
    path = os.path.join( sys.argv[1], "fig3_3.png" )
    plt.savefig(path)
#     plt.show()
    
    
print_generate_img()   
    
    
    
    


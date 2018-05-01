
import os
import sys
# %env CUDA_VISIBLE_DEVICES=0
import skimage.io as img_io
import numpy as np
import gc

from multiprocessing import Process , Manager

from matplotlib import pyplot as plt

## my function
from process_method import *



# data_root = "./data"
# train_path = os.path.join( data_root , "train" )
# valid_path = os.path.join( data_root , "validation" )

# train_name = [ n.split("_")[0] for n in os.listdir(train_path)]
# train_name = np.unique(train_name)


valid_path = sys.argv[1]
save_path = sys.argv[2]

valid_name = [ n.split("_")[0] for n in os.listdir(valid_path)]
valid_name = np.unique(valid_name)

# train_data , train_mask = load_img( train_path , train_name)
valid_data , valid_mask = load_img( valid_path , valid_name)
print("Finish load data")
# print("Train data shape : {}".format(train_data.shape))
print("Valid data shape : {}".format(valid_data.shape))


print("Preprocessing")
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

# train_data = preprocess_input(train_data.astype("float"))
valid_data = preprocess_input(valid_data.astype("float"))




print( "Build Model." )
from keras.models import load_model , Model
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3 ## "model_para/Inception_v3_imagenet.h5"
from keras.utils import to_categorical , plot_model

from keras.layers import Input , Conv2D , MaxPooling2D , Deconv2D , Conv2DTranspose , Dense
from keras.layers import UpSampling2D , BatchNormalization , GlobalAveragePooling2D
from keras.layers import Multiply , Add , Concatenate
from keras.layers import Lambda , Reshape

from keras.callbacks import EarlyStopping , ModelCheckpoint
from keras import backend as K
from keras import optimizers
from keras import regularizers

import tensorflow as tf


def run_model(data,model_name,history=None):
#     %env CUDA_VISIBLE_DEVICES=0
#     print( "Model name is " , model_name )
#     gpu_opt = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8 , allow_growth=False) 
#                              ,device_count={'GPU': 1})
#     sess = tf.Session(config=gpu_opt)
#     K.set_session(sess)
    
    base_model = InceptionV3(weights=None, include_top=False , input_shape=[512,512,3])

    my_extract = Model(base_model.input , base_model.get_layer("mixed6").output)
    my_extract.load_weights( "model_para/Inception_v3_imagenet.h5" , by_name=True )

    img = Input(shape=[512,512,3])

    x = my_extract( img )

    x_in = Conv2D(1024 , [1,1] , strides=[1,1] , activation="relu")(x)
    x_in = Conv2D(1024 , [1,1] , strides=[1,1] , activation="relu")(x_in)
    x_in = BatchNormalization()(x_in)
    x_in = Conv2DTranspose(256 , [3,3] , strides=[1,1] , activation="relu")(x_in)

    de_1 = Conv2DTranspose(32 , [32,32] , strides=[16,16] , activation="relu" , padding="same")(x_in)
    de_1 = Conv2D(7 , [1,1] , strides=[1,1] , activation="softmax")(de_1)

    model = Model(img , de_1)
    model.summary()
    plot_model(model , "Inception_v3_16s.png" , show_shapes=True)

    decay_policy = tf.train.exponential_decay(1e-4 , decay_rate=0.9 , decay_steps=10000 , global_step=2000 )
    opt = optimizers.Adam(decay_policy)
    early = EarlyStopping('val_loss' , patience=6 , mode="min")
    model_check = ModelCheckpoint("model_para/{}.h5".format(model_name) 
                                  , monitor="val_loss", mode="min" , verbose=1 
                                  , save_best_only=True , save_weights_only=True)
    
    model.compile(loss=my_loss , optimizer=opt , metrics=[my_acc])
    
    train_data , train_label , new_valid_data , new_valid_label = data
    
    h = inception_segment.fit(train_data , train_label , batch_size=16 , verbose=1 , epochs=200, callbacks=[early , model_check]
                    ,validation_data=[new_valid_data , new_valid_label])
    for k,v in h.history.items():
        history[k] = v
    model.save("model_para/{}_structure.h5".format(model_name))
    return model , history

# model = run_model(data=[ train_data , train_label , valid_data , valid_label ],"Inception-v3-FCN16s",history={})
# model.load_weights( "Inception-v3-FCN16s.h5" )

model_path = "improvement.h5"
model = load_model(model_path, custom_objects={"my_loss":my_loss , "my_acc":my_acc})

z = np.argmax(model.predict( valid_data,batch_size=8) ,axis=-1).reshape(257,-1)

pred_valid_img = np.apply_along_axis( from_label_to_RGB , arr=z , axis=1 )

orig = read_masks(valid_mask)
pred_m = read_masks( pred_valid_img )

mean_iou_score(pred_m , orig)



# save_path = os.path.join( data_root , "valid_pred" )
if not os.path.exists(save_path):
    os.mkdir(save_path)

import warnings
warnings.simplefilter("ignore")

for name,img in zip(valid_name , pred_valid_img):
    img_io.imsave( os.path.join( save_path , name+"_mask.png" )  , img.astype("uint8"), quality=100  )


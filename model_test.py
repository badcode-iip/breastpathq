import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
#from torch.utils.data import Dataset
#from torch.utils.data import DataLoader, TensorDataset
#from torchvision.transforms import transforms
from skimage import transform,exposure
from skimage.io import imread, imshow, imread_collection, concatenate_images
#import matplotlib.pyplot as plt
import skimage.io as io
import skimage
import csv
import scipy.stats as stats
import keras.backend as k
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from xception_padding import Xception
import keras
from keras.losses import categorical_crossentropy,mean_squared_error,logcosh
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy
import os
import warnings
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.models import Model
from keras import layers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from xception_padding import Xception
# define the u-net
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation,UpSampling2D, Conv2D, BatchNormalization, Activation, concatenate, Add
from keras.layers.core import Lambda, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras import backend as K
from keras import metrics
import tensorflow as tf
from keras.utils import get_file
from  keras.engine import Layer

import keras
def predprob(x, y, initial_lexsort=True):
    """
    Calculates the prediction probability. Adapted from scipy's implementation of Kendall's Tau

    Note: x should be the truth labels.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they will
        be flattened to 1-D.
    initial_lexsort : bool, optional
        Whether to use lexsort or quicksort as the sorting method for the
        initial sort of the inputs. Default is lexsort (True), for which
        `predprob` is of complexity O(n log(n)). If False, the complexity is
        O(n^2), but with a smaller pre-factor (so quicksort may be faster for
        small arrays).
    Returns
    -------
    Prediction probability : float

    Notes
    -----
    The definition of prediction probability that is used is::
      p_k = (((P - Q) / (P + Q + T)) + 1)/2
    where P is the number of concordant pairs, Q the number of discordant
    pairs, and T the number of ties only in `x`.
    References
    ----------
    Smith W.D, Dutton R.C, Smith N.T. (1996) A measure of association for assessing prediction accuracy
    that is a generalization of non-parametric ROC area. Stat Med. Jun 15;15(11):1199-215
    """

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if not x.size or not y.size:
        return (np.nan, np.nan)  # Return NaN if arrays are empty

    n = np.int64(len(x))
    temp = list(range(n))  # support structure used by mergesort
    # this closure recursively sorts sections of perm[] by comparing
    # elements of y[perm[]] using temp[] as support
    # returns the number of swaps required by an equivalent bubble sort

    def mergesort(offs, length):
        exchcnt = 0
        if length == 1:
            return 0
        if length == 2:
            if y[perm[offs]] <= y[perm[offs+1]]:
                return 0
            t = perm[offs]
            perm[offs] = perm[offs+1]
            perm[offs+1] = t
            return 1
        length0 = length // 2
        length1 = length - length0
        middle = offs + length0
        exchcnt += mergesort(offs, length0)
        exchcnt += mergesort(middle, length1)
        if y[perm[middle - 1]] < y[perm[middle]]:
            return exchcnt
        # merging
        i = j = k = 0
        while j < length0 or k < length1:
            if k >= length1 or (j < length0 and y[perm[offs + j]] <=
                                                y[perm[middle + k]]):
                temp[i] = perm[offs + j]
                d = i - j
                j += 1
            else:
                temp[i] = perm[middle + k]
                d = (offs + i) - (middle + k)
                k += 1
            if d > 0:
                exchcnt += d
            i += 1
        perm[offs:offs+length] = temp[0:length]
        return exchcnt

    # initial sort on values of x and, if tied, on values of y
    if initial_lexsort:
        # sort implemented as mergesort, worst case: O(n log(n))
        perm = np.lexsort((y, x))
    else:
        # sort implemented as quicksort, 30% faster but with worst case: O(n^2)
        perm = list(range(n))
        perm.sort(key=lambda a: (x[a], y[a]))

    # compute joint ties
    first = 0
    t = 0
    for i in range(1, n):
        if x[perm[first]] != x[perm[i]] or y[perm[first]] != y[perm[i]]:
            t += ((i - first) * (i - first - 1)) // 2
            first = i
    t += ((n - first) * (n - first - 1)) // 2

    # compute ties in x
    first = 0
    u = 0
    for i in range(1,n):
        if x[perm[first]] != x[perm[i]]:
            u += ((i - first) * (i - first - 1)) // 2
            first = i
    u += ((n - first) * (n - first - 1)) // 2

    # count exchanges
    exchanges = mergesort(0, n)
    # compute ties in y after mergesort with counting
    first = 0
    v = 0
    for i in range(1,n):
        if y[perm[first]] != y[perm[i]]:
            v += ((i - first) * (i - first - 1)) // 2
            first = i
    v += ((n - first) * (n - first - 1)) // 2

    tot = (n * (n - 1)) // 2
    if tot == u or tot == v:
        return (np.nan, np.nan)    # Special case for all ties in both ranks

    p_k = (((tot - (v + u - t)) - 2.0 * exchanges) / (tot - u) + 1)/2

    return p_k
def tau_accuracy(y_true, y_pred):
    tau,p_tau=stats.kendalltau(y_true, y_pred)
    #print(tau)
    tau= k.variable(value=tau, dtype='float32')
    #sess.run(tf.global_variables_initializer())
    return k.abs(tau)
def add_image_mask(img,mask):
    mask=mask/255
    mask=np.reshape(mask,(256,256,1))
    mask=np.concatenate((mask,mask,mask),axis=2)
    if np.max(mask+img)!=0:
        img=(mask+img)/np.max(mask+img)
    else:
        img=0
    return img
train_mask_path='/data5/badcode/_temp/cell_counting_v2/lzq_train/pred_lab_0.3/'
valid_mask_path='/data5/badcode/_temp/cell_counting_v2/lzq_train/validation_mask/'
test_mask_path='/data5/badcode/_temp/cell_counting_v2/lzq_train/test_mask/'
train_image_path='/data5/image/breastpathq/datasets/train/'
test_image_path='/data5/image/breastpathq/datasets/test/'
valid_image_path='/data5/image/breastpathq/datasets/validation/'

pre_mode_path='/data4/lzq/xception_0.82.h5'
save_model_path='/home/lzq/tf/spie2019/model_valid/'
train_csv_path='/data5/image/breastpathq/datasets/train_labels.csv'
valid_csv_path='/data5/image/breastpathq/datasets/valid_labels.csv'
test_csv_path='/data5/image/breastpathq/datasets/Sample_Submission_Updated.csv'
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
config = tf.ConfigProto()  
config.gpu_options.per_process_gpu_memory_fraction=True #不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)
def rgb_clahe_justl(x):

    lab = skimage.color.rgb2lab(x) 
    lab = (lab + [0, 128, 128])/[100, 255, 255]
    #print(np.shape(lab))
    l=exposure.equalize_adapthist(lab[:,:,0], kernel_size=8, clip_limit=0.01, nbins=256)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #l= clahe.apply(lab[:,:,0])
    lab[:,:,0]=l
        
    return lab
filename=[]
tain_imgs=[]
test_imgs=[]
id_list=[]
train_imgs=[]
train_score_list=[]
train_masks=[]
valid_imgs=[]
valid_masks=[]
valid_score_list=[]
test_imgs=[]
test_masks=[]
train_csv_reader = csv.reader(open(train_csv_path))
valid_csv_reader=csv.reader(open(valid_csv_path))
test_csv_reader=csv.reader(open(test_csv_path))
slides=[]
rids=[]
for i,row in enumerate(train_csv_reader):
    if(i>0):
        id=row[0]+'_'+row[1]
        #id_list.append(id)
        img=imread(train_image_path+id+'.tif')
        img=transform.resize(img, (IMG_HEIGHT, IMG_WIDTH,3),mode='reflect')
        train_imgs.append(img)
for i,row in enumerate(valid_csv_reader):
    if(i>0):
        id=row[0]+'_'+row[1]
        slides.append(row[0])
        rids.append(row[1])
        img=imread(valid_image_path+id+'.tif')
        img=transform.resize(img, (IMG_HEIGHT, IMG_WIDTH,3),mode='reflect')

        #valid_imgs.append(img)
        mask_id=id+'_mask.tif'
        mask=imread(valid_mask_path+mask_id)
        mask = ndimage.gaussian_filter(mask, sigma=(2, 2), order=0)
        mask=np.reshape(mask,(256,256,1))
        valid_imgs.append(img)
        valid_masks.append(mask)

for i,row in enumerate(test_csv_reader):
    if(i>0):
        id=row[0]+'_'+row[1]
        #slides.append(row[0])
        #rids.append(row[1])
        #id_list.append(id)
        img=imread(test_image_path+id+'.tif')
        img=transform.resize(img, (IMG_HEIGHT, IMG_WIDTH,3),mode='reflect')
        mask_id=id+'_mask.tif'
        mask=imread(test_mask_path+mask_id)
        mask = ndimage.gaussian_filter(mask, sigma=(2, 2), order=0)
        mask=np.reshape(mask,(256,256,1))
        test_imgs.append(img)
        test_masks.append(mask)
print(len(train_imgs))
print(len(valid_imgs))
print(len(test_imgs))
#imgs=np.concatenate((train_imgs,valid_imgs,test_imgs),axis=0)

#print(len(imgs))

print('success')

def weighted_binary_crossentropy(y_true, y_pred):
    '''
    Calculates the weighted pixel-wise binary cross entropy. Expects target to be encoded as `(mask - weights)`. 
    '''
    # mask <- where value==0
    target_mask = K.cast(K.equal(y_true, 0), 'float32') 
    
    # weights calculated as described above
    weights = (-1 * y_true) + target_mask
    
    cce = binary_crossentropy(target_mask, y_pred)  
    wcce = cce * K.squeeze(weights, axis=-1)
    return K.mean(wcce, axis=-1)


def huber_loss(y_true, y_pred):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    clip_value=1
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if condition==True:
        return K.mean(squared_loss)
    else:
        return K.mean(linear_loss)
    
    '''
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))
    '''
def mycrossentropy(y_true, y_pred, e=0.1):
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/4, y_pred)
    return (1-e)*loss1 + e*loss2
def dice(y_true,y_pred,smooth=1e-3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
def dice_loss(y_true,y_pred):
    return 1-dice(y_true,y_pred)
def softmax_dice_loss(y_true,y_pred):
    return weighted_binary_crossentropy(y_true, y_pred) * 0.7 + dice_loss(y_true, y_pred) * 0.3
def conv_block(inputs, filters,name,filter_size=3, drop_prob=0.2, regularizer=regularizers.l2(0.0001)):
    x = Conv2D(filters, filter_size, padding='same', kernel_regularizer=regularizer,name=name+'_1')(inputs)
    x = BatchNormalization(name=name+'_bn_1')(x)
    x = Activation('relu',name=name+'_act_1')(x)
    x = Conv2D(filters, filter_size, padding='same', kernel_regularizer=regularizer,name=name+'_2')(x)
    x = BatchNormalization(name=name+'_bn_2')(x)
    x = Activation('relu',name=name+'_act_2')(x)
    x = SpatialDropout2D(drop_prob,name=name+'_drop_1')(x)
    return x

def downsample(block):
    x = MaxPooling2D(pool_size=(2, 2)) (block)
    return x

def upsample(block, skip_connection, filters, regularizer=regularizers.l2(0.0001)):
    x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizer)(block)
    stack = concatenate([skip_connection, x])
    return stack
def build_unet(IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=1, drop_prob=0.2):
    
    regularizer=regularizers.l2(0.0001)
    # ---- Model ----
    mask_input = Input((IMG_HEIGHT, IMG_WIDTH, 1),name="input_2")
    #add noise
    # Downsample
    encode_1 = conv_block(mask_input, 16, regularizer=regularizer,name='conv2d_2_1')
     #add noise
    stddev=0.05
    #encode_1=keras.layers.noise.GaussianNoise(stddev)(encode_1)
    down_1 = downsample(encode_1)
    
    encode_2 = conv_block(down_1, 32, regularizer=regularizer,name='conv2d_2_2')
    down_2 = downsample(encode_2)
    
    encode_3 = conv_block(down_2, 64, regularizer=regularizer,name='conv2d_2_3')
    down_3 = downsample(encode_3)
    
    encode_4 = conv_block(down_3, 128, regularizer=regularizer,name='conv2d_2_4')
    down_4 = downsample(encode_4)
    bridge = conv_block(down_4, 256, regularizer=regularizer,name='conv2d_2_5')
    end_2=GlobalAveragePooling2D()(bridge)
    return end_2
def tau_accuracy(y_true,y_pred):
   tau,p_tau=stats.kendalltau(y_true, y_pred)
   return abs(tau)

mean = np.mean(train_imgs)
std = np.std(train_imgs)
valid_imgs = (valid_imgs - mean) / std



valid_imgs=np.array(valid_imgs)
valid_masks=np.array(valid_masks)
batch_size=1
test_steps = np.ceil(len(valid_imgs) / batch_size)
model=load_model(save_model_path+'5add_mask_0.89pre_logcosh_loss_lr=0.0005(0.9173).h5', custom_objects={'mean_squared_error': mean_squared_error})
model.load_weights(save_model_path+'5add_mask_0.89pre_logcosh_loss_lr=0.0005(0.9173).h5')
#tests=model.predict(imgs)

tests=model.predict({'input_1': valid_imgs, 'input_2': valid_masks})


scoles=[]
tests=np.reshape(tests,185)
print(np.shape(tests))
for test in tests:
    test=round(test,3)
    scoles.append(test)
print(scoles)
scoles=np.array(scoles)
print(np.shape(scoles))
import pandas as pd
import csv
df=pd.DataFrame({'slide':slides, 'rid':rids ,'p':scoles})
columns = ['slide','rid','p']
df.to_csv(save_model_path+'[iceli]_Results.csv',header=True,index=False,columns=columns)



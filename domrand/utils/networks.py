from tensorflow.python.layers import base
import tensorflow as tf
import numpy as np
from functools import partial

# TODO: make activation parameterized in all these functions

def ssam(conv_layer):
    """spatial soft argmax"""
    _, num_rows, num_cols, num_fp = conv_layer.get_shape()
    num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
    x_map = np.empty([num_rows, num_cols], np.float32)
    y_map = np.empty([num_rows, num_cols], np.float32)
    
    for i in range(num_rows):
        for j in range(num_cols):
            x_map[i, j] = (i - num_rows / 2.0) / num_rows
            y_map[i, j] = (j - num_cols / 2.0) / num_cols
    
    x_map = tf.convert_to_tensor(x_map)
    y_map = tf.convert_to_tensor(y_map)
    
    x_map = tf.reshape(x_map, [num_rows * num_cols])
    y_map = tf.reshape(y_map, [num_rows * num_cols])
    
    # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
    features = tf.reshape(tf.transpose(conv_layer, [0,3,1,2]),
                          [-1, num_rows*num_cols])
    softmax = tf.nn.softmax(features)
    
    fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
    fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)
    
    conv_out_flat = tf.reshape(tf.concat(axis=1, values=[fp_x, fp_y]), [-1, num_fp*2])
    return conv_out_flat

def bn_conv(inputs, **kwargs):
    """Batch norm convolution layer"""
    out = tf.layers.conv2d(inputs, **kwargs)
    out = tf.layers.batch_normalization(out, training=True, fused=True)
    out = tf.nn.relu(out)
    return out

def flatten(inputs, use_ssam):
    """Flatten conv out using reshape or spatial soft argmax"""
    if use_ssam:
        out = ssam(inputs)
        #out = tf.contrib.layers.spatial_softmax(inputs, name='ssam')
    else:
        out = tf.layers.flatten(inputs)
    return out 

def binned_head(inputs, outdim, hparams):
    """Output of binned network, make heads for xyz and stack them"""
    outx = tf.layers.dense(inputs, outdim, activation=None)
    outy = tf.layers.dense(inputs, outdim, activation=None)
    outz = tf.layers.dense(inputs, outdim, activation=None)

    out = tf.stack([outx, outy, outz], 1)
    return out

def trivial_forward(inputs, outdim, hparams):
    """for testing speed"""
    out = tf.contrib.layers.spatial_softmax(inputs)
    out = tf.layers.dense(out, outdim)
    return out

# TODO: convert all the extra args to a dictionary
#s2_layers=None, s1_layers=None, fc_layers=None, batch_norm=False, ssam=False
def vgg_forward(inputs, outdim, hparams):
    """Forward pass of VGG network"""
    maxpool = partial(tf.layers.max_pooling2d, pool_size=2, strides=(2,2), padding='SAME')
    conv2d = partial(tf.layers.conv2d, kernel_size=3, strides=(1,1), padding='SAME', activation=tf.nn.relu) # uses 3x3, stride 1, zero-padding throughout
    vgg_bn_conv = partial(bn_conv, kernel_size=3, strides=(1,1), padding='SAME', activation=tf.nn.relu)
    conv = vgg_bn_conv if hparams['batch_norm'] else conv2d # add batch norm

    out = inputs

    out = conv(inputs=out, filters=64)
    out = conv(inputs=out, filters=64)
    out = maxpool(inputs=out)

    out = conv(inputs=out, filters=128)
    out = conv(inputs=out, filters=128)
    out = maxpool(inputs=out)

    out = conv(inputs=out, filters=256)
    out = conv(inputs=out, filters=256)
    out = conv(inputs=out, filters=256)
    out = maxpool(inputs=out)

    out = conv(inputs=out, filters=512)
    out = conv(inputs=out, filters=512)
    out = conv(inputs=out, filters=512)
    out = maxpool(inputs=out)

    out = conv(inputs=out, filters=512)
    out = conv(inputs=out, filters=512)
    out = conv(inputs=out, filters=512)
    out = maxpool(inputs=out)

    out = flatten(out, hparams['ssam'])

    out = tf.layers.dense(out, 256, tf.nn.relu)
    out = tf.layers.dense(out, 64, tf.nn.relu)

    if hparams['output'] == 'xyz': 
        out = tf.layers.dense(out, outdim, kernel_initializer=None)
    elif hparams['output'] == 'binned':
        out = binned_head(out, outdim, hparams)
    return out

def reg_forward(inputs, outdim, hparams):
    """
    Forward pass of neural network

    Parameterized versions of:
    input
    3x3 with stride 2
    3x3 with stirde 1
    flatten
    fc
    out
    """
    conv2d = partial(tf.layers.conv2d, kernel_size=3, padding='SAME', activation=tf.nn.relu) # uses 4x3, stride 1, zero-padding throughout
    bn_conv2d = partial(bn_conv, kernel_size=3, padding='SAME', activation=tf.nn.relu)
    conv = bn_conv2d if hparams['batch_norm ']else conv2d # add batch norm

    out = inputs

    for s2_layer in hparams['s2_layers']:
        out = conv(out, s2_layer, 3, strides=(2,2), padding='SAME')
    for s1_layer in hparams['s1_layers']:
        out = conv(out, s1_layer, 3, strides=(1,1), padding='SAME')

    out = flatten(out, hparams['ssam'])

    for fc_layer in hparams['fc_layers']:
        out = tf.layers.dense(out, fc_layer, activation=tf.nn.relu)

    out = tf.layers.dense(out, outdim)
    return out

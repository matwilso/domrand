import numpy as np
import tensorflow as tf

from domrand.define_flags import FLAGS
from domrand.utils.general import bin_to_xyz_tf
from domrand.utils.networks import reg_forward, vgg_forward, trivial_forward

activ = {'relu': tf.nn.relu, 'tanh': tf.tanh}[FLAGS.activ]
forward = {'reg': reg_forward, 'resnet': None, 'vgg': vgg_forward, 'trivial':trivial_forward}[FLAGS.arch]

def xyz_cross_entropy_loss(onehots, preds):
    xloss = tf.losses.softmax_cross_entropy(onehot_labels=onehots[:,0,:], logits=preds[:,0,:], label_smoothing=FLAGS.label_smoothing)
    yloss = tf.losses.softmax_cross_entropy(onehot_labels=onehots[:,1,:], logits=preds[:,1,:], label_smoothing=FLAGS.label_smoothing)
    zloss = tf.losses.softmax_cross_entropy(onehot_labels=onehots[:,2,:], logits=preds[:,2,:], label_smoothing=FLAGS.label_smoothing)
    return xloss, yloss, zloss

class Model(object):
    def __init__(self, t_inputs, t_labels, e_inputs, e_labels, global_step):
        raise NotImplementedError

    def _model_init(self, global_step):
        """call this at end of subclass init"""
        # just for logging
        self.euc = tf.reduce_mean(tf.norm(self.label_xyz[:,:2]-self.pred_xyz[:,:2], axis=1))

        self.lr_ph = tf.placeholder(tf.float32, shape=None)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_ph)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.minimize_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

class BinnedModel(Model):
    def __init__(self, inputs, labels, global_step):
        net_forward = lambda x: forward(x, FLAGS.coarse_bin, FLAGS.flag_values_dict())
        BINS = tf.constant(FLAGS.coarse_bin, dtype=tf.int32)

        self.label_onehots = tf.one_hot(labels, BINS)
        self.label_xyz = bin_to_xyz_tf(labels, BINS)

        with tf.variable_scope('model', reuse=False):
            self.preds = net_forward(inputs)

        self.am_preds = tf.argmax(self.preds, -1)
        self.pred_xyz = bin_to_xyz_tf(self.am_preds, BINS)
        with tf.variable_scope('loss'):
            xloss, yloss, zloss = xyz_cross_entropy_loss(self.label_onehots, self.preds)
            self.loss = xloss + yloss + zloss

        super()._model_init(global_step)


class XYZModel(Model):
    def __init__(self, inputs, labels, global_step):
        net_forward = lambda x: forward(x, 3, FLAGS.flag_values_dict())
        self.dlabel_xyz = labels

        with tf.variable_scope('model', reuse=False):
            self.pred_xyz = self.preds = net_forward(inputs)

        # loss for training
        self.loss = tf.losses.mean_squared_error(labels, self.preds)

        super()._model_init(global_step)

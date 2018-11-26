import os
import multiprocessing
import sys
import itertools
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from domrand.define_flags import FLAGS
from domrand.utils.models import XYZModel, BinnedModel
from domrand.utils.general import notify
from domrand.utils.data import load_eval_data, parse_record, brighten_image, bin_label
from domrand.utils.image import make_pred_plot
#from domrand.utils.constants import MEAN_TENSOR

# TODO: Try making this a Trainer class (I think this would let us reuse a bit more)

def train_simple():
    ncpu = multiprocessing.cpu_count()
    global_step = tf.get_variable('global_step',
                                  initializer=tf.constant(0, dtype=tf.int64),
                                  trainable=False)
    global_epoch = tf.get_variable('global_epoch',
                                  initializer=tf.constant(1, dtype=tf.int64),
                                  trainable=False)
    # make these loadable from file, so we don't start over when killing and rerunning a script
    # TODO: not sure these work as intended
    global_lr = tf.get_variable('global_lr',
                                  initializer=tf.constant(FLAGS.lr, dtype=tf.float32),
                                  trainable=False)
    global_bs = tf.get_variable('global_bs',
                                  initializer=tf.constant(FLAGS.bs, dtype=tf.int64),
                                  trainable=False)

    EVAL_BS = 64
    PLOT_SHAPE = [None, 480, 640, 3]

    file_eval_imgs, file_eval_labels = load_eval_data(FLAGS.real_data_path, FLAGS.real_data_shape)
    eval_img_ph = tf.placeholder(file_eval_imgs.dtype, file_eval_imgs.shape)
    eval_label_ph = tf.placeholder(file_eval_labels.dtype, file_eval_labels.shape)
    pred_plot_ph = tf.placeholder(file_eval_imgs.dtype, PLOT_SHAPE)
    eval_dataset = tf.data.Dataset.from_tensor_slices((eval_img_ph, eval_label_ph))
    if FLAGS.output == 'binned':
        eval_dataset = eval_dataset.map(bin_label, num_parallel_calls=ncpu) 
    eval_dataset = eval_dataset.batch(EVAL_BS)

    rand_files = lambda : np.random.choice(FLAGS.filenames, len(FLAGS.filenames), replace=False)
    filenames_ph = tf.placeholder(tf.string, shape=[None])
    train_dataset = tf.data.TFRecordDataset(filenames_ph, compression_type="GZIP")
    train_dataset = train_dataset.map(parse_record, num_parallel_calls=ncpu) 
    if FLAGS.output == 'binned':
        train_dataset = train_dataset.map(bin_label, num_parallel_calls=ncpu) 
    #train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(global_bs)
    train_dataset = train_dataset.map(brighten_image, num_parallel_calls=ncpu)
    train_dataset = train_dataset.prefetch(2)


    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    d_image, d_label = iterator.get_next()
    d_rebuilt_image = 0.5*(d_image + 1)

    train_iter_op = iterator.make_initializer(train_dataset)
    eval_iter_op = iterator.make_initializer(eval_dataset)

    Model = BinnedModel if FLAGS.output == 'binned' else XYZModel
    model = Model(d_image, d_label, global_step=global_step)

    inc_step = tf.assign_add(global_step, tf.constant(1, dtype=tf.int64))
    inc_epoch = tf.assign_add(global_epoch, tf.constant(1, dtype=tf.int64))
    anneal_lr = tf.assign(global_lr, tf.multiply(global_lr, tf.constant(FLAGS.lr_anneal, dtype=tf.float32)))
    anneal_bs = tf.assign(global_bs, tf.cast(tf.multiply(tf.cast(global_bs, dtype=tf.float32), tf.constant(FLAGS.bs_anneal, dtype=tf.float32)), dtype=tf.int64))
    saver = tf.train.Saver()

    with tf.name_scope('train'):
        tf.summary.image('image', d_rebuilt_image)
        tf.summary.scalar('mean_euclidean', model.euc)
        tf.summary.scalar('mean_loss', model.loss)
        tf.summary.scalar('zlearning_rate', global_lr)
    train_summaries = tf.summary.merge_all(scope='train')

    with tf.name_scope('aviz_train'):
        viz_train_summary = tf.summary.image('viz', pred_plot_ph, max_outputs=10)
    with tf.name_scope('aviz_eval'):
        viz_eval_summary = tf.summary.image('viz', pred_plot_ph, max_outputs=60)

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(FLAGS.checkpoint + '/train', sess.graph)
        eval_writer = tf.summary.FileWriter(FLAGS.checkpoint + '/eval')
        sess.run(tf.global_variables_initializer())
        sess.run(train_iter_op, {filenames_ph: rand_files()})
        if tf.train.latest_checkpoint(FLAGS.checkpoint):
            print('Restoring from checkpoint...')
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint))

        print("~{} params".format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        print('Training...')
        step = sess.run(global_step) # just init it. it may get set to different value immediately
        epoch = sess.run(global_epoch)

        
        early_stop_counter = 0 # counter to make sure we're still improving
        best_eval_euc = 99
        # TODO: convert this into a progress bar with finite run time, so I can see how long things take
        while True:
            lr = sess.run(global_lr)
            bs = sess.run(global_bs)
            losses = []
            eucs = []
            try:
                total_steps_per_epoch = ((len(FLAGS.filenames) * 1000) // bs)
                pbar = tqdm.tqdm(total_steps_per_epoch+1, unit='step', desc='Epoch step progress')
                for i in range(total_steps_per_epoch+2):
                    # Train
                    _, train_loss, train_euc, train_preds, labels, step = sess.run([model.minimize_op, model.loss, model.euc, model.preds, d_label, inc_step], feed_dict={model.lr_ph: lr})
                    if FLAGS.output == 'binned' and (np.count_nonzero(labels >= FLAGS.coarse_bin) or np.count_nonzero(labels < 0)):
                        # this is bad. should not happen. 
                        import ipdb; ipdb.set_trace()
                    losses.append(train_loss)
                    eucs.append(train_euc)
                    pbar.update(1)
                assert False, "we should never get down here"

            except tf.errors.OutOfRangeError:
                pbar.close()

                # Train summary
                sess.run(train_iter_op, {filenames_ph: rand_files()}) # reset so we can grab some summaries
                train_images, train_preds, train_labels, summary = sess.run([d_rebuilt_image, model.preds, d_label, train_summaries], feed_dict={model.lr_ph: lr})
                train_writer.add_summary(summary, global_step=epoch*len(FLAGS.filenames))

                # Eval summary 
                sess.run(eval_iter_op, feed_dict={eval_img_ph: file_eval_imgs, eval_label_ph: file_eval_labels}) # swap to eval dataset
                eval_loss, eval_euc, eval_images, eval_preds, eval_labels, eval_summary = sess.run([model.loss, model.euc, d_rebuilt_image, model.preds, d_label, train_summaries])
                eval_writer.add_summary(eval_summary, global_step=epoch*len(FLAGS.filenames))

                # make sure we are still improving
                if eval_euc > best_eval_euc:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
                    best_eval_euc = eval_euc

                if FLAGS.plot_preds:
                    # takes about 2s 
                    # EVAL VIZ
                    zipped = list(zip(eval_images, eval_preds, eval_labels))
                    #idxs = np.linspace(0,50,50).astype(int)
                    #zipped = [zipped[idx] for idx in idxs]
                    eval_plots = np.array([make_pred_plot(img, pred, label, mode=FLAGS.output) for (img, pred, label) in zipped])
                    plot_summary = sess.run([viz_eval_summary], {pred_plot_ph:eval_plots})[0]
                    eval_writer.add_summary(plot_summary, global_step=epoch*len(FLAGS.filenames))
                    # TRAIN VIZ
                    zipped = list(zip(train_images[:3], train_preds[:3], train_labels[:3]))
                    eval_plots = np.array([make_pred_plot(img, pred, label, mode=FLAGS.output) for (img, pred, label) in zipped])
                    plot_summary = sess.run([viz_train_summary], {pred_plot_ph:eval_plots})[0]
                    eval_writer.add_summary(plot_summary, global_step=epoch*len(FLAGS.filenames))

                train_loss = np.mean(losses); losses = []
                train_euc = np.mean(eucs); eucs = []
                log_string = 'epoch {}: train_loss = {:0.3f} eval_loss = {:0.3f} train_euc = {:0.3f} eval_euc = {:0.3f}'.format(epoch, train_loss, eval_loss, train_euc, eval_euc); print(log_string)

                if FLAGS.more_notify:
                    notify(log_string)

                if FLAGS.anneal_interval and epoch % FLAGS.anneal_interval == 0:
                    sess.run([anneal_lr, anneal_bs])

                # ready to move to next epoch
                _, epoch = sess.run([train_iter_op, inc_epoch], {filenames_ph: rand_files()}) # reset so we can grab some summaries  

                savepath = os.path.join(FLAGS.checkpoint, 'ckpt')

                #if FLAGS.serve:
                #    tf.saved_model.simple_save(sess, savepath+'model'+str(epoch), inputs={'image': d_image}, outputs={'prediction': model.preds})

                print('Saved to {}'.format(savepath))
                saver.save(sess, savepath, global_step=epoch)
                if FLAGS.num_epochs is not None and epoch > FLAGS.num_epochs:
                    return dict(train_euc=train_euc, eval_euc=best_eval_euc)
                elif early_stop_counter >= 4:
                    return dict(train_euc=train_euc, eval_euc=best_eval_euc)
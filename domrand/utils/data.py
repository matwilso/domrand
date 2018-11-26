import yaml
import time
import os
from tqdm import trange
import tensorflow as tf
import numpy as np
from tensorflow.python.lib.io.tf_record import TFRecordOptions, TFRecordCompressionType, TFRecordWriter
import matplotlib.pyplot as plt
from domrand.utils.image import preproc_image
from domrand.utils.constants import OBJ_DZ, TABLE_GRID_OFFX, TABLE_GRID_OFFY,  \
    GRID_SPACING, TABLE_WX, TABLE_WY, TBS_TF, TBE_TF

"""
Data pipeline utils
"""

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_data(sim_manager, data_path):
    """Make some domrand data and save it to tfrecords."""
    image, label = sim_manager.get_data()

    rows = image.shape[0]
    cols = image.shape[1]
    depth = image.shape[2]

    # OUTER = 5e2
    # INNER = 1e3
    # generate 1e5 examples, spread across 1e2 files
    print()
    print('Generating 1e5 examples (~30 GB). You can ctrl-c anytime you want')
    print()
    for i in trange(int(1e2), desc='Files created'):
        date_string = time.strftime('%Y-%m-%d-%H-%M-%S')
        filename = os.path.join(data_path, date_string + '.tfrecords')
        with TFRecordWriter(filename, options=TFRecordOptions(TFRecordCompressionType.GZIP)) as writer:
            try:
                for j in trange(int(1e3), desc='Examples generated'):
                    image, label = sim_manager.get_data()
                    assert image.dtype == np.uint8
                    image_raw = image.tostring()
                    label_raw = label.astype(np.float32).tostring()

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'label_raw': _bytes_feature(label_raw),
                                'image_raw': _bytes_feature(image_raw)
                            }))
                    writer.write(example.SerializeToString())
            except:
                writer.close()
                os.remove(filename)

    writer.close()

def parse_record(args):
    """parse record function for loading data from file. should get called as first dataset.map() function"""
    features = {'label_raw': tf.FixedLenFeature((), tf.string),
                'image_raw': tf.FixedLenFeature((), tf.string),
    }
    parsed = tf.parse_single_example(args, features)

    image = tf.cast(tf.reshape(tf.decode_raw(parsed['image_raw'], tf.uint8), (224, 224, 3)), tf.float32)
    image = (image / 127.5) - 1.0

    label = tf.decode_raw(parsed['label_raw'], tf.float32)
    return image, label

# NOTE: FLAGS are used here because it is hard to pass arguments to these functions, but it makes
# them reliant on running from command line

def bin_label(image, label):
    from domrand.define_flags import FLAGS
    frac = (label - TBS_TF) / (TBE_TF - TBS_TF)
    assert_op = tf.Assert(tf.logical_not(tf.reduce_any(tf.logical_or(frac < -0.1, frac > 1.1))), [label, frac]) # if this triggers, the label was way off the table

    with tf.control_dependencies([assert_op]):
        frac = tf.clip_by_value(frac, 0, 0.99999) # never want it to be 1
        binned_label = tf.cast(frac * FLAGS.coarse_bin, tf.int32)
    return image, binned_label 

def brighten_image(image, label):
    """Add random brightness to image to better match the distribution"""
    from domrand.define_flags import FLAGS
    delta = tf.random_uniform(shape=[], minval=FLAGS.minval, maxval=FLAGS.maxval, dtype=tf.float32)
    image = tf.image.adjust_brightness(image, delta)

    if FLAGS.random_noise:
        noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=FLAGS.noise_std, dtype=tf.float32)  
        image = image + noise

    image = tf.clip_by_value(image, -1.0, 1.0)

    return image, label

def get_real_cam_pos(real_data_path):
    """Return xyz numpy array of camera position of associated data"""
    meta_file = os.path.join(real_data_path, 'metadata.yaml')
    metadata = yaml.load(open(meta_file, 'r'))['camera']
    cam_pos = np.array([metadata['x'], metadata['y'], metadata['z']])
    return cam_pos

# TODO: probably fix this data pipeline to be more like the other. Maybe write a script to save to TFRecords
def load_eval_data(eval_data_path, data_shape='asus'):
    """Read the data from the real world stored in jpgs"""
    imgs = []
    labels = []
    files = sorted(os.listdir(eval_data_path))
    for f in files: 
        if not f.endswith('.jpg'):
            continue

        x, y = map(int, f[:-4].split('-')) # grab xy coords from x-y.jpg named file

        dx = TABLE_GRID_OFFX - x*GRID_SPACING
        dy = TABLE_GRID_OFFY - y*GRID_SPACING
        dz = OBJ_DZ
        dobj = np.array([dx, dy, dz])
        
        filename = os.path.join(eval_data_path, f)
        img = plt.imread(filename)
        if data_shape == 'asus':
            img = preproc_image(img, dtype=np.float32)
        elif data_shape == 'kinect2':
            img = preproc_image(img[:,240:-240,:], dtype=np.float32)

        img = (img / 127.5) - 1.0
        #plt.imshow(img); plt.show()

        imgs.append(img)
        labels.append(dobj)

    return np.array(imgs, dtype=np.float32), np.array(labels, dtype=np.float32)



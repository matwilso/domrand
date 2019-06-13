import os
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

# TODO: add meta-flags that cover several hyperamaters so that we can easily change them all at once instead of setting them individually

# DomRand flags
flags.DEFINE_string('xml', default='xmls/kuka/lbr4_reflex.xml', help='')
flags.DEFINE_bool('gpu_render', default=True, help='Called gpu_render because it makes rendering fast on gpu, but I think you can also use this on cpu')
flags.DEFINE_bool('gui', default=False, help='')
flags.DEFINE_bool('display_data', default=False, help='')
flags.DEFINE_string('look_at', default='robot_table_link', help='')

# Data and logging
flags.DEFINE_string('data_path', default='./data/sim/', help='')
flags.DEFINE_string('real_data_path', default='./data/real/', help='')
flags.DEFINE_string('real_data_shape', default='asus', help='')
flags.DEFINE_string('checkpoint', default='checkpoint', help='checkpoint directory')
flags.DEFINE_string('manual_checkpoint', default=None, help='instead of tacking on extra terms, use their exact path')
flags.DEFINE_string('logpath', default='./logs', help='log directory')
flags.DEFINE_bool('serve', default=False, help='export the model to allow for tensorflow serving')
flags.DEFINE_integer('num_files', default=None, help='')
flags.DEFINE_integer('shuffle_files', default=0, help='')
flags.DEFINE_integer('num_epochs', default=None, help='')
flags.DEFINE_list('filenames', default=None, help='')
flags.DEFINE_bool('notify', default=False, help='notify on end')
flags.DEFINE_bool('more_notify', default=False, help='notify on epoch')
flags.DEFINE_bool('plot_preds', default=True, help='plot pred plots')
flags.DEFINE_bool('random_noise', default=True, help='random noise to output')
flags.DEFINE_float('maxval', default=0.1, help='random noise to output')
flags.DEFINE_float('minval', default=0.0, help='random noise to output')
flags.DEFINE_float('noise_std', default=0.02, help='random noise to output')

# Architecture
flags.DEFINE_string('arch', default='vgg', help='')
flags.DEFINE_string('output', default='binned', help='')
flags.DEFINE_integer('coarse_bin', default=64, help='')
#flags.DEFINE_string('loss/output', default='vgg', help='')
flags.DEFINE_bool('coord_all', default=False, help='always use coord convs')
flags.DEFINE_bool('batch_norm', default=False, help='')
flags.DEFINE_bool('ssam', default=False, help='spatial soft-argmax')
flags.DEFINE_bool('softmax', default=False, help='just softmax right before flattening (only when not using ssam)')
flags.DEFINE_string('activ', default='relu', help='activation function')
flags.DEFINE_string('suffix', default='', help='')

flags.DEFINE_list('s2_layers', default=[64, 64, 64, 64], help='conv layers of stride 2')
flags.DEFINE_list('s1_layers', default=[64], help='conv layers of stride 1')
flags.DEFINE_list('fc_layers', default=[100, 100, 100], help='fully connected layers')
flags.DEFINE_bool('zero_last', default=False, help='start weights in last layer as 0')
flags.DEFINE_float('label_smoothing', default=0.05, help='used for regularization in tf.losses.softmax...')

# Training
flags.DEFINE_integer('anneal_interval', default=None, help='')
flags.DEFINE_float('lr', default=1e-4, help='')
flags.DEFINE_float('lr_anneal', default=0.5, help='')
flags.DEFINE_integer('bs', default=64, help='')
flags.DEFINE_float('bs_anneal', default=1, help='')

FLAGS.s2_layers = list(map(int, FLAGS.s2_layers))
FLAGS.s1_layers = list(map(int, FLAGS.s1_layers))
FLAGS.fc_layers = list(map(int, FLAGS.fc_layers))



if FLAGS.filenames is None:
    data_path = FLAGS.data_path
    data_files = os.listdir(data_path)
    FLAGS.filenames = [os.path.join(data_path, df) for df in data_files]

if FLAGS.num_files is not None:
    FLAGS.filenames = FLAGS.filenames[:FLAGS.num_files]


num_files_str = str(len(FLAGS.filenames))

# TODO: add label smoothing, sep folders for binned
settings_str = ''
settings_str += FLAGS.output
settings_str += '-'+FLAGS.arch
settings_str += '-ssam' if FLAGS.ssam else '-noss'
settings_str += '-soft' if FLAGS.softmax and not FLAGS.ssam else ''
settings_str += '-bn' if FLAGS.batch_norm else '-nb'
settings_str += '-{}'.format(FLAGS.suffix) if FLAGS.suffix is not '' else ''

if FLAGS.arch == 'reg':
    hp_str = '{}/{}/{}_2-{}_1-{}/{}-{}-{}-{}'.format(num_files_str, settings_str, FLAGS.s2_layers, FLAGS.s1_layers, FLAGS.fc_layers, FLAGS.lr, FLAGS.lr_anneal, FLAGS.anneal_interval, FLAGS.bs).replace(' ', '').replace(',','-').replace('[','').replace(']','')
else:
    hp_str = '{}/{}/{}-{}-{}-{}-{}'.format(num_files_str, settings_str, FLAGS.label_smoothing, FLAGS.lr, FLAGS.lr_anneal, FLAGS.anneal_interval, FLAGS.bs).replace(' ', '').replace(',','-').replace('[','').replace(']','')

if FLAGS.manual_checkpoint:
    FLAGS.checkpoint = FLAGS.manual_checkpoint
else:
    FLAGS.checkpoint = os.path.join(FLAGS.checkpoint, hp_str)

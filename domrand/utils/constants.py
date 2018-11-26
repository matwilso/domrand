import tensorflow as tf
import numpy as np


# taken from github:tianheyu/mil
#MEAN_IMAGE = np.array([103.939, 116.779, 123.68], np.float32) / 255.0
#MEAN_IMAGE = np.array([0, 0, 0], np.float32)
#MEAN_IMAGE = np.array([127.5]*3, np.float32)
MEAN_IMAGE = np.array([1.0]*3, np.float32)
MEAN_TENSOR = tf.constant(MEAN_IMAGE, tf.float32)
#SUB_IMAGE = np.array([127.5]*3, np.float32)
#SUB_TENSOR = tf.constant(SUB_IMAGE, tf.float32)

TOX = TABLE_GRID_OFFX = -0.5  # offset from robot center to the start of the X grid
TOY = TABLE_GRID_OFFY = 0.61  # ditto for Y
TWX = TABLE_WX = 0.76 # table xy dims
TWY = TABLE_WY = 1.22
OBJ_DZ = 0.33  # base (0.6) to table (0.89) + to middle of object (0.04)
GRID_SPACING = 0.15

TABLE_OFFSET_TF = tf.constant([TABLE_GRID_OFFX, TABLE_GRID_OFFY, 0], tf.float32)

TBSX = TOX
TBSY = TOY
TBSZ = 0.0
TBS = TABLE_BIN_START = np.array([TBSX, TBSY, TBSZ], np.float32)

TBEX = TOX - TWX
TBEY = TOY - TWY
TBEZ = 1.0
TBE = TABLE_BIN_END = np.array([TBEX, TBEY, TBEZ], np.float32)

TBS_TF = TABLE_BIN_START_TF = tf.constant(TBS, tf.float32)
TBE_TF = TABLE_BIN_END_TF = tf.constant(TBE, tf.float32)
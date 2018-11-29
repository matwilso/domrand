#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordOptions, TFRecordCompressionType, TFRecordWriter
from tensorflow.python.framework.errors import DataLossError
from domrand.define_flags import FLAGS
from domrand.utils.data import parse_record


def validate_dataset(filenames, reader_opts=None):
    """
    Attempt to iterate over every record in the supplied iterable of TFRecord filenames
    :param filenames: iterable of filenames to read
    :param reader_opts: (optional) tf.python_io.TFRecordOptions to use when constructing the record iterator
    """
    sess = tf.InteractiveSession()
    i = 0
    for fname in filenames:
        print('validating ', fname)

        record_iterator = tf.python_io.tf_record_iterator(path=fname, options=TFRecordOptions(TFRecordCompressionType.GZIP))
        try:
            checked = False
            for record in record_iterator:
                if not checked:
                    img, label = parse_record(record)

                    sess.run([img,label])
                    checked = True  # assume that all records in the same file will hold the same type of data
                i += 1
        except DataLossError as e:
            print('error in {} at record {}'.format(fname, i))
            print(e)


if __name__ == '__main__':
    validate_dataset(FLAGS.filenames)

#!/usr/bin/env python3
import os
import random
import tensorflow as tf
from domrand.define_flags import FLAGS
from domrand.trainer import train_simple
from domrand.utils.general import notify

def main():
    print(FLAGS.checkpoint)
    results = train_simple()

    logpath = os.path.join(FLAGS.logpath, '{:0.5f}.txt'.format(results['eval_euc']))

    with open(logpath, 'w+') as f:
        f.write(FLAGS.checkpoint)
    
    if FLAGS.notify:
        notify('Finished run. train: {} eval: {}'.format(results['train_euc'], results['eval_euc']))


if __name__ == '__main__':
    main()


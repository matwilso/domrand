#!/usr/bin/env python3
import os
import time

from domrand.define_flags import FLAGS
from domrand.sim_manager import SimManager
from domrand.utils.data import write_data

"""
GPU:
```
mjpython run_domrand.py # --gui 1 for viewing

CPU:
```
python run_domrand.py  --gpu_render 0
```
"""

def main():
    if FLAGS.gui:
        assert FLAGS.gpu_render,  "can't use gui without gpu_render"

    # Viewer is required to run GPU (just because I am too lazy to figure out why it stops working without it)
    # (Also, GPU is WAY faster)
    sim_manager = SimManager(filepath=FLAGS.xml, gpu_render=FLAGS.gpu_render, gui=FLAGS.gui, display_data=FLAGS.display_data)

    if not FLAGS.gui:
        write_data(sim_manager, FLAGS.data_path)
    else:
        while True:
            sim_manager._randomize()
            sim_manager._forward()

if __name__ == '__main__':
    main()






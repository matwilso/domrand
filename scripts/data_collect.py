#!/usr/bin/env python2
from __future__ import print_function
import os
import argparse
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

"""
Hacky script for collecting real world samples
(uses ros)

0. Plug in camera and launch it (for Asus: `roslaunch openni_launch openni.launch`)
1. Use rviz to look at camera image.
2. When ready to take, enter x, y to save the current image to a file

"""

rospy.init_node('test')
bridge = CvBridge()

def save_image(filename):
    raw_img = rospy.wait_for_message(img_topic, Image)
    cv_image = bridge.imgmsg_to_cv2(raw_img, 'bgr8')

    # Save the file
    # make dir if it doesn't exist
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    success = cv2.imwrite(filename, cv_image)
    assert success, "File write failed somehow"


parser = argparse.ArgumentParser(description='MAML')
parser.add_argument('--filepath', type=str, default='./data/real_world', help='')
parser.add_argument('--prefix', type=str, default=None, help='')
parser.add_argument('--camera', type=str, default='asus', help='')
FLAGS = parser.parse_args()

if FLAGS.camera == 'asus':
    img_topic = '/camera/rgb/image_raw'
else:
    img_topic = '/kinect2/hd/image_color'

print('Enter <x y> when ready to grab snapshot')
print()
while not rospy.is_shutdown():
    inp = raw_input('x y: ')
    x,y = inp[0], inp[1]

    if FLAGS.prefix is None:
        filename = '{}-{}.jpg'.format(x,y)
    else:
        filename = '{}-{}-{}.jpg'.format(FLAGS.prefix, x,y)

    full_path = os.path.join(FLAGS.filepath, filename) 
    save_image(full_path)
    print("Saved {}".format(full_path))

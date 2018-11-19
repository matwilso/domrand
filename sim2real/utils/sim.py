import os
import time
import numpy as np
import quaternion
import skimage
from mujoco_py import functions

normalize = lambda x: x / np.linalg.norm(x)

# MATH UTILS
def look_at(from_pos, to_pos):
    """Compute quaternion to point from `from_pos` to `to_pos`

    We define this ourselves, rather than using Mujoco's body tracker,
    because it makes it easier to randomize without relying on calling forward() 
    
    Reference: https://stackoverflow.com/questions/10635947/what-exactly-is-the-up-vector-in-opengls-lookat-function
    """
    up = np.array([0, 0, 1]) # I guess this is who Mujoco does it 
    n = normalize(from_pos - to_pos)
    u = normalize(np.cross(up, n))
    v = np.cross(n, u)
    mat = np.stack([u, v, n], axis=1).flatten()
    quat = np.zeros(4)
    functions.mju_mat2Quat(quat, mat) # this can be replaced with np.quaternion something if we need
    return quat


# OBJECT TYPE THINGS
def Range(min, max):
    """Return 1d numpy array of with min and max"""
    if min <= max:
        return np.array([min, max])
    else:
        print("WARNING: min {} was greater than max {}".format(min, max))
        return np.array([max, min])

def Range3D(x, y, z):
    """Return numpy 1d array of with min and max"""
    return np.array([x,y,z])

def rto3d(r):
    return Range3D(r, r, r)


# UTIL FUNCTIONS FOR RANDOMIZATION
def sample(num_range, mode='standard', as_int=False):
    """Sample a float in the num_range

    mode: logspace means the range 0.1-0.3 won't be sample way less frequently then the range 1-3, because of different scales (i think)
    """
    if mode == 'standard':
        samp = np.random.uniform(num_range[0], num_range[1])
    elif mode == 'logspace':
        num_range = np.log(num_range)
        samp = np.exp(np.random.uniform(num_range[0], num_range[1]))

    if as_int:
        return int(samp)
    else:
        return samp


def sample_geom_type(types=["sphere", "capsule", "ellipsoid", "cylinder", "box"], p=[0.05, 0.05, 0.1, 0.2, 0.6]):
    """Sample a mujoco geom type (range 3-6 capsule-box)"""
    ALL_TYPES = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"]

    shape = np.random.choice(types, p=p)
    return ALL_TYPES.index(shape)

def sample_xyz(range3d, mode='standard'):
    """Sample 3 floats in the 3 num_ranges"""
    x = sample(range3d[0], mode=mode)
    y = sample(range3d[1], mode=mode)
    z = sample(range3d[2], mode=mode)
    return (x, y, z)

def sample_xyz_restrict(range3d, restrict):
    """Like sample_xyz, but if it lands in any of the restricted ranges, then resample"""
    while True:
        x, y, z = sample_xyz(range3d)

def sample_joints(jnt_range, jnt_shape):
    """samples joints"""
    return (jnt_range[:,1] - jnt_range[:,0]) * np.random.sample(jnt_shape) + jnt_range[:,0]

def sample_light_dir():
    """Sample a random direction for a light. I don't quite understand light dirs so
    this might be wrong"""
    # Pretty sure light_dir is just the xyz of a quat with w = 0.
    # I random sample -1 to 1 for xyz, normalize the quat, and then set the tuple (xyz) as the dir
    LIGHT_DIR = Range3D(Range(-1,1), Range(-1,1), Range(-1,1))
    return np.quaternion(0, *sample_xyz(LIGHT_DIR)).normalized().components.tolist()[1:]

def sample_quat(angle3):
    """Sample a quaterion from a range of euler angles in degrees"""
    roll = sample(angle3[0]) * np.pi / 180
    pitch = sample(angle3[1]) * np.pi / 180
    yaw = sample(angle3[2]) * np.pi / 180

    quat = quaternion.from_euler_angles(roll, pitch, yaw)
    return quat.normalized().components

def jitter_angle(quat, angle3):
    """Jitter quat with an angle range"""
    if len(angle3) == 2:
        angle3 = rto3d(angle3)

    sampled = sample_quat(angle3)
    return (np.quaternion(*quat) * np.quaternion(*sampled)).normalized().components

def random_quat():
    """Sample a completely random quaternion"""
    quat_random = np.quaternion(*(np.random.randn(4))).normalized()
    return quat_random.components

def jitter_quat(quat, amount):
    """Jitter a given quaternion by amount"""
    jitter = amount * np.random.randn(4)
    quat_jittered = np.quaternion(*(quat + jitter)).normalized()
    return quat_jittered.components
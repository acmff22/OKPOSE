from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import misc
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils

os.environ["CUDA_VISIBLE_DEVICES"]="0"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean("predict", False, "Running inference if true")
tf.app.flags.DEFINE_string(
    "input",
    "",
    "Input folder containing images")
tf.app.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.app.flags.DEFINE_string(
    "dset",
    "",
    "Path to the directory containing the dataset.")
tf.app.flags.DEFINE_integer("steps", 50000, "Training steps")
tf.app.flags.DEFINE_integer("batch_size", 6, "Size of mini-batch.")
tf.app.flags.DEFINE_string(
    "hparams", "",
    "A comma-separated list of `name=value` hyperparameter values. This flag "
    "is used to override hyperparameter settings either when manually "
    "selecting hyperparameters or when using Vizier.")
tf.app.flags.DEFINE_integer(
    "sync_replicas", -1, 
    "If > 0, use SyncReplicasOptimizer and use this many replicas per sync.")

# Fixed input size 128 x 128.
vw = vh = 128


def create_input_fn(split, batch_size):
  """Returns input_fn for tf.estimator.Estimator.

  Reads tfrecords and construts input_fn for either training or eval. All
  tfrecords not in test.txt or dev.txt will be assigned to training set.

  Args:
    split: A string indicating the split. Can be either 'train' or 'validation'.
    batch_size: The batch size!

  Returns:
    input_fn for tf.estimator.Estimator.

  Raises:
    IOError: If test.txt or dev.txt are not found.
  """

  if (not os.path.exists(os.path.join(FLAGS.dset, "test.txt")) or
      not os.path.exists(os.path.join(FLAGS.dset, "dev.txt"))):
    raise IOError("test.txt or dev.txt not found")

  with open(os.path.join(FLAGS.dset, "test.txt"), "r") as f:
    testset = [x.strip() for x in f.readlines()]

  with open(os.path.join(FLAGS.dset, "dev.txt"), "r") as f:
    validset = [x.strip() for x in f.readlines()]

  files = os.listdir(FLAGS.dset)
  filenames = []
  for f in files:
    sp = os.path.splitext(f)
    if sp[1] != ".tfrecord" or sp[0] in testset:
      continue

    if ((split == "validation" and sp[0] in validset) or
        (split == "train" and sp[0] not in validset)):
      filenames.append(os.path.join(FLAGS.dset, f))

  def input_fn():
    """input_fn for tf.estimator.Estimator."""

    def parser(serialized_example):
      """Parses a single tf.Example into image and label tensors."""
      fs = tf.io.parse_single_example(        #tf.parse_single_example(
          serialized_example,
          features={
              "img0": tf.io.FixedLenFeature([], tf.string),# tf.FixedLenFeature([], tf.string),
              "img1": tf.io.FixedLenFeature([], tf.string),
              "mv0": tf.io.FixedLenFeature([16], tf.float32),
              "mvi0": tf.io.FixedLenFeature([16], tf.float32),
              "mv1": tf.io.FixedLenFeature([16], tf.float32),
              "mvi1": tf.io.FixedLenFeature([16], tf.float32),
              "size0": tf.io.FixedLenFeature([1], tf.float32),
              "size1": tf.io.FixedLenFeature([1], tf.float32),
              "trans_local0": tf.io.FixedLenFeature([3], tf.float32),
              "trans_local1": tf.io.FixedLenFeature([3], tf.float32),
              "center0": tf.io.FixedLenFeature([2], tf.float32),
              "center1": tf.io.FixedLenFeature([2], tf.float32),
              "mask0": tf.io.FixedLenFeature([16384], tf.float32),
              "mask1": tf.io.FixedLenFeature([16384], tf.float32),
          })

      fs["img0"] = tf.math.divide(tf.cast(tf.image.decode_png(fs["img0"], 4),dtype=tf.float32) ,255)
      fs["img1"] = tf.math.divide(tf.cast(tf.image.decode_png(fs["img1"], 4),dtype=tf.float32) ,255)

      fs["img0"].set_shape([vh, vw, 4])
      fs["img1"].set_shape([vh, vw, 4])



      fs["lr0"] = tf.convert_to_tensor([fs["mv0"][0]])
      fs["lr1"] = tf.convert_to_tensor([fs["mv1"][0]])
      
      
      fs["mask0"]=tf.reshape(tf.convert_to_tensor([fs["mask0"]]),[vh, vw])
      fs["mask1"]=tf.reshape(tf.convert_to_tensor([fs["mask1"]]),[vh, vw])
      

      return fs

    np.random.shuffle(filenames)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser, num_parallel_calls=4)
    dataset = dataset.shuffle(400).repeat().batch(batch_size)
    dataset = dataset.prefetch(buffer_size=256)

    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next(), None

    

  return input_fn


class Transformer(object):
  """A utility for projecting 3D points to 2D coordinates and vice versa.

  3D points are represented in 4D-homogeneous world coordinates. The pixel
  coordinates are represented in normalized device coordinates [-1, 1].
  See https://learnopengl.com/Getting-started/Coordinate-Systems.
  """

  def __get_matrix(self, lines):
    return np.array([[float(y) for y in x.strip().split(" ")] for x in lines])

  def __read_projection_matrix(self, filename):
    if not os.path.exists(filename):
      filename = "/home/zhangshaobo/yangli/dataset/keypointnet/ape_simple_linemod/projection.txt"
    with open(filename, "r") as f:
      lines = f.readlines()
    return self.__get_matrix(lines)

  def __init__(self, w, h, dataset_dir):
    self.w = w
    self.h = h
    p = self.__read_projection_matrix(dataset_dir + "projection.txt")

    # transposed of inversed projection matrix.
#    self.pinv_t = tf.constant([[1.0 / p[0, 0], 0, 0,
#                                0], [0, 1.0 / p[1, 1], 0, 0], [0, 0, 1, 0],
#                               [0, 0, 0, 1]])
    self.pinv_t = tf.constant([[0.5590385,  0. , 0.,0.],
                               [0.,0.41843161,  0. , 0.],
                               [-0.00919112, -0.00357234, 1., 0.  ],
                               [0., 0.,  0.,1.  ]])
    self.f = p[0, 0]
    self.pt=p
    
  def project_back(self, xyzw):
    """Projects homogeneous 3D coordinates to normalized device coordinates."""

    z = xyzw[:, :, 2:3] + 1e-8
#    z= tf.Print(z, [z,tf.shape(z)], "z: ",summarize=1000) 
    return tf.concat([self.f * xyzw[:, :, :2] / z, z], axis=2)

  def project(self, xyzw):
    """Projects homogeneous 3D coordinates to normalized device coordinates."""
    #从cv映射回blender摄像机
    
    z = xyzw[:, :, 2:3] + 1e-8
    xy=xyzw[:, :, :2] / z
    xyz=tf.concat([xy, z,tf.ones_like(z)], axis=2)
    p_transpose=tf.transpose(tf.cast(self.pt,dtype=tf.float32))
    xyz_homo=tf.matmul(xyz,p_transpose)

    return xyz_homo

  def unproject(self, xyz):
    """Unprojects normalized device coordinates with depth to 3D coordinates."""

    z = xyz[:, :, 2:]
    xy = xyz * z

    def batch_matmul(a, b):
      return tf.reshape(
          tf.matmul(tf.reshape(a, [-1, a.shape[2].value]), b),
          [-1, a.shape[1].value, a.shape[2].value])
    
    return batch_matmul(
        tf.concat([xy[:, :, :2], z, tf.ones_like(z)], axis=2), self.pinv_t)

def meshgrid(h):
  """Returns a meshgrid ranging from [-1, 1] in x, y axes."""

  r = np.arange(0.5, h, 1) / (h / 2) - 1
#  ranx, rany = tf.meshgrid(r, -r)
  ranx, rany = tf.meshgrid(r, r)
  return tf.cast(ranx,dtype=tf.float32), tf.cast(rany,dtype=tf.float32) #tf.to_float(ranx), tf.to_float(rany)


def estimate_rotation(xyz0, xyz1, pconf, noise):
  """Estimates the rotation between two sets of keypoints.

  The rotation is estimated by first subtracting mean from each set of keypoints
  and computing SVD of the covariance matrix.

  Args:
    xyz0: [batch, num_kp, 3] The first set of keypoints.
    xyz1: [batch, num_kp, 3] The second set of keypoints.
    pconf: [batch, num_kp] The weights used to compute the rotation estimate.
    noise: A number indicating the noise added to the keypoints.

  Returns:
    [batch, 3, 3] A batch of transposed 3 x 3 rotation matrices.
  """

  
  pconf2 = tf.expand_dims(pconf, 2)
  
  cen0 = tf.reduce_sum(xyz0 * pconf2, 1, keepdims=True)
  cen1 = tf.reduce_sum(xyz1 * pconf2, 1, keepdims=True)

  x = xyz0 - cen0
  y = xyz1 - cen1
  

  cov = tf.matmul(tf.matmul(x, tf.linalg.diag(pconf), transpose_a=True), y)
  _, u, v = tf.linalg.svd(cov, full_matrices=True)
  
  d = tf.linalg.det(tf.matmul(v, u, transpose_b=True))
  ud = tf.concat(
      [u[:, :, :-1], u[:, :, -1:] * tf.expand_dims(tf.expand_dims(d, 1), 1)],
      axis=2)
  R=tf.matmul(ud, v, transpose_b=True)
  t= tf.matmul(-R,cen0,transpose_a=True,transpose_b=True)+tf.transpose((cen1),[0, 2, 1])
  return R,t



def relative_pose_loss(xyz0, xyz1, rot, T, pconf, noise):
  """Computes the relative pose loss (chordal, angular).

  Args:
    xyz0: [batch, num_kp, 3] The first set of keypoints.
    xyz1: [batch, num_kp, 3] The second set of keypoints.
    rot: [batch, 4, 4] The ground-truth rotation matrices.
    pconf: [batch, num_kp] The weights used to compute the rotation estimate.
    noise: A number indicating the noise added to the keypoints.

  Returns:
    A tuple (chordal loss, angular loss).
  """
  
  r_transposed,t = estimate_rotation(xyz0, xyz1, pconf, noise)
  t=tf.reshape(t, [-1, 3])
  rotation = rot[:, :3, :3]
  trans=rot[:, 3, :3]

  Translations_relative=tf.squeeze(tf.expand_dims(T[1],1)-tf.matmul(tf.expand_dims(T[0],1),rotation))

  
  frob_sqr = tf.reduce_sum(tf.square(r_transposed - rotation), axis=[1, 2])
  frob = tf.sqrt(frob_sqr)

  t_sqr = tf.reduce_sum(tf.square(Translations_relative-trans), axis=[1])

  mean_R=2.0 * tf.reduce_mean(tf.asin(tf.minimum(1.0, frob / (2 * math.sqrt(2)))))
  mean_R=tf.Print(mean_R, [mean_R], "mean_R: ",summarize=1000)
  mean_t=tf.reduce_mean(t_sqr)
  mean_t=tf.Print(mean_t, [mean_t], "mean_t: ",summarize=1000)

  return mean_R, mean_t


def separation_loss(xyz, delta):
  """Computes the separation loss.

  Args:
    xyz: [batch, num_kp, 3] Input keypoints.
    delta: A separation threshold. Incur 0 cost if the distance >= delta.

  Returns:
    The seperation loss.
  """

  num_kp = tf.shape(xyz)[1]
  t1 = tf.tile(xyz, [1, num_kp, 1])



  t2 = tf.reshape(tf.tile(xyz, [1, 1, num_kp]), tf.shape(t1))
  diffsq = tf.square(t1 - t2)

  # -> [batch, num_kp ^ 2],
  lensqr = tf.reduce_sum(diffsq, axis=2)

  v=tf.constant([0,1,1,1,1,1,1,1,1,1,
                 1,0,1,1,1,1,1,1,1,1,
                 1,1,0,1,1,1,1,1,1,1,
                 1,1,1,0,1,1,1,1,1,1,
                 1,1,1,1,0,1,1,1,1,1,                   
                 1,1,1,1,1,0,1,1,1,1,
                 1,1,1,1,1,1,0,1,1,1,
                 1,1,1,1,1,1,1,0,1,1,
                 1,1,1,1,1,1,1,1,0,1,
                 1,1,1,1,1,1,1,1,1,0
                                     ])

  return (tf.reduce_sum(tf.maximum(tf.multiply((-lensqr + delta),tf.cast(v,dtype=tf.float32)), 0.0)) / tf.cast(
    FLAGS.batch_size,dtype=tf.float32)) # num_kp *      * (num_kp-1)


def consistency_loss(uv0, uv1, pconf):
  """Computes multi-view consistency loss between two sets of keypoints.

  Args:
    uv0: [batch, num_kp, 2] The first set of keypoint 2D coordinates.
    uv1: [batch, num_kp, 2] The second set of keypoint 2D coordinates.
    pconf: [batch, num_kp] The weights used to compute the rotation estimate.

  Returns:
    The consistency loss.
  """


  # [batch, num_kp, 2]
  wd = tf.square(uv0 - uv1) * tf.expand_dims(pconf, 2)
  wd = tf.reduce_sum(wd, axis=[1, 2])
  return tf.reduce_mean(wd)

def transformation_loss(xyz0,xyz1, pconf):
  """Computes multi-view consistency loss between two sets of 3D Skeypoints.

  Args:
    xyz0: [batch, num_kp, 2] The first set of keypoint 3D coordinates.
    xyz1: [batch, num_kp, 2] The second set of keypoint 3D coordinates.
    pconf: [batch, num_kp] The weights used to compute the rotation estimate.

  Returns:
    The transformation loss.
  """


  # [batch, num_kp, 2]
  wd = tf.square(xyz0 - xyz1) * tf.expand_dims(pconf, 2)
  wd = tf.reduce_sum(wd, axis=[1, 2])
  return tf.reduce_mean(wd)


def variance_loss(probmap, ranx, rany, uv):
  """Computes the variance loss as part of Sillhouette consistency.

  Args:
    probmap: [batch, num_kp, h, w] The distribution map of keypoint locations.
    ranx: X-axis meshgrid.
    rany: Y-axis meshgrid.
    uv: [batch, num_kp, 2] Keypoint locations (in NDC).

  Returns:
    The variance loss.
  """

  ran = tf.stack([ranx, rany], axis=2)

  sh = tf.shape(ran)
  # [batch, num_kp, vh, vw, 2]
  ran = tf.reshape(ran, [1, 1, sh[0], sh[1], 2])

  sh = tf.shape(uv)
  uv = tf.reshape(uv, [sh[0], sh[1], 1, 1, 2])

  diff = tf.reduce_sum(tf.square(uv - ran), axis=4)
  diff *= probmap

  return tf.reduce_mean(tf.reduce_sum(diff, axis=[2, 3]))


def dilated_cnn(images, num_filters, is_training):
  """Constructs a base dilated convolutional network.

  Args:
    images: [batch, h, w, 3] Input RGB images.
    num_filters: The number of filters for all layers.
    is_training: True if this function is called during training.

  Returns:
    Output of this dilated CNN.
  """

  net = images

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      normalizer_fn=slim.batch_norm,
      activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
      normalizer_params={"is_training": is_training}):
    for i, r in enumerate([1, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1]):
      net = slim.conv2d(net, num_filters, [3, 3], rate=r, scope="dconv%d" % i)

  return net

def translation_network(net, is_training):
  """Constructs a network that infers the relative translation of image pairs.

  Args:
    images: [batch, h, w, 3] Input RGB images.
    is_training: True if this function is called during training.

  Returns:
    Output of the translation network.
  """
  with  tf.compat.v1.variable_scope("TranslationNetwork"): 
    with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      normalizer_fn=slim.batch_norm,
      activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
      normalizer_params={"is_training": is_training}):

        net=slim.conv2d(net, 128, [3, 3], stride=2 ,scope="conv_downsample%d" % 1)

        net=slim.conv2d(net, 256, [3, 3], stride=2 ,scope="conv_downsample%d" % 2)

        net=slim.conv2d(net, 512, [3, 3], stride=2 ,scope="conv_downsample%d" % 3)

        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')    

        
        for i in range(3):
            net=slim.conv2d(net, 256, [3, 3], stride=1 ,scope="conv_%d" % i)
 
        net = slim.flatten(net)
        net=slim.fully_connected(net, 4096, normalizer_fn=None, scope='fc1')
        net=slim.fully_connected(net, 4096, normalizer_fn=None,scope='fc2')
        net=slim.fully_connected(net, 3, activation_fn=None, normalizer_fn=None,scope='fc3')

        
        
   

  return net


def keypoint_network(rgba,
                     mask_m,
                     num_filters,
                     num_kp,
                     is_training,
                     lr_gt=None,
                     anneal=1):
  """Constructs our main keypoint network that predicts 3D keypoints.

  Args:
    rgba: [batch, h, w, 4] Input RGB images with alpha channel.
    num_filters: The number of filters for all layers.
    num_kp: The number of keypoints.
    is_training: True if this function is called during training.
    lr_gt: The groundtruth orientation flag used at the beginning of training.
        Then we linearly anneal in the prediction.
    anneal: A number between [0, 1] where 1 means using the ground-truth
        orientation and 0 means using our estimate.

  Returns:
    uv: [batch, num_kp, 2] 2D locations of keypoints.
    z: [batch, num_kp] The depth of keypoints.
    orient: [batch, 2, 2] Two 2D coordinates that correspond to [1, 0, 0] and
        [-1, 0, 0] in object space.
    sill: The Sillhouette loss.
    variance: The variance loss.
    prob_viz: A visualization of all predicted keypoints.
    prob_vizs: A list of visualizations of each keypoint.

  """
  
  images = rgba[:, :, :, :3]
  
  # [batch, 1]
  

#  mask = rgba[:, :, :, 3]
  mask = mask_m
  mask = tf.cast(tf.greater(mask, tf.zeros_like(mask)), dtype=tf.float32)

  net = dilated_cnn(images, num_filters, is_training)

  Translation=translation_network(net,is_training)
  # The probability distribution map.
  prob = slim.conv2d(
      net, num_kp, [3, 3], rate=1, scope="conv_xy", activation_fn=None)

  # We added the  fixed camera distance as a bias.
  z = 1 + slim.conv2d(
      net, num_kp, [3, 3], rate=1, scope="conv_z", activation_fn=None)
  

  prob = tf.transpose(prob, [0, 3, 1, 2])
  z = tf.transpose(z, [0, 3, 1, 2])

  prob = tf.reshape(prob, [-1, num_kp, vh * vw])
  prob = tf.nn.softmax(prob, name="softmax")

  ranx, rany = meshgrid(vh)
  prob = tf.reshape(prob, [-1, num_kp, vh, vw])

  # These are for visualizing the distribution maps.
  prob_viz = tf.expand_dims(tf.reduce_sum(prob, 1), 3)
  prob_vizs = [tf.expand_dims(prob[:, i, :, :], 3) for i in range(num_kp)]

  sx = tf.reduce_sum(prob * ranx, axis=[2, 3])
  sy = tf.reduce_sum(prob * rany, axis=[2, 3])  # -> batch x num_kp

  # [batch, num_kp]
  sill = tf.reduce_sum(prob * tf.expand_dims(mask, 1), axis=[2, 3])
  sill = tf.reduce_mean(-tf.math.log(sill + 1e-12)) #tf.reduce_mean(-tf.log(sill + 1e-12))

  z = tf.reduce_sum(prob * z, axis=[2, 3])
  uv = tf.reshape(tf.stack([sx, sy], -1), [-1, num_kp, 2])

  variance = variance_loss(prob, ranx, rany, uv)

  return uv, z, sill, variance, prob_viz, prob_vizs, Translation


def keypoint_network_test(rgba,
                     mask_m,
                     num_filters,
                     num_kp,
                     is_training,
                     lr_gt=None,
                     anneal=1):
  """Constructs our main keypoint network that predicts 3D keypoints.

  Args:
    rgba: [batch, h, w, 4] Input RGB images with alpha channel.
    num_filters: The number of filters for all layers.
    num_kp: The number of keypoints.
    is_training: True if this function is called during training.
    lr_gt: The groundtruth orientation flag used at the beginning of training.
        Then we linearly anneal in the prediction.
    anneal: A number between [0, 1] where 1 means using the ground-truth
        orientation and 0 means using our estimate.

  Returns:
    uv: [batch, num_kp, 2] 2D locations of keypoints.
    z: [batch, num_kp] The depth of keypoints.
    orient: [batch, 2, 2] Two 2D coordinates that correspond to [1, 0, 0] and
        [-1, 0, 0] in object space.
    sill: The Sillhouette loss.
    variance: The variance loss.
    prob_viz: A visualization of all predicted keypoints.
    prob_vizs: A list of visualizations of each keypoint.

  """
  
  images = rgba[:, :, :, :3]
  
  # [batch, 1]

  mask = mask_m
  mask = tf.cast(tf.greater(mask, tf.zeros_like(mask)), dtype=tf.float32)
  
  is_training=False
  net = dilated_cnn(images, num_filters, is_training)
  Translation=translation_network(net,is_training)

  # The probability distribution map.
  prob = slim.conv2d(
      net, num_kp, [3, 3], rate=1, scope="conv_xy", activation_fn=None)

  # We added the  fixed camera distance as a bias.
  z = 1 + slim.conv2d(
      net, num_kp, [3, 3], rate=1, scope="conv_z", activation_fn=None)
  

  prob = tf.transpose(prob, [0, 3, 1, 2])
  z = tf.transpose(z, [0, 3, 1, 2])

  prob = tf.reshape(prob, [-1, num_kp, vh * vw])
  prob = tf.nn.softmax(prob, name="softmax")

  ranx, rany = meshgrid(vh)
  prob = tf.reshape(prob, [-1, num_kp, vh, vw])

  # These are for visualizing the distribution maps.
  prob_viz = tf.expand_dims(tf.reduce_sum(prob, 1), 3)
  prob_vizs = [tf.expand_dims(prob[:, i, :, :], 3) for i in range(num_kp)]

  sx = tf.reduce_sum(prob * ranx, axis=[2, 3])
  sy = tf.reduce_sum(prob * rany, axis=[2, 3])  # -> batch x num_kp

  # [batch, num_kp]
  sill = tf.reduce_sum(prob * tf.expand_dims(mask, 1), axis=[2, 3])
  sill = tf.reduce_mean(-tf.math.log(sill + 1e-12)) #tf.reduce_mean(-tf.log(sill + 1e-12))

  z = tf.reduce_sum(prob * z, axis=[2, 3])
  uv = tf.reshape(tf.stack([sx, sy], -1), [-1, num_kp, 2])

  variance = variance_loss(prob, ranx, rany, uv)

  return uv, z, sill, variance, prob_viz, prob_vizs, Translation


def model_fn(features, labels, mode, hparams):
  """Returns model_fn for tf.estimator.Estimator."""

  del labels
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  t = Transformer(vw, vh, FLAGS.dset)

  def func1(x):
    return tf.transpose(tf.reshape(features[x], [-1, 4, 4]), [0, 2, 1])

  mv = [func1("mv%d" % i) for i in range(2)]
  mvi = [func1("mvi%d" % i) for i in range(2)]

  uvz = [None] * 2
  uvz_proj = [None] * 2  # uvz coordinates projected on to the other view.
  xyz = [None] * 2
  xyz_trans = [None] * 2
  Translations = [None] * 2
  T = [None] * 2
  viz = [None] * 2
  vizs = [None] * 2

  loss_sill = 0
  loss_variance = 0
  loss_con = 0
  loss_sep = 0
  loss_lr = 0
  loss_tran = 0
  translationa=0
  
  for i in range(2):
    with tf.compat.v1.variable_scope("KeypointNetwork", reuse=i > 0): #tf.variable_scope("KeypointNetwork", reuse=i > 0):
      # anneal: 1 = using ground-truth, 0 = using our estimate orientation.
      anneal = tf.cast(hparams.lr_anneal_end - tf.compat.v1.train.get_global_step(),dtype=tf.float32)#tf.to_float(hparams.lr_anneal_end - tf.train.get_global_step())
      anneal = tf.clip_by_value(
          anneal / (hparams.lr_anneal_end - hparams.lr_anneal_start), 0.0, 1.0)

      uv, z, sill, variance, viz[i], vizs[i], Translation = keypoint_network(
          features["img%d" % i],
          features["mask%d" % i],
          hparams.num_filters,
          hparams.num_kp,
          is_training,
          lr_gt=features["lr%d" % i],
          anneal=anneal)

      loss_lr +=tf.constant(0.)
      loss_variance += variance
      loss_sill += sill
      
      uv = tf.reshape(uv, [-1, hparams.num_kp, 2])
      z = tf.reshape(z, [-1, hparams.num_kp, 1])
      


      size=features["size%d" % i]
      radio1=tf.constant(128.0)/size
      
      u_ori=((tf.reshape(uv[:,:,0],[-1, hparams.num_kp])*tf.constant(64.0)+tf.constant(64.0))/radio1+tf.expand_dims(features["center%d" % i][:,0],1)-(size/tf.constant(2.0))-tf.constant(320.0))/tf.constant(320.0)
      v_ori=((tf.reshape(uv[:,:,1],[-1, hparams.num_kp])*tf.constant(64.0)+tf.constant(64.0))/radio1+tf.expand_dims(features["center%d" % i][:,1],1)-(size/tf.constant(2.0))-tf.constant(240.0))/tf.constant(240.0)
      u_ori=tf.reshape(u_ori,[-1, hparams.num_kp,1])
      v_ori=tf.reshape(v_ori,[-1, hparams.num_kp,1])
      

      uv_ori=tf.concat([u_ori, v_ori], axis=2)
      uv=uv_ori
      
      T[i]=Translation
      
      size=tf.squeeze(features["size%d" % i])
      radio=tf.constant(128.0)/size
      
      Translation_z=Translation[:,2]*radio
      Translation_x=((Translation[:,0]*size+features["center%d" % i][:,0]-tf.constant(320.0))/tf.constant(320.0)-tf.constant(0.016440937499999995))*Translation_z/tf.constant(1.7887856249999998)
      Translation_y=((Translation[:,1]*size+features["center%d" % i][:,1]-tf.constant(240.0))/tf.constant(240.0)-tf.constant(0.008537458333333348))*Translation_z/tf.constant(2.3898767916666666)
      

      
      Translations[i]=tf.concat([tf.expand_dims(Translation_x,1), tf.expand_dims(Translation_y,1),tf.expand_dims(Translation_z,1)], axis=1)

      
      # [batch, num_kp, 3]
      uvz[i] = tf.concat([uv, z], axis=2)

      
      world_coords = tf.matmul(t.unproject(uvz[i]), mvi[i])

    
      # [batch, num_kp, 3]
      uvz_proj[i] = t.project(tf.matmul(world_coords, mv[1 - i]))

      
      
      xyz[i]=t.unproject(uvz[i])
      xyz_trans[i]=tf.matmul(t.unproject(uvz[i]), tf.matmul(mvi[i],mv[i-1]))
      
  pconf = tf.ones(
      [tf.shape(uv)[0], tf.shape(uv)[1]], dtype=tf.float32) / hparams.num_kp

  for i in range(2):
    loss_con += consistency_loss(uvz_proj[i][:, :, :2], uvz[1 - i][:, :, :2],
                                 pconf)
    loss_sep += separation_loss(
        t.unproject(uvz[i])[:, :, :3], hparams.sep_delta)
    
    loss_tran += transformation_loss(xyz_trans[i][:, :, :3], xyz[1 - i][:, :, :3],
                                 pconf)
    

    translationa+=tf.reduce_mean(tf.reduce_sum(tf.square(Translations[i]-mv[i][:,3,:3]), axis=[1]))

  angular, translation = relative_pose_loss(
      t.unproject(uvz[0])[:, :, :3],
      t.unproject(uvz[1])[:, :, :3], tf.matmul(mvi[0], mv[1]), Translations,pconf,
      hparams.noise)
  T0=tf.reduce_sum(tf.square(T[0]), axis=[1])
  T1=tf.reduce_sum(tf.square(T[1]), axis=[1])

  T0_reg=tf.reduce_mean(T0)
  T1_reg=tf.reduce_mean(T1)

  T0_reg=tf.Print(T0_reg, [T0_reg], "T0_reg: ",summarize=1000)
  T1_reg=tf.Print(T1_reg, [T1_reg], "T1_reg: ",summarize=1000)

  loss = (
      hparams.loss_pose * angular +
      hparams.loss_trans * translation +
      hparams.loss_transformation * loss_tran +
      hparams.loss_con * loss_con +
      hparams.loss_sep * loss_sep +
      hparams.loss_sill * loss_sill +
      hparams.loss_lr * loss_lr +
      hparams.loss_variance * loss_variance
  )

  def touint8(img):
    return tf.cast(img * 255.0, tf.uint8)

  with tf.compat.v1.variable_scope("output"): #tf.variable_scope("output"):
    tf.compat.v1.summary.image("0_img0", touint8(features["img0"][:, :, :, :3])) #tf.summary.image("0_img0", touint8(features["img0"][:, :, :, :3]))
    tf.compat.v1.summary.image("1_combined", viz[0]) #tf.summary.image("1_combined", viz[0])
    for i in range(hparams.num_kp):
      tf.compat.v1.summary.image("2_f%02d" % i, vizs[0][i]) #tf.summary.image("2_f%02d" % i, vizs[0][i])

  with tf.compat.v1.variable_scope("stats"): #tf.variable_scope("stats"):
    tf.compat.v1.summary.scalar("anneal", anneal)
    tf.compat.v1.summary.scalar("closs", loss_con)
    tf.compat.v1.summary.scalar("seploss", loss_sep)
    tf.compat.v1.summary.scalar("angular", angular)
    tf.compat.v1.summary.scalar("translation", translation)
    tf.compat.v1.summary.scalar("transformation", loss_tran)
    tf.compat.v1.summary.scalar("lrloss", loss_lr)
    tf.compat.v1.summary.scalar("sill", loss_sill)
    tf.compat.v1.summary.scalar("vloss", loss_variance)
  return {
      "loss": loss,
      "angular":angular,
      "translation":translation,
      "loss_tran":loss_tran,
      "loss_con":loss_con,
      "loss_sep":loss_sep, 
      "loss_sill":loss_sill,
      "loss_lr":loss_lr,
      "loss_variance":loss_variance,
      "predictions": {
          "img0": features["img0"],
          "img1": features["img1"],
          "uvz0": uvz[0],
          "uvz1": uvz[1],
          "Translation0": Translations[0],
          "Translation1": Translations[1],
      },
      "eval_metric_ops": {
          "closs": tf.compat.v1.metrics.mean(loss_con), #tf.metrics.mean(loss_con),
          "angular_loss": tf.compat.v1.metrics.mean(angular),
          "translation_loss": tf.compat.v1.metrics.mean(translation),
      }
  }


def predict(input_folder, hparams):
  """Predicts keypoints on all images in input_folder."""
  
  cols = plt.cm.get_cmap("rainbow")(
      np.linspace(0, 1.0, hparams.num_kp))[:, :4]

  img = tf.compat.v1.placeholder(tf.float32, shape=(1, 128, 128, 4))#tf.placeholder(tf.float32, shape=(1, 128, 128, 4))
  mask_m = tf.compat.v1.placeholder(tf.float32, shape=(1, 128, 128))#tf.placeholder(tf.float32, shape=(1, 128, 128, 4))
  
  with  tf.compat.v1.variable_scope("KeypointNetwork"): #tf.variable_scope("KeypointNetwork"):
    ret = keypoint_network_test(
        img,mask_m,hparams.num_filters, hparams.num_kp, False)

  uv = tf.reshape(ret[0], [-1, hparams.num_kp, 2])
  z = tf.reshape(ret[1], [-1, hparams.num_kp, 1])
  uvz = tf.concat([uv, z], axis=2)
  Translations=tf.reshape(ret[6], [-1, 1, 3])
  
  sess = tf.compat.v1.Session() #tf.Session()
  saver = tf.compat.v1.train.Saver() #tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  print("loading model: ", ckpt.model_checkpoint_path)
  saver.restore(sess, ckpt.model_checkpoint_path)

  files = [x for x in os.listdir(input_folder)
           if x[-3:] in ["jpg", "png"]]

  output_folder = os.path.join(input_folder, "output")
  if not os.path.exists(output_folder):
    os.mkdir(output_folder)

  for f in files:
    orig = misc.imread(os.path.join(input_folder, f)).astype(float) / 255
    mask = misc.imread(os.path.join(input_folder,'mask', f)).astype(float).reshape(1,128,128)

    
    if orig.shape[2] == 3:
      orig = np.concatenate((orig, np.ones_like(orig[:, :, :1])), axis=2)  

    uv_ret,t_ret = sess.run([uvz,Translations],feed_dict={img: np.expand_dims(orig, 0),mask_m:mask})
    

    np.savetxt(input_folder+'/uv_ret'+'/%06d.txt' % int(f[:6]), uv_ret.reshape(10,3))

     
    utils_6.draw_ndc_points(orig, uv_ret.reshape(hparams.num_kp, 3), cols)
    misc.imsave(os.path.join(output_folder, f), orig)
    
    center=np.loadtxt(os.path.join(input_folder, "center",f[-10:-4])+".txt")
    size=np.loadtxt(os.path.join(input_folder, "size",f[-10:-4])+".txt")
    
    uv_ret_ori_x=((uv_ret[:,:,0]*64+64)/(128/size)+(center[0]-(size/2))-320)/320
    uv_ret_ori_y=((uv_ret[:,:,1]*64+64)/(128/size)+(center[1]-(size/2))-240)/240
    

    uv_ret[:,:,0]=uv_ret_ori_x
    uv_ret[:,:,1]=uv_ret_ori_y
    
    t_ret=t_ret.squeeze()

    
    t_ret[2]= t_ret[2]*(128/size)
    t_ret[0]=((t_ret[0]*size+center[0]-320.0)/320.0-0.016440937499999995)*t_ret[2]/1.7887856249999998
    t_ret[1]=((t_ret[1]*size+center[1]-240.0)/240.0-0.008537458333333348)*t_ret[2]/2.3898767916666666

    np.savetxt(input_folder+'/t_results'+'/%06d.txt' % int(f[:6]), t_ret)
    
    
    k=np.loadtxt("/home/zhangshaobo/yangli/dataset/keypointnet/ape_linemod/projection.txt")
    z=uv_ret[0,:,2:3]
    uv_ret[0,:,:2]=uv_ret[0,:,:2]*z
    object_coord_camera=np.dot(uv_ret,np.linalg.inv(k[:3,:3].T)).reshape(hparams.num_kp, 3)
    object_coord_camera=np.insert(object_coord_camera,3,1,axis=1)
    print(os.path.join(input_folder, f))  

    np.savetxt(input_folder+'/results'+'/%06d.txt' % int(f[:6]), object_coord_camera)
    pose=np.loadtxt(os.path.join(input_folder, "pose",f[-10:-4])+".txt")
    a=np.array([[0,0,0,1]])
    pose=np.concatenate((pose,a),axis=0)
    object_coord=np.dot(object_coord_camera,np.linalg.inv(pose.T))



def _default_hparams():
  """Returns default or overridden user-specified hyperparameters."""

  hparams = tf.contrib.training.HParams(
      num_filters=64,  # Number of filters.
      num_kp=10,  # Numer of keypoints. 

      loss_pose=1.0,  # Pose Loss. 
      loss_trans=1.0,
      loss_transformation=1.0,
      loss_con=1.0,  # Multiview consistency Loss.
      loss_sep=1.0,  # Seperation Loss. 
      loss_sill=1.0,  # Sillhouette Loss.
      loss_variance=0.5,  # Variance Loss (part of Sillhouette loss).

      sep_delta=0.0005,  # Seperation threshold. 
      noise=0.1,  # Noise added during estimating rotation. 

      learning_rate=1.0e-4, #default=1.0e-3
      lr_anneal_start=30000,  # When to anneal in the orientation prediction.
      lr_anneal_end=60000,  # When to use the prediction completely.
  )
  if FLAGS.hparams:
    hparams = hparams.parse(FLAGS.hparams)
  return hparams


def main(argv):
  del argv

  hparams = _default_hparams()
  if FLAGS.predict:
    predict(FLAGS.input, hparams)
  else:
    utils_6.train_and_eval(
        model_dir=FLAGS.model_dir,
        model_fn=model_fn,
        input_fn=create_input_fn,
        hparams=hparams,
        steps=FLAGS.steps,
        batch_size=FLAGS.batch_size,
        save_checkpoints_secs=600,
        eval_throttle_secs=1800,
        eval_steps=5,
        sync_replicas=FLAGS.sync_replicas,
    )


if __name__ == "__main__":
  sys.excepthook = utils_6.colored_hook(
      os.path.dirname(os.path.realpath(__file__)))
  tf.compat.v1.app.run() #tf.app.run()

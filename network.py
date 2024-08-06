# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:59:43 2023

@author: Administrator
"""
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
from tensorflow.contrib.slim.nets import resnet_v2


class Model(object):
    """xxx definition."""
    
    def __init__(self, is_training):
        """Constructor.
        
        Args:
            is_training: A boolean indicating whether the training version of
                computation graph should be constructed.
            num_classes: Number of classes.
        """
        self.vh=256
        self.vw=256
    
    def meshgrid(self,h):
      """Returns a meshgrid ranging from [-1, 1] in x, y axes."""
    
      r = np.arange(0.5, h, 1) / (h / 2) - 1
    #  ranx, rany = tf.meshgrid(r, -r)
      ranx, rany = tf.meshgrid(r, r)
      return tf.cast(ranx,dtype=tf.float32), tf.cast(rany,dtype=tf.float32) #tf.to_float(ranx), tf.to_float(rany)
    
    def dilated_cnn(self, images, num_filters, is_training):
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
    
    def variance_loss(self, probmap, ranx, rany, uv):
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
    
    def translation_network(self, net, is_training):
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
          normalizer_params={"is_training": is_training}):
    #        net=tf.Print(net, [tf.shape(net)],"net",summarize=1000) 
    
    #        net=slim.conv2d(net, 128, [3, 3], stride=2 ,scope="conv_downsample%d" % 1)
    ##        net=tf.Print(net, [tf.shape(net)],"net",summarize=1000) 
    #        net=slim.conv2d(net, 256, [3, 3], stride=2 ,scope="conv_downsample%d" % 2)
    ##        net=tf.Print(net, [tf.shape(net)],"net",summarize=1000) 
    #        net=slim.conv2d(net, 512, [3, 3], stride=2 ,scope="conv_downsample%d" % 3)
    ##        net=tf.Print(net, [tf.shape(net)],"net",summarize=1000) 
    #        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')    
    #        net=tf.Print(net, [tf.shape(net)],"net",summarize=1000) 
            
            for i in range(3):
                net=slim.conv2d(net, 256, [3, 3], stride=1 ,scope="conv_%d" % i)
    #            net=tf.Print(net, [tf.shape(net)],"dconv_%d" % i,summarize=1000) 
            net = slim.flatten(net)
    #        net=tf.Print(net, [tf.shape(net),net[0,:10]],"net_flatten",summarize=1000) 
            net=slim.fully_connected(net, 4096, scope='fc1')
    #        net=tf.Print(net, [tf.shape(net)], "fc1: ",summarize=1000)
            net=slim.fully_connected(net, 4096, scope='fc2')
    #        net=tf.Print(net, [tf.shape(net)], "fc2: ",summarize=1000)
            net=slim.fully_connected(net, 3, activation_fn=None, normalizer_fn=None,scope='fc3')
    #        net=tf.Print(net, [tf.shape(net)], "fc3: ",summarize=1000)
            
            
      return net
        
    def keypoint_network(self, 
                     rgba,
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
    
      net = self.dilated_cnndilated_cnn(images, num_filters, is_training)
      with slim.arg_scope(resnet_v2.resnet_arg_scope()):
          net1, end_points = resnet_v2.resnet_v2_50(images, num_classes=None, is_training=True, global_pool=False)
    
      Translation=self.translation_network(net1,is_training)
    #  Translation=tf.Print(Translation, [Translation,tf.shape(Translation)], "Translation: ",summarize=1000) 
      net1=tf.Print(net1, [tf.shape(net1)], "net1: ",summarize=1000)
      # The probability distribution map.
      prob = slim.conv2d(
          net, num_kp, [3, 3], rate=1, scope="conv_xy", activation_fn=None)
    #  prob=tf.Print(prob, [tf.shape(prob)], "prob: ",summarize=1000)
      # We added the  fixed camera distance as a bias.
      z = 1 + slim.conv2d(
          net, num_kp, [3, 3], rate=1, scope="conv_z", activation_fn=None)
      
    
      prob = tf.transpose(prob, [0, 3, 1, 2])
      z = tf.transpose(z, [0, 3, 1, 2])
    
      prob = tf.reshape(prob, [-1, num_kp, self.vh * self.vw])
      prob = tf.nn.softmax(prob, name="softmax")
    
      ranx, rany = self.meshgrid(self.vh)
      prob = tf.reshape(prob, [-1, num_kp, self.vh, self.vw])
    
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
    
      variance = self.variance_loss(prob, ranx, rany, uv)
    
      return uv, z, sill, variance, prob_viz, prob_vizs, Translation

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Utility functions for KeypointNet.

These are helper / tensorflow related functions. The actual implementation and
algorithm is in main.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import os
import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import traceback

import matplotlib.pyplot as plt
from scipy import misc
import test_for_training
from model import Model3D


class TrainingHook (tf.estimator.SessionRunHook): #(tf.train.SessionRunHook):
  """A utility for displaying training information such as the loss, percent
  completed, estimated finish date and time."""

  def __init__(self, steps,params):
    self.steps = steps

    self.last_time = time.time()
    self.last_est = self.last_time

    self.eta_interval = int(math.ceil(0.1 * self.steps))
    self.current_interval = 0
    self.params=params

  def before_run(self, run_context):
    graph = tf.compat.v1.get_default_graph()#tf.get_default_graph()
    return tf.estimator.SessionRunArgs(     #tf.train.SessionRunArgs(
        {"loss": graph.get_collection("total_loss")[0],
        "angular_loss": graph.get_collection("angular_loss")[0],
        "loss_translation": graph.get_collection("loss_translation")[0],
        "loss_tran": graph.get_collection("loss_tran")[0],
        "loss_con": graph.get_collection("loss_con")[0],
        "loss_sep": graph.get_collection("loss_sep")[0],
        "loss_sill": graph.get_collection("loss_sill")[0],
        "loss_lr": graph.get_collection("loss_lr")[0],
        "loss_variance": graph.get_collection("loss_variance")[0],})

  def after_run(self, run_context, run_values):
    step = run_context.session.run(tf.compat.v1.train.get_global_step()) #tf.train.get_global_step())
    now = time.time()

    if self.current_interval < self.eta_interval:
      self.duration = now - self.last_est
      self.current_interval += 1
    if step % self.eta_interval == 0:
      self.duration = now - self.last_est
      self.last_est = now

    eta_time = float(self.steps - step) / self.current_interval * \
        self.duration
    m, s = divmod(eta_time, 60)
    h, m = divmod(m, 60)
    eta = "%d:%02d:%02d" % (h, m, s)
    
    input_folder_train='/home/zhangshaobo/yangli/dataset/keypointnet/test_lm_benchvise_CDPN_128_train/'
    input_folder_test='/home/zhangshaobo/yangli/dataset/keypointnet/test_lm_benchvise_CDPN_128_test/'
    
    print("%.2f%% (%d/%d): loss: %.3e angular_loss: %.3e loss_translation: %.3e loss_tran: %.3e loss_con:%.3e loss_sep:%.3e loss_sill:%.3e loss_lr:%.3e loss_variance:%.3e t %.3f  @ %s (%s)" % (
            step * 100.0 / self.steps,
            step,
            self.steps,
            run_values.results["loss"],
            run_values.results["angular_loss"],
            run_values.results["loss_translation"],
            run_values.results["loss_tran"],
            run_values.results["loss_con"],
            run_values.results["loss_sep"],
            run_values.results["loss_sill"],
            run_values.results["loss_lr"],
            run_values.results["loss_variance"],
            now - self.last_time,
            time.strftime("%a %d %H:%M:%S", time.localtime(time.time() + eta_time)),
            eta))
    
#    if(step % 2267 == 0):
#        
#        with open("/home/zhangshaobo/yangli/models/benchvise/model_benchvise_1/log.txt", "a") as file:
#            file.write("%.2f%% (%d/%d): loss: %.3e angular_loss: %.3e loss_translation: %.3e loss_tran: %.3e loss_con:%.3e loss_sep:%.3e loss_sill:%.3e loss_lr:%.3e loss_variance:%.3e t %.3f  @ %s (%s) \n" % (
#            step * 100.0 / self.steps,
#            step,
#            self.steps,
#            run_values.results["loss"],
#            run_values.results["angular_loss"],
#            run_values.results["loss_translation"],
#            run_values.results["loss_tran"],
#            run_values.results["loss_con"],
#            run_values.results["loss_sep"],
#            run_values.results["loss_sill"],
#            run_values.results["loss_lr"],
#            run_values.results["loss_variance"],
#            now - self.last_time,
#            time.strftime("%a %d %H:%M:%S", time.localtime(time.time() + eta_time)),
#            eta))
#            
#        predict_during_training(input_folder_train,input_folder_train,self.params)
#        predict_during_training(input_folder_test,input_folder_train,self.params)   
            
    self.last_time = now


def standard_model_fn(
    func, steps, run_config=None, sync_replicas=0, optimizer_fn=None):
  """Creates model_fn for tf.Estimator.

  Args:
    func: A model_fn with prototype model_fn(features, labels, mode, hparams).
    steps: Training steps.
    run_config: tf.estimatorRunConfig (usually passed in from TF_CONFIG).
    sync_replicas: The number of replicas used to compute gradient for
        synchronous training.
    optimizer_fn: The type of the optimizer. Default to Adam.

  Returns:
    model_fn for tf.estimator.Estimator.
  """

  def fn(features, labels, mode, params):
    """Returns model_fn for tf.estimator.Estimator."""

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    ret = func(features, labels, mode, params)

    tf.compat.v1.add_to_collection("total_loss", ret["loss"]) #tf.add_to_collection("total_loss", ret["loss"])
    tf.compat.v1.add_to_collection("angular_loss", ret["angular"])
    tf.compat.v1.add_to_collection("loss_translation", ret["translation"])
    tf.compat.v1.add_to_collection("loss_tran", ret["loss_tran"])
    tf.compat.v1.add_to_collection("loss_con", ret["loss_con"])
    tf.compat.v1.add_to_collection("loss_sep", ret["loss_sep"])
    tf.compat.v1.add_to_collection("loss_sill", ret["loss_sill"])
    tf.compat.v1.add_to_collection("loss_lr", ret["loss_lr"])
    tf.compat.v1.add_to_collection("loss_variance", ret["loss_variance"])
    train_op = None

    training_hooks = []
    if is_training:
      training_hooks.append(TrainingHook(steps,params))
#      optimizer_fn=tf.compat.v1.train.RMSPropOptimizer(params.learning_rate)
      if optimizer_fn is None:
        optimizer = tf.compat.v1.train.AdamOptimizer(params.learning_rate) #tf.train.AdamOptimizer(params.learning_rate)

      else:
        optimizer = optimizer_fn

      if run_config is not None and run_config.num_worker_replicas > 1:
        print(ooo)
        sr = sync_replicas
        if sr <= 0:
          sr = run_config.num_worker_replicas
        print(ooo)
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer,
            replicas_to_aggregate=sr,
            total_num_replicas=run_config.num_worker_replicas)

        training_hooks.append(
            optimizer.make_session_run_hook(
                run_config.is_chief, num_tokens=run_config.num_worker_replicas))

      optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5)
#      optimizer=tf.Print(optimizer, [optimizer], "optimizer: ",summarize=1000)
      print('optimizer',optimizer)
      train_op = slim.learning.create_train_op(ret["loss"], optimizer)

    if "eval_metric_ops" not in ret:
      ret["eval_metric_ops"] = {}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=ret["predictions"],
        loss=ret["loss"],
        train_op=train_op,
        eval_metric_ops=ret["eval_metric_ops"],
        training_hooks=training_hooks)
  return fn


def train_and_eval(
    model_dir,
    steps,
    batch_size,
    model_fn,
    input_fn,
    hparams,
    keep_checkpoint_every_n_hours=0.5,
    save_checkpoints_secs=180,
    save_summary_steps=50,
    eval_steps=20,
    eval_start_delay_secs=10,
    eval_throttle_secs=300,
    sync_replicas=0):
  """Trains and evaluates our model. Supports local and distributed training.

  Args:
    model_dir: The output directory for trained parameters, checkpoints, etc.
    steps: Training steps.
    batch_size: Batch size.
    model_fn: A func with prototype model_fn(features, labels, mode, hparams).
    input_fn: A input function for the tf.estimator.Estimator.
    hparams: tf.HParams containing a set of hyperparameters.
    keep_checkpoint_every_n_hours: Number of hours between each checkpoint
        to be saved.
    save_checkpoints_secs: Save checkpoints every this many seconds.
    save_summary_steps: Save summaries every this many steps.
    eval_steps: Number of steps to evaluate model.
    eval_start_delay_secs: Start evaluating after waiting for this many seconds.
    eval_throttle_secs: Do not re-evaluate unless the last evaluation was
        started at least this many seconds ago
    sync_replicas: Number of synchronous replicas for distributed training.

  Returns:
    None
  """
  strategy = tf.distribute.MirroredStrategy()
  

  run_config = tf.estimator.RunConfig(
      keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
      save_checkpoints_secs=save_checkpoints_secs,
      save_summary_steps=save_summary_steps,
      #train_distribute=strategy
      )
  
  #print("hparams",hparams)
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
  print("run_config",run_config.num_worker_replicas)
  estimator = tf.estimator.Estimator(
      model_dir=model_dir,
      model_fn=standard_model_fn(
          model_fn,
          steps,
          run_config,
          sync_replicas=sync_replicas),
      params=hparams, config=run_config)

  train_spec = tf.estimator.TrainSpec(
      input_fn=input_fn(split="train", batch_size=batch_size),
      max_steps=steps)

  eval_spec = tf.estimator.EvalSpec(
      input_fn=input_fn(split="validation", batch_size=batch_size),
      steps=eval_steps,
      start_delay_secs=eval_start_delay_secs,
      throttle_secs=eval_throttle_secs)
  
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def draw_circle(rgb, u, v, col, r):
  """Draws a simple anti-aliasing circle in-place.

  Args:
    rgb: Input image to be modified.
    u: Horizontal coordinate.
    v: Vertical coordinate.
    col: Color.
    r: Radius.
  """

  ir = int(math.ceil(r))
  for i in range(-ir-1, ir+2):
    for j in range(-ir-1, ir+2):
      nu = int(round(u + i))
      nv = int(round(v + j))
      if nu < 0 or nu >= rgb.shape[1] or nv < 0 or nv >= rgb.shape[0]:
        continue

      du = abs(nu - u)
      dv = abs(nv - v)

      # need sqrt to keep scale
      t = math.sqrt(du * du + dv * dv) - math.sqrt(r * r)
      if t < 0:
        rgb[nv, nu, :] = col
      else:
        t = 1 - t
        if t > 0:
          # t = t ** 0.3
          rgb[nv, nu, :] = col * t + rgb[nv, nu, :] * (1-t)


def draw_ndc_points(rgb, xy, cols):
  """Draws keypoints onto an input image.

  Args:
    rgb: Input image to be modified.
    xy: [n x 2] matrix of 2D locations.
    cols: A list of colors for the keypoints.
  """

  vh, vw = rgb.shape[0], rgb.shape[1]

  for j in range(len(cols)):
    x, y = xy[j, :2]
    x = (min(max(x, -1), 1) * vw / 2 + vw / 2) - 0.5
#    y = vh - 0.5 - (min(max(y, -1), 1) * vh / 2 + vh / 2)
    y = (min(max(y, -1), 1) * vh / 2 + vh / 2) - 0.5

    x = int(round(x))
    y = int(round(y))

    if x < 0 or y < 0 or x >= vw or y >= vh:
      continue

    rad = 1.5
    rad *= rgb.shape[0] / 128.0
    draw_circle(rgb, x, y, np.array([0.0, 0.0, 0.0, 1.0]), rad * 1.5)
    draw_circle(rgb, x, y, cols[j], rad)


def colored_hook(home_dir):
  """Colorizes python's error message.

  Args:
    home_dir: directory where code resides (to highlight your own files).
  Returns:
    The traceback hook.
  """

  def hook(type_, value, tb):
    def colorize(text, color, own=0):
      """Returns colorized text."""
      endcolor = "\x1b[0m"
      codes = {
          "green": "\x1b[0;32m",
          "green_own": "\x1b[1;32;40m",
          "red": "\x1b[0;31m",
          "red_own": "\x1b[1;31m",
          "yellow": "\x1b[0;33m",
          "yellow_own": "\x1b[1;33m",
          "black": "\x1b[0;90m",
          "black_own": "\x1b[1;90m",
          "cyan": "\033[1;36m",
      }
      return codes[color + ("_own" if own else "")] + text + endcolor

    for filename, line_num, func, text in traceback.extract_tb(tb):
      basename = os.path.basename(filename)
      own = (home_dir in filename) or ("/" not in filename)

      print(colorize("\"" + basename + '"', "green", own) + " in " + func)
      print("%s:  %s" % (
          colorize("%5d" % line_num, "red", own),
          colorize(text, "yellow", own)))
      print("  %s" % colorize(filename, "black", own))

    print(colorize("%s: %s" % (type_.__name__, value), "cyan"))
  return hook

def get_camera_intrinsic(u0, v0, fx, fy):
    return np.array([[fx, 0.0, u0], [0.0, fy, v0], [0.0, 0.0, 1.0]])
def calcAngularDistance(gt_rot, pr_rot):
    rotDiff = np.dot(gt_rot, np.transpose(pr_rot))
    trace = np.trace(rotDiff)
    if trace > 3 :
        trace = 3.0
    if trace < 0 :
        trace = 0.0
    
    frob_sqr=np.sum(np.square(gt_rot - pr_rot))  
    frob = np.sqrt(frob_sqr)
    angluar=2.0 * np.mean(np.arcsin(np.minimum(1.0, frob / (2 * math.sqrt(2)))))
    return np.rad2deg(np.arccos((trace-1.0)/2.0)), angluar
   
def calc_pts_diameter(pts):
    diameter = -1
    for pt_id in range(pts.shape[0]):
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter
def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d
def compute_transformation(points_3D, transformation):
    return transformation.dot(points_3D)

def rigid_transform_3D(A, B):
    assert len(A) == len(B)
    N = A.shape[0];
    mu_A = np.mean(A, axis=0)
    mu_B = np.mean(B, axis=0)

    AA = A - np.tile(mu_A, (N, 1))
    BB = B - np.tile(mu_B, (N, 1))
    

    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T

    if np.linalg.det(R) < 0:
#        print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * mu_A.T + mu_B.T

    return R, t    

def read_data(gtdir):
    dirs = os.listdir(gtdir)
    dirs.sort(key=lambda x : int(x.split('.')[0]))
    data = []
    for i in dirs :
        mat = np.loadtxt(gtdir + i)
        # mat = torch.from_numpy(mat)
        R = mat[:3,:3]
#        R =np.array([[0., 1., 0.],
#                               [ 0., 0., -1.],
#                               [ -1., 0., 0.]])
        t = mat[:3,3].reshape(3,1)
        data.append({
            'R':np.array(R) ,
            't':np.array(t),
        })
    return data

def test_add(poses,input_folder):
    gt_data = read_data(input_folder+'/pose/')
    pr_data = poses
    
    meshname     = "/home/zhangshaobo/yangli/dataset/Hinterstoisser/models/obj_02.ply"
    fx           = 572.4114
    fy           = 573.57043
    u0           = 325.2611
    v0           = 242.04899
    
    testing_samples = 0.0
    testing_error_trans = 0.0
    testing_error_angle = 0.0
    testing_error_pixel = 0.0
    errs_2d             = []
    errs_3d             = []
    errs_trans          = []
    errs_angle          = []
    
    model_obj=Model3D()
    model_obj.load(meshname,scale=0.001)
    vertices  = np.c_[np.array(model_obj.vertices), np.ones((len(model_obj.vertices), 1))].transpose()
    diam = model_obj.diameter
    
    intrinsic_calibration = get_camera_intrinsic(u0, v0, fx, fy)
    print("   Testing  ...")
    for i in range(len(pr_data)):
        R_gt,t_gt, R_pr, t_pr = gt_data[i].get('R'), gt_data[i].get('t'), pr_data[i][:3,:3], pr_data[i][:3,3].reshape(3,1)
#        print("index",i)
        # Compute translation error
        trans_dist = np.sqrt(np.sum(np.square(t_gt - t_pr)))
        trans_dist_sep = np.sqrt(np.square(t_gt - t_pr))
        errs_trans.append(trans_dist)
        # Compute angle error
#        print('i',i)
        angle_dist,angluar = calcAngularDistance(R_gt, R_pr)
#        print('angle_dist',angle_dist)
#        print('angluar',angluar)
        errs_angle.append(angle_dist)
        # Compute pixel error
        Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
#        print('Rt_gt',Rt_gt)
        Rt_pr = np.concatenate((R_pr, t_pr.reshape(3,1)), axis=1)
#        print('Rt_pr',Rt_pr)
        proj_2d_gt = compute_projection(vertices, Rt_gt, intrinsic_calibration)
        proj_2d_pred = compute_projection(vertices, Rt_pr, intrinsic_calibration)
        norm = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
        pixel_dist = np.mean(norm)
        errs_2d.append(pixel_dist)
        # Compute 3D distances
        transform_3d_gt = compute_transformation(vertices, Rt_gt)
        transform_3d_pred = compute_transformation(vertices, Rt_pr)
        norm3d = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
        vertex_dist = np.mean(norm3d)

#        if(vertex_dist<diam * 0.1):
#            print("index",i)
##            print('diam * 0.1',diam * 0.1)
##            print('vertex_dist',vertex_dist)
##            print('angle_dist',angle_dist)
#            print('angluar',angluar)

        errs_3d.append(vertex_dist)
        # Sum errors
        testing_error_trans += trans_dist
        testing_error_angle += angle_dist
        testing_error_pixel += pixel_dist
        testing_samples += 1
    
    px_threshold = 5 # 5 pixel threshold for 2D reprojection error is standard in recent sota 6D object pose estimation works
    eps          = 1e-5
    acc          = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
    acc3d10      = len(np.where(np.array(errs_3d) <= diam * 0.1)[0]) * 100. / (len(errs_3d)+eps)
    print(len(np.where(np.array(errs_3d) <= diam * 0.1)[0]))
    # print(errs_angle)
    acc5cm5deg   = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
    nts = float(testing_samples)
    # Print test statistics
    with open("/home/zhangshaobo/yangli/models/benchvise/model_benchvise_1/log.txt", "a") as file:
#            file.write('\n')
            file.write('   Results of ' + input_folder +'\n')
            file.write('       Acc using {} px 2D Projection = {:.2f}% \n'.format(px_threshold, acc))
            file.write('       Acc using 10% threshold - {} vx 3D Transformation = {:.2f}% \n'.format(diam * 0.1, acc3d10))
            file.write('       Acc using 5 cm 5 degree metric = {:.2f}% \n'.format(acc5cm5deg))
            file.write('       Translation error: %f m, angle error: %f degree, pixel error: % f pix, vertex error: % f \n' % (testing_error_trans/nts, testing_error_angle/nts, testing_error_pixel/nts,np.mean(errs_3d)))
            file.write('\n')
            
    print('Results of ')
    print('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
    print('   Acc using 10% threshold - {} vx 3D Transformation = {:.2f}%'.format(diam * 0.1, acc3d10))
    print('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
    print('   Translation error: %f m, angle error: %f degree, pixel error: % f pix, vertex error: % f ' % (testing_error_trans/nts, testing_error_angle/nts, testing_error_pixel/nts,np.mean(errs_3d)) )

def valid(poses,input_folder):
    gt_data = read_data(input_folder+'/pose/')
    pr_data = poses
    errs_3d=[]
    
    # Parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    meshname = "/home/zhangshaobo/yangli/dataset/Hinterstoisser/models/obj_02.ply"
    model_obj=Model3D()
    model_obj.load(meshname,scale=0.001)
    vertices  = np.c_[np.array(model_obj.vertices), np.ones((len(model_obj.vertices), 1))].transpose()
    diam = model_obj.diameter
#    print("   Testing  ...")
#    print('len',len(pr_data))
    for i in range(len(pr_data)):
        R_gt,t_gt, R_pr, t_pr = gt_data[i].get('R'), gt_data[i].get('t'), pr_data[i][:3,:3], pr_data[i][:3,3].reshape(3,1)
        
#        print('R_pr',R_pr)
#        print('t_pr',t_pr)

        # Compute pixel error
        Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
        Rt_pr = np.concatenate((R_pr, t_pr), axis=1)

        # Compute 3D distances
        transform_3d_gt = compute_transformation(vertices, Rt_gt)
        transform_3d_pred = compute_transformation(vertices, Rt_pr)
        norm3d = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
        vertex_dist = np.mean(norm3d)
        
        eps          = 1e-5
        errs_3d.append(vertex_dist)
        acc3d10 = len(np.where(np.array(errs_3d) <= diam * 0.1)[0]) * 100. / (len(errs_3d)+eps)

    return acc3d10    
    
def calPose(keypoints,Ts,input_folder):
    ref=0
    pose_ref=np.loadtxt(input_folder+'/pose/%06d.txt' % ref)   
#    pose_ref[:3,:3]=np.array([[0., 1., 0.],
#                               [ 0., 0., -1.],
#                               [ -1., 0., 0.]])
    pose12=np.ones_like(pose_ref)
    poses=[]
    for i in range(len(keypoints)):
        points_3D_1=keypoints[ref]
        points_3D_2=keypoints[i]
        pose_1=np.loadtxt(input_folder+'/pose/%06d.txt' % ref)
        pose_2=np.loadtxt(input_folder+'/pose/%06d.txt' % i)
    
        ret_R, ret_t=rigid_transform_3D(np.mat(points_3D_1), np.mat(points_3D_2))
        pose12[:3,:3]=ret_R
        pose12[:3,3]=ret_t.T
#        print('pose12',pose12)
        pose=np.dot(pose12,pose_ref)
        poses.append(pose)
        
    test_add(poses,input_folder)    

def calPose_loop(keypoints,Ts,input_folder,ref_folder):
    correct_nums=[]
    poses_best=[]
    for j in range(183): 
        ref=j
        pose_ref=np.loadtxt(ref_folder+'/pose/%06d.txt' % ref)   
        a=np.array([[0,0,0,1]])
        pose_ref=np.concatenate((pose_ref,a),axis=0)
    
    #    pose_ref[:3,:3]=np.array([[0., 1., 0.],
    #                               [ 0., 0., -1.],
    #                               [ -1., 0., 0.]])
        pose12=np.ones_like(pose_ref)
        poses=[]
        for i in range(len(keypoints)):
#            points_3D_1=keypoints[ref]
            points_3D_1=np.loadtxt(ref_folder+'results_ref'+'/%06d.txt' % ref)
            points_3D_2=keypoints[i]
            pose_1=np.loadtxt(input_folder+'/pose/%06d.txt' % ref)
            pose_2=np.loadtxt(input_folder+'/pose/%06d.txt' % i)
        
            ret_R, ret_t=rigid_transform_3D(np.mat(points_3D_1), np.mat(points_3D_2))
            pose12[:3,:3]=ret_R
            pose=np.dot(pose12,pose_ref)
            
#            t_1=Ts[ref]
            t_1=np.loadtxt(ref_folder+'t_ref'+'/%06d.txt' % ref)
            t_2=Ts[i]
            t_relative=t_2-np.dot(t_1,ret_R[:3,:3].T)
            t_2=t_relative+np.dot(pose_ref[:3,3],ret_R[:3,:3].T)
#            print('t_2',np.squeeze(np.asarray(t_2)),type(t_2),type(np.asarray(t_2)))
#            pose[:3,3]= np.expand_dims(t_2, axis=0).T
            pose[:3,3]= np.squeeze(np.asarray(t_2)).T
            poses.append(pose)
            
        correct_num=valid(poses,input_folder)
        correct_nums.append(correct_num)
        
    for k in range(len(keypoints)):
        ref=correct_nums.index(max(correct_nums))
        points_3D_1=keypoints[ref]
        points_3D_2=keypoints[k]
        ret_R, ret_t=rigid_transform_3D(np.mat(points_3D_1), np.mat(points_3D_2))
        pose12[:3,:3]=ret_R
        pose=np.dot(pose12,pose_ref)
            
        t_1=Ts[ref]
        t_2=Ts[i]
        t_relative=t_2-np.dot(t_1,ret_R[:3,:3].T)
        t_2=t_relative+np.dot(pose_ref[:3,3],ret_R[:3,:3].T)
#        pose[:3,3]= np.array(t_2.T)
#        print(np.squeeze(np.asarray(t_2)),np.asarray(t_2))
        pose[:3,3]= np.squeeze(np.asarray(t_2)).T
#        pose[:3,3]= np.expand_dims(t_2, axis=0).T
        
        poses_best.append(pose)
    test_add(poses,input_folder)        

def predict_during_training(input_folder, ref_folder,hparams):
  """Predicts keypoints on all images in input_folder."""
  with tf.Graph().as_default():
      cols = plt.cm.get_cmap("rainbow")(
          np.linspace(0, 1.0, hparams.num_kp))[:, :4]
    
      img = tf.compat.v1.placeholder(tf.float32, shape=(1, 128, 128, 4))#tf.placeholder(tf.float32, shape=(1, 128, 128, 4))
      mask_m = tf.compat.v1.placeholder(tf.float32, shape=(1, 128, 128))
      with  tf.compat.v1.variable_scope("KeypointNetwork"): #tf.variable_scope("KeypointNetwork"):
        ret = test_for_training.keypoint_network_dilated_test(
            img, mask_m,hparams.num_filters, hparams.num_kp, False)
    
      uv = tf.reshape(ret[0], [-1, hparams.num_kp, 2])
      z = tf.reshape(ret[1], [-1, hparams.num_kp, 1])
      uvz = tf.concat([uv, z], axis=2)
      Translations=tf.reshape(ret[6], [-1, 1, 3])
      
      sess = tf.compat.v1.Session() #tf.Session()
      saver = tf.compat.v1.train.Saver() #tf.train.Saver()
      ckpt = tf.train.get_checkpoint_state('/home/zhangshaobo/yangli/models/benchvise/model_benchvise_1')
      print("loading model: ", ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
    
      files = [x for x in os.listdir(input_folder)
               if x[-3:] in ["jpg", "png"]]
    
      output_folder = os.path.join(input_folder, "output")
      if not os.path.exists(output_folder):
        os.mkdir(output_folder)
      keypoints=[]
      Ts=[]
      files.sort()
      
      for f in files:
        orig = misc.imread(os.path.join(input_folder, f)).astype(float) / 255
        mask = misc.imread(os.path.join(input_folder,'mask', f)).astype(float).reshape(1,128,128)
        if orig.shape[2] == 3:
          orig = np.concatenate((orig, np.ones_like(orig[:, :, :1])), axis=2)
    
        uv_ret,t_ret = sess.run([uvz,Translations],feed_dict={img: np.expand_dims(orig, 0),mask_m:mask})
        
        center=np.loadtxt(os.path.join(input_folder, "center",f[-10:-4])+".txt")
        size=np.loadtxt(os.path.join(input_folder, "size",f[-10:-4])+".txt")
    
        uv_ret_ori_x=((uv_ret[:,:,0]*64+64)/(128/size)+(center[0]-(size/2))-320)/320
        uv_ret_ori_y=((uv_ret[:,:,1]*64+64)/(128/size)+(center[1]-(size/2))-240)/240
        

        uv_ret[:,:,0]=uv_ret_ori_x
        uv_ret[:,:,1]=uv_ret_ori_y

         
        k=np.loadtxt("/home/zhangshaobo/yangli/dataset/keypointnet/ape_linemod/projection.txt")
        z=uv_ret[0,:,2:3]
        uv_ret[0,:,:2]=uv_ret[0,:,:2]*z
        p_position4D_T=np.dot(uv_ret,np.linalg.inv(k[:3,:3].T))
#        print('p_position4D_T',p_position4D_T)
        keypoints.append(p_position4D_T)
        
        t_ret=t_ret.squeeze()
        t_ret[2]= t_ret[2]*(128/size)
        t_ret[0]=((t_ret[0]*size+center[0]-320.0)/320.0-0.016440937499999995)*t_ret[2]/1.7887856249999998
        t_ret[1]=((t_ret[1]*size+center[1]-240.0)/240.0-0.008537458333333348)*t_ret[2]/2.3898767916666666
        if(input_folder==ref_folder):
            np.savetxt(ref_folder+'t_ref'+'/%06d.txt' % int(f[:6]), t_ret)
            np.savetxt(ref_folder+'/results_ref'+'/%06d.txt' % int(f[:6]), np.squeeze(p_position4D_T))
        Ts.append(t_ret)
        
  calPose_loop(keypoints,Ts,input_folder,ref_folder)

#    print(os.path.join(input_folder, f))  
#    print("object_coord_camera",object_coord_camera)
#    np.savetxt(input_folder+'/results'+'/%06d.txt' % int(f[:6]), object_coord_camera[0])
#    pose=np.loadtxt(os.path.join(input_folder, "pose",f[-10:-4])+".txt")
#    object_coord=np.dot(object_coord_camera,np.linalg.inv(pose.T))
#    print("object_coord",object_coord)


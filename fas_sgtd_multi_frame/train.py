import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as pd
import tensorflow as tf
import FLAGS
from generate_data_train import input_fn_maker
from generate_network import generate_network as model_fn

flags = FLAGS.flags

import argparse

parser = argparse.ArgumentParser()

# parser.add_argument('--model', type=str, default='./model_save/', help='path of model')
parser.add_argument('--epoch', type=int, default=1000, help='epoch of training')
parser.add_argument('--batch', type=int, default=2, help='batch size of training')
parser.add_argument('--depth-path', type=str, required=True, help='path of depth image folder')
parser.add_argument('--image-path', type=str, required=True, help='path of rgb image folder')

# parse arguments
args = parser.parse_args()

# flags.path.model = args.model
flags.path.data = [args.image_path, args.depth_path]
flags.paras.epoch = args.epoch
flags.paras.batch_size_train = args.batch

# log info setting
tf.logging.set_verbosity(tf.logging.INFO)
# data fn

# train_data_list=[flags.path.train_file]
train_data_list=[flags.path.data]
train_input_fn = input_fn_maker(train_data_list, shuffle=True, 
                                batch_size = flags.paras.batch_size_train,
                                epoch=flags.paras.epoch)
# test_data_list=[flags.path.test_file, flags.path.dev_file]
# test_input_fn = input_fn_maker_test(test_data_list, shuffle=False, 
#                                 batch_size = flags.paras.batch_size_test,
#                                 epoch=1)

# model fn
model_fn_this=model_fn

# GPU config
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True 
# # create estimator
this_config=tf.estimator.RunConfig(
    save_summary_steps=flags.display.summary_iter,
    save_checkpoints_steps=flags.display.display_iter,
    keep_checkpoint_max=102400,
    log_step_count_steps=flags.display.log_iter,
    session_config=tf.ConfigProto(allow_soft_placement=True,
                                gpu_options=tf.GPUOptions(allow_growth=True))
)
mnist_classifier = tf.estimator.Estimator(
    model_fn=model_fn_this, config=this_config, model_dir=flags.path.model)

''' only run train set '''
mnist_classifier.train(input_fn=train_input_fn)



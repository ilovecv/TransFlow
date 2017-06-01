import os
import sys
import numpy as np
import scipy.misc
from utils import pp, visualize, to_json

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from transflow import TransFlow
from transflow_eval import TransflowExperiment

np.random.seed(23)
tf.set_random_seed(23)
# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) #flush

flags = tf.app.flags

flags.DEFINE_string("dbname", "KITTI_FLOW", "Name of the dataset [KITTI_RAW|KITTI_FLOW|Sintel]")
flags.DEFINE_boolean("augment", False, "Apply Real-Time data augmentation [False]")
flags.DEFINE_boolean("use_bilat", False, "Apply bilateral filtering to the flow [False]")

flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate of for adam [1e-4]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("epochs", 100, "Max-training Epochs to train [100]")
flags.DEFINE_integer("num_batches", 1000, "Batches per epoch")
flags.DEFINE_string("colorspace", "yuv", "the colorspace to be used")

flags.DEFINE_string("check_save", "checkpoint/aaa", "Directory name to save the checkpoints [checkpoint/transflow]")
flags.DEFINE_string("check_load", None, "Directory name to load the checkpoints [None]")
flags.DEFINE_string('log_dir', 'logs/transflow', 'Directory with the log data.')

flags.DEFINE_string("transform", "SmoothJointBase", "Global transform type [Affine2D|Projective2D]")

flags.DEFINE_string("mode", "eval", "Training or testing [train|test]")
flags.DEFINE_string("competitor", "deepflow", "competitor for the evaluation")
flags.DEFINE_string("where", "modena", "switch for dataset paths [modena|...]")

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    
    s = tf.Session()
    with tf.Session() as sess:
        runtime = TransFlow(sess, flags.FLAGS)
        if flags.FLAGS.mode == 'train':
            print('Running the training')
            runtime.train()
        elif flags.FLAGS.mode == 'test':
            print('Running the inference')
            runtime.inference()
        elif(flags.FLAGS.mode == 'eval'):
            print('Running the evaluation')
            exp = TransflowExperiment(runtime)
            exp.run_experiment()
        elif flags.FLAGS.mode == 'train_eval':
            print('Running the train/evaluation')
            exp = TransflowExperiment(runtime)
            exp.run_experiment(mode='train_eval')
        elif flags.FLAGS.mode == 'reconstruction':
            print('Running the reconstruction experiment')
            exp = TransflowExperiment(runtime)
            exp.run_experiment_reconstruction(flags.FLAGS.competitor)

if __name__ == '__main__':
    tf.app.run()

TransFlow: Unsupervised Motion Flow by Joint Geometric and Pixel-level Estimation

Main content:
transflow_app.py: entry point of the application
transflow.py: core of the method, contains the model, train and inference functions
transflow_eval.py: evaluation class used to perform experiments on various datasets
data_io.py: class handling the interaction with the different datasets
spatial_transformer.py: spatial transformer class
tf_writer.py: class handling the creation of tfrecords

Usage examples:

Training on the KITTI FLOW dataset
python transflow_app.py --learning_rate 1e-4 --mode train --batch_size 8 --check_save your/checkpoint/dir --log_dir your/log/dir --dbname KITTI_FLOW

Testing on the KITTI FLOW dataset
python transflow_app.py --mode eval --batch_size 1 --dbname KITTI_FLOW --check_load your/checkpoint/dir

Developed and tested on Ubuntu 16.04, python 2.7.13, tensorflow 1.0.0

import copy
import os
import sys
import signal
import time
import datetime

from skimage.transform import resize

from os.path import join
from glob import glob

import h5py
import numpy as np
from six.moves import xrange
import skimage.io as io
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops

from ops import *
from utils import *

from spatial_transformer import transformer, meshgrid
from data_augmentation import DataAugmentation
from flowToColor import flowToColor
from data_io import data_io
from yuv import *
from bilateral_utils import *
import bilateral_op_and_grad

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self,signum, frame):    	self.kill_now = True


class TransFlow(data_io):

    def __init__(self, sess, config):
        """
            Transformational flow estimator architecture
        """
        self.sess = sess
        self.config = config
        self.summary_list = []
        self.theta_dim = 6 if self.config.transform == 'Affine2D' else 9
        data_io.__init__(self, config)

        # Bilateral filter grid
        self.npwcolor = 10.
        self.npwspatial = 10.
        self.stdv = 1.
        self.x_feat = tf.expand_dims(tf.expand_dims(tf.matmul(tf.ones(shape=tf.stack([self.h, 1])),tf.transpose(tf.expand_dims(tf.linspace(0., self.w-1, self.w),1),[1,0])) / self.npwspatial, axis=0), axis=3)
        self.y_feat = tf.expand_dims(tf.expand_dims(tf.matmul(tf.expand_dims(tf.linspace(0., self.h-1, self.h),1),tf.ones(shape=tf.stack([1, self.w]))) / self.npwspatial, axis=0), axis=3)
        
        self.verbose = False

        if self.config.batch_size > 1:
            self.x_feat = tf.concat([self.x_feat for i in range(self.config.batch_size)], axis=0)
            self.y_feat = tf.concat([self.y_feat for i in range(self.config.batch_size)], axis=0)
        self.build_model() 

    def vbn(self, tensor, name, half=None):
        return tensor

    def charbonnier_distance(self, x,y,e=0.1,gate=None):
        if(gate is None):
            return tf.sqrt(tf.reduce_mean(tf.square(x-y)+e))
        else:
            gated_diff = (x-y) * gate
            return tf.sqrt(tf.reduce_mean(tf.square(gated_diff)+e))

    def smooth_penalty(self, xx, e=0.1, gating=None):
        bn = xx.get_shape()[0].value
        hn = xx.get_shape()[1].value
        wn = xx.get_shape()[2].value
        cn = xx.get_shape()[3].value
        # print bn,hn,wn,cn
        aa=tf.slice(xx, [0,0,0,0], [bn, hn-1, wn,cn])
        bb=tf.slice(xx, [0,1,0,0], [bn, hn-1, wn,cn])
        cc=tf.slice(xx, [0,0,0,0], [bn, hn, wn-1,cn])
        dd=tf.slice(xx, [0,0,1,0], [bn, hn, wn-1,cn])
        if(gating is None):
            return self.charbonnier_distance(aa,bb,e,gating) + self.charbonnier_distance(cc,dd,e,gating)
        else:
            gg_aabb=tf.slice(gating, [0,0,0,0], [bn, hn-1, wn,1])
            gg_ccdd=tf.slice(gating, [0,0,0,0], [bn, hn, wn-1,1])
            return self.charbonnier_distance(aa,bb,e,gg_aabb) + self.charbonnier_distance(cc,dd,e,gg_ccdd)

    def add_summary_variable(self, var, what_type, desc):
        if(what_type is 'scalar'):
            self.summary_list.append( tf.summary.scalar(desc, var) )
        if(what_type is 'image'):
            self.summary_list.append( tf.summary.image(desc, var, max_outputs=1) )
        if(what_type is 'histogram'):
            self.summary_list.append( tf.summary.histogram(desc, var) )

    def get_batch(self):
        return data_io.get_batch(self)

    def theta_generator_small(self, images):
        net = slim.conv2d(images, 32, [3,3], activation_fn=lrelu, stride=2, padding='SAME', scope='g_conv_11')
        net = slim.conv2d(net, 32, [3,3], activation_fn=lrelu, stride=2, padding='SAME', scope='g_conv_12')
        net = lrelu(self.vbn(net, name='g_vbn_1'))

        net = slim.conv2d(net, 64, [3,3], activation_fn=lrelu, stride=2, padding='SAME', scope='g_conv_21')
        net = slim.conv2d(net, 64, [3,3], activation_fn=lrelu, stride=2, padding='SAME', scope='g_conv_22')
        net = lrelu(self.vbn(net, name='g_vbn_2'))

        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, stride=2, padding='SAME', scope='g_conv_31')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, stride=2, padding='SAME', scope='g_conv_32')
        net = lrelu(self.vbn(net, name='g_vbn_3'))

        net = slim.fully_connected(slim.flatten(net), 128, activation_fn=lrelu, scope='g_fc_1')
        net = slim.fully_connected(slim.flatten(net), 1024, activation_fn=lrelu, scope='g_fc_2')
        net = slim.fully_connected(slim.flatten(net), self.theta_dim, activation_fn=None, scope='g_fc_3')

        return net

    def flow_generator_flownet(self, images, base_flow=None):
        grid  = meshgrid(self.h, self.w, flatten=False)
        base = tf.tile(tf.expand_dims(grid,0), [self.config.batch_size,1,1,1])
        if base_flow is not None:
            inp = tf.concat(axis=3, values=[images, base, base_flow])
        else:
            inp = tf.concat(axis=3, values=[images, base])

        net = slim.conv2d(inp, 96, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_conv1_1')
        net = slim.conv2d(net, 96, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_conv1_2')
        net = slim.conv2d(net, 96, [3,3], activation_fn=lrelu, stride=2, padding='SAME', scope='g_fl_conv1_3')
        net1 = lrelu(self.vbn(net, name='g_fl_vbn1'))

        net = slim.conv2d(net1, 96, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_conv2_1')
        net = slim.conv2d(net, 96, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_conv2_2')
        net = slim.conv2d(net, 96, [3,3], activation_fn=lrelu, stride=2, padding='SAME', scope='g_fl_conv2_3')
        net2 = lrelu(self.vbn(net, name='g_fl_vbn2'))

        net = slim.conv2d(net2, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_conv3_1')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_conv3_2')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, stride=2, padding='SAME', scope='g_fl_conv3_3')
        net3 = lrelu(self.vbn(net, name='g_fl_vbn3'))

        net = slim.conv2d(net3, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_conv4_1')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_conv4_2')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, stride=2, padding='SAME', scope='g_fl_conv4_3')
        net4 = lrelu(self.vbn(net, name='g_fl_vbn4'))

        net = slim.conv2d(net4, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_conv5_1')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_conv5_2')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, stride=2, padding='SAME', scope='g_fl_conv5_3')
        net5 = lrelu(self.vbn(net, name='g_fl_vbn5'))
        #deconv
        net = slim.conv2d_transpose(net5, 128, [4, 4], activation_fn=lrelu, stride=2, padding='SAME', scope='g_fl_dcnv5')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_cnv5_1')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_cnv5_2')
        net = lrelu(self.vbn(net, name='g_fl_vbn6'))

        net = tf.concat(axis=3, values=[net, net4])
        net = slim.conv2d_transpose(net, 128, [4, 4], activation_fn=lrelu, stride=2, padding='SAME', scope='g_fl_dcnv4')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_cnv4_1')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_cnv4_2')
        net = lrelu(self.vbn(net, name='g_fl_vbn7'))

        net = tf.concat(axis=3, values=[net, net3])
        net = slim.conv2d_transpose(net, 128, [4, 4], activation_fn=lrelu, stride=2, padding='SAME', scope='g_fl_dcnv3')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_cnv3_1')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_cnv3_2')
        net = lrelu(self.vbn(net, name='g_fl_vbn8'))

        net = tf.concat(axis=3, values=[net, net2])
        net = slim.conv2d_transpose(net, 128, [4, 4], activation_fn=lrelu, stride=2, padding='SAME', scope='g_fl_dcnv2')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_cnv2_1')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_cnv2_2')
        net = lrelu(self.vbn(net, name='g_fl_vbn9'))

        net = tf.concat(axis=3, values=[net, net1])
        net = slim.conv2d_transpose(net, 128, [4, 4], activation_fn=lrelu, stride=2, padding='SAME', scope='g_fl_dcnv1')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_cnv1_1')
        net = slim.conv2d(net, 128, [3,3], activation_fn=lrelu, padding='SAME', scope='g_fl_cnv1_2')
        net = lrelu(self.vbn(net, name='g_fl_vbn10'))

        # # absolute flow
        flow = slim.conv2d(net, 2, [11, 11], activation_fn=None, scope='g_fl_cnv0')
        if base_flow is not None:
            base_flow = tf.reshape(base_flow, tf.stack(flow.get_shape())) # dunno why but w/o get crash on batch_size=None on load
            flow = flow + meshgrid(self.h, self.w, flatten=False) + base_flow
        else:
            flow = flow + meshgrid(self.h, self.w, flatten=False)

        # possible to add few layers here
        flow = tf.tanh(flow)
        # flow = tf.clip_by_value(flow, -1, 1)
        flow = (flow - meshgrid(self.h, self.w, flatten=False))/2.0

        return flow

    def bilateral_filter(self, inps, img_wrt):
        """
        inps: the thing we want to filter, e.g. the flow
        img_wrt: the image we want to compute features wrt, e.g. I2
        """
        img_wrt = tf.ones_like(img_wrt)*255.
        inps_feat = tf.concat(axis=3, values=( img_wrt / self.npwcolor, self.x_feat, self.y_feat))

        nchw_cat = NHWC_to_NCHW(inps)
        nchw_cat_feat = NHWC_to_NCHW(inps_feat)

        ret = bilateral_op_and_grad.bilateral_filters(nchw_cat,
                                nchw_cat_feat)
        ret = NCHW_to_NHWC(ret) # the filter uses nchw, tensorflow nhwc
        return ret

    def log2(self, x):
        l = tf.log(x)
        return l / tf.log(2.)

    def generator_theta(self, i1i2):
        theta = self.theta_generator_small(i1i2)
        i2hat, flow = transformer(U=i1i2[...,0:3], theta=theta, out_size=[self.h, self.w], mode=self.config.transform, name='g_transformer')
        return i2hat, flow

    def generator_flow(self, i1i2):
        flow = self.flow_generator_flownet(i1i2)
        i2hat = transformer(U=i1i2[...,0:3], flow=flow, out_size=[self.h, self.w], mode='Flow', name='g_transformer')
        return i2hat, flow

    def generator_flow_joint(self, i1i2):
        theta = self.theta_generator_small(i1i2)
        i2hat0, flow0 = transformer(U=i1i2[...,0:3], theta=theta, out_size=[self.h, self.w], mode='Projective2D', name='g_transformer')
        flow1 = self.flow_generator_flownet(i1i2)
        i2hat1 = transformer(U=i1i2[...,0:3], flow=flow0+flow1, out_size=[self.h, self.w], mode='Flow', name='g_transformer')
        return i2hat0, i2hat1, flow0, flow1

    def generator_flow_joint_base(self, i1i2):
        theta = self.theta_generator_small(i1i2)
        i2hat0, flow0 = transformer(U=i1i2[...,0:3], theta=theta, out_size=[self.h, self.w], mode='Projective2D', name='g_transformer')
        flow1 = self.flow_generator_flownet(i1i2, flow0)
        i2hat1 = transformer(U=i1i2[...,0:3], flow=flow1, out_size=[self.h, self.w], mode='Flow', name='g_transformer')
        return i2hat0, i2hat1, flow0, flow1

    def stn_only(self, img, flow):
        b,h,w,c = flow.shape
        img = tf.convert_to_tensor(img)
        flow = tf.convert_to_tensor(flow)
        interp= transformer(U=img, flow=flow, out_size=[h, w], mode='Flow', name='stn_only')
        out_interp = self.sess.run([interp])
        return out_interp

    def build_model(self):
        self.inputs, self.output = self.get_batch()

        if(self.config.transform == 'SmoothFlow'):
            print('%% setting up SmoothFlow transform mode')
            self.G, self.flow = self.generator_flow(self.inputs)
            self.lchr = self.charbonnier_distance(self.G, self.output)
            self.loss = self.lchr + self.smooth_penalty(self.flow)

        if(self.config.transform == 'Projective2D'):
            print('%% setting up Projective2D transform mode')
            self.G, self.flow = self.generator_theta(self.inputs)
            self.lchr = self.charbonnier_distance(self.G, self.output)
            self.loss = self.lchr

        if(self.config.transform == 'SmoothJoint'):
            print('%% setting up SmoothJoint transform mode')
            self.G0, self.G, self.flow, self.flow1 = self.generator_flow_joint(self.inputs)
            self.lchr = 0.5 * (self.charbonnier_distance(self.G0, self.output) + self.charbonnier_distance(self.G, self.output))
            self.loss = self.lchr + self.smooth_penalty(self.flow1)
            self.flow+= self.flow1
            self.add_summary_variable(self.G0, 'image', 'outG0')

        if(self.config.transform == 'SmoothJointBase'):
            print('%% setting up SmoothJointBase transform mode')
            self.G0, self.G, self.flow0, self.flow = self.generator_flow_joint_base(self.inputs)
            if self.config.use_bilat:
                self.flow = self.bilateral_filter(self.flow, self.inputs[...,3:6])
            self.lchr = 0.5 * (self.charbonnier_distance(self.G0, self.output) + self.charbonnier_distance(self.G, self.output))
            self.loss = self.lchr + self.smooth_penalty(self.flow)
            self.add_summary_variable(self.G0, 'image', 'outG0')

        if(self.config.transform == 'BilatJoint'):
            print('%% setting up SmoothJointBase transform mode')
            self.G0, self.G, self.flow0, self.flow = self.generator_flow_joint_base(self.inputs)
            self.flow = self.bilateral_filter(self.flow, self.inputs[...,3:6])
            self.lchr = 0.5 * (self.charbonnier_distance(self.G0, self.output) + self.charbonnier_distance(self.G, self.output))
            self.loss = self.lchr + self.smooth_penalty(self.flow)
            self.add_summary_variable(self.G0, 'image', 'outG0')

        self.add_summary_variable(self.lchr, 'scalar', 'charbonnier_distance')
        self.add_summary_variable(self.loss, 'scalar', 'loss')
        self.add_summary_variable(self.inputs[...,0:3], 'image', 'img1')
        self.add_summary_variable(self.inputs[...,3:6], 'image', 'img2')
        self.add_summary_variable(tf.expand_dims(self.flow[...,0], axis=3), 'image', 'fx')
        self.add_summary_variable(tf.expand_dims(self.flow[...,1], axis=3), 'image', 'fy')
        self.add_summary_variable(self.G, 'image', 'outG')

        tf.add_to_collection('G', self.G)
        tf.add_to_collection('flow', self.flow)
        tf.add_to_collection('inputs_big', self.inputs_big)
        tf.add_to_collection('lchr', self.lchr)
        tf.add_to_collection('loss', self.loss)

        # can select trainable here 
        self.vars = tf.trainable_variables()
        print('%% Total parameters : ', np.sum( [np.prod(dim) for dim in [variable.get_shape() for variable in tf.trainable_variables()] ] ))
        print('%% Total trainable  : ', np.sum( [np.prod(dim) for dim in [variable.get_shape() for variable in self.vars] ] ))

        save_vars = [var for var in tf.trainable_variables() if 'g_conv' in var.name or 'g_fl' in var.name or 'g_fc' in var.name]
        self.saver = tf.train.Saver(save_vars)

        if self.config.mode == 'train':
            restore_vars = [var for var in tf.trainable_variables() if 'g_conv' in var.name or 'g_fl' in var.name]
            self.loader = tf.train.Saver(restore_vars)
        else:
            self.loader = tf.train.Saver()

    def train(self):
        """Trainer"""
        killer = GracefulKiller()
        adam = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1)
        optim = adam.minimize(self.loss, var_list=self.vars)

        tf.global_variables_initializer().run(session=self.sess)
        tf.local_variables_initializer().run(session=self.sess) #needed to start the pool of threads for the input queue

        self.load()

        self.summary = tf.summary.merge(self.summary_list)
        self.writer = tf.summary.FileWriter(self.config.log_dir, self.sess.graph)

        inps, _ = self.get_symbolic_batch()
        counter = 0

        tf.get_default_graph().finalize()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)
        start_time = time.time()

        for epoch in xrange(self.config.epochs):
            for idx in xrange(0, self.config.num_batches ):
                counter += 1
                if counter < 500:
                    lr = self.config.learning_rate
                else:
                    lr = self.config.learning_rate /10.
                feed_dict = {self.inputs_big: inps.eval(session=self.sess)}
                batch_loss, batch_lchr, summary_str, _, batch_flow = self.sess.run([self.loss, self.lchr, self.summary, optim, self.flow], feed_dict=feed_dict)
                self.writer.add_summary(summary_str, counter)
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, lchr: %.8f, flow m|M: %.3f | %.3f" %
                      (epoch, idx, self.config.num_batches, time.time()-start_time,
                       np.mean(batch_loss), np.mean(batch_lchr), np.min(batch_flow), np.max(batch_flow)) )
                if np.mod(counter, 100) == 0:
                    self.save(counter)
        if killer.kill_now:
            print('Got a SIGKILL, snapshotting model before exiting..')
            self.save(counter)
            print('Bye!')
            return
        coord.request_stop()
        coord.join(threads)

    def inference(self, i=0):
        inp, gt, val, orig_shape = self.get_testing_pair(i)
        # inp, gt, bound = self.reader.get_testing_pair(i, reverse=True)

        feed_dict={self.inputs_big: inp}
        batch_lchr, batch_loss, batch_G, batch_flow, batch_G0, batch_flow0 = self.sess.run([self.lchr, self.loss, self.G, self.flow, self.G0, self.flow0], feed_dict=feed_dict)
        
        print("lchr: %.8f, loss: %.8f, flow m/M: %.3f %.3f | %.3f %.3f" %
              (np.mean(batch_lchr), np.mean(batch_loss), np.min(batch_flow[...,0]), np.max(batch_flow[...,0]), np.min(batch_flow[...,1]), np.max(batch_flow[...,1])) )
        if self.verbose:
            io.imsave('output_kitti12/{0:06d}_flow1.png'.format(i[0]), flowToColor(-batch_flow[0]))
            io.imsave('output_kitti12/{0:06d}_flow0.png'.format(i[0]), flowToColor(-batch_flow0[0]))
            io.imsave('output_kitti12/{0:06d}_G0.png'.format(i[0]),  np_yuv2rgb(255*(1+batch_G0[0])/2).astype(np.uint8))
            io.imsave('output_kitti12/{0:06d}_G.png'.format(i[0]), np_yuv2rgb(255*(1+batch_G[0])/2).astype(np.uint8))
            io.imsave('output_kitti12/{0:06d}_inps2.png'.format(i[0]), np_yuv2rgb(255*(1+inp[0,...,3:6])/2).astype(np.uint8))
            io.imsave('output_kitti12/{0:06d}_inps1.png'.format(i[0]), np_yuv2rgb(255*(1+inp[0,...,0:3])/2).astype(np.uint8))
        batch_flow[...,0] *= -self.w
        batch_flow[...,1] *= -self.h
	print('inference: batchflow shape',batch_flow.shape)
        return batch_flow,gt, self.inputs_small.eval(feed_dict=feed_dict), batch_G, val, orig_shape

    def save(self, step, keep=False):
        print('Saved model at', self.config.check_save)
        if not os.path.exists(self.config.check_save):
            os.makedirs(self.config.check_save)
        self.saver.save(self.sess, self.config.check_save+'/model', global_step=step)

    def load(self, useMeta=False):
        if self.config.check_load is not None:
            fn = self.config.check_load
            print(" [*] Reading checkpoints from", fn)
            ckpt = tf.train.get_checkpoint_state(fn)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                if useMeta: #load the meta
                    meta = os.path.join(fn,ckpt_name)+'.meta'
                    print(' [*] Loading model from meta', meta)
                    self.loader = tf.train.import_meta_graph(meta)
                    self.G = tf.get_collection('G')[0]
                    self.flow = tf.get_collection('flow')[0]
                    self.inputs_big = tf.get_collection('inputs_big')[0]
                    self.lchr = tf.get_collection('lchr')[0]
                    self.loss = tf.get_collection('loss')[0]
                self.loader.restore(self.sess, os.path.join(fn, ckpt_name))
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

    def imgplot(self, img, num, mode=0):
            ax = plt.subplot(num)
            if mode==0 :
                plt.imshow(img)
            elif mode==1:
                #img = resize(img,(self.orig_h, self.orig_w),preserve_range=True, mode='edge')
                img = np_yuv2rgb(255 * (1+img)/2).astype(np.uint8)
                plt.imshow(img)
                # io.imshow(img)
                # plt.imshow((1+img)/2.0)
            elif mode==2:
                plt.imshow((1+img)/2.0)
            elif mode==3:
                img = np_yuv2rgb(img).astype(np.uint8)
                plt.imshow(img)

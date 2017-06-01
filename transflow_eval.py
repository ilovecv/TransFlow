from copy import deepcopy
import glob
import os
import sys
import copy
from os.path import join
from model import InterpolNet
import tensorflow as tf
import tensorflow.contrib.slim as slim
import skimage.io as io
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from utils import *
import time
from utils_misc import *
from scipy.spatial.distance import euclidean
import png
import itertools
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize
from flowToColor import flowToColor
from read_sintel_flow import SintelReader
from data_io import data_io
from yuv import *
from CtypesPermutohedralLattice import CtypesPermutohedralLattice
from bilateral_utils import *

class TransflowExperiment(data_io):
    def __init__(self, runtime):
        data_io.__init__(self, runtime.config)
        self.runtime = runtime
        self.config = runtime.config
        self.stride = 1
        self.dist_thr = 30

    def imgplot(self, img, num, mode=0):
        ax = plt.subplot(num)
        if mode==0 :
            plt.imshow(img)
        elif mode==1:
            img = np_yuv2rgb(255 * (1+img)/2).astype(np.uint8)
            plt.imshow(img)
        elif mode==2:
            plt.imshow((1+img)/2.0)

    def plot(self, img, num):
        ax = plt.subplot(num)
        plt.plot(img)

    def reshift_flows(self, u, v):
        def _avgnz(xx,iii,jjj,mm=1):
            aa=0 ; nn=0
            for ii in range(iii-mm,iii+mm+1):
                for jj in range(jjj-mm,jjj+mm+1):
                    if ii>0 and ii<h and jj>0 and jj<w:
                        if xx[ii,jj] != 0:
                            aa += xx[ii,jj]
                            nn += 1.0
            if nn>0:
                mean=aa/nn
                return mean
            else:
                return 0
        def _rmzeros(un, vn):
            for ii in range(h):
                for jj in range(w):
                    if un[ii,jj] == 0:
                        un[ii,jj] = _avgnz(un,ii,jj)
                    if vn[ii,jj] == 0:
                        vn[ii,jj] = _avgnz(vn,ii,jj)
            return un, vn

        h,w = u.shape
        un=np.zeros_like(u)
        vn=np.zeros_like(v)
        for ii in range(h):
            for jj in range(w):
                ni=np.clip(ii-v[ii,jj],0,h-1).astype(int)
                nj=np.clip(jj-u[ii,jj],0,w-1).astype(int)
                un[ni,nj]=u[ii,jj]
                vn[ni,nj]=v[ii,jj]
        un, vn = _rmzeros(un, vn)
        un, vn = _rmzeros(un, vn)
        return un, vn

    def eval_flow(self, flows, gts, inps, outs, idx, val, orig_shape, doPrint=True, doDisplay=False, save16=True):
        acc = [] ; ape = []
        
        h,w,c = flows[0].shape
        
        for sample in range(flows.shape[0]):
            u, v = (flows[sample, ..., 0], flows[sample, ..., 1])
            u, v = self.reshift_flows(u,v)

            if self.db_name == 'KITTI_FLOW_TEST':
                    gts = np.expand_dims(np.zeros(orig_shape), axis=0)
            u_gts, v_gts, val = (gts[sample, ..., 0], gts[sample, ..., 1], gts[sample, ..., 2])
            # Downsample the ground truth (manually 'cause library downsamplings do weird things to "improve" quality)
            r_h = u_gts.shape[0] / float(h) #ratios between gts and resized images
            r_w = u_gts.shape[1] / float(w)

            EPE_THR = 5.
            if True: # eval
                
                u_up = resize(u, (self.orig_h, self.orig_w), preserve_range=True, mode='edge')*r_w
                v_up = resize(v, (self.orig_h, self.orig_w), preserve_range=True, mode='edge')*r_h
                
                self.write_of_16bits(u_up, v_up, None, idx)

                dists = np.sqrt((u_up - u_gts)**2 + (v_up - v_gts)**2)
                valid = dists[np.nonzero(val)]
                ape_cur = np.mean(valid)
                acc_cur = np.sum(valid < EPE_THR) / float(len(valid))
                acc.append(acc_cur)
                ape.append(ape_cur)
                im = flowToColor(np.stack((u_up, v_up), axis=2))
                io.imsave('tmp/ccflow_{0:05d}'.format(idx)+'.png', im)
                print('ACC@'+str(EPE_THR)+'(no filtro):', acc_cur)
                print('APE (no filtro)', ape_cur)
        return acc, ape

    def write_of_16bits(self, u, v, val, i):
        if not os.path.isdir('flows_16bits'):
            os.system('mkdir flows_16bits')
        of16 = np.ones((u.shape[0], u.shape[1], 3), dtype=np.uint16)
        print 'SAVE16BITS: shapes', of16.shape
        of16[...,0] = u * 64 + np.power(2.,15)
        of16[...,1] = v * 64 + np.power(2.,15)
        if val != None: #val is not strictly needed by the matlab evaluation
            of16[...,2] = val
        with open('submission/{0:06d}_10'.format(i)+'.png', 'wb') as f:
            writer = png.Writer(width=u.shape[1], height=u.shape[0], bitdepth=16)
            of_list = of16.reshape(-1, of16.shape[1]*of16.shape[2]).tolist()
            writer.write(f, of_list)

    def dump_outs(self, idx, inps, outs, flows, gts=None):
        if not os.path.isdir('outputs'):
            os.system('mkdir outputs')
        inps = (inps+1)/2.
        outs = (outs+1)/2.
        for j in range(inps.shape[0]): #save interpolated results to file
            io.imsave('outputs/{0:05d}-i1'.format(idx+j)+'.png', inps[j][...,0:3])
            io.imsave('outputs/{0:05d}-i2'.format(idx+j)+'.png', inps[j][...,3:6])
            io.imsave('outputs/{0:05d}-gen'.format(idx+j)+'.png', outs[j])
            io.imsave('outputs/{0:05d}-gen-rgb'.format(idx+j)+'.png', np_yuv2rgb(outs[j]*255).astype(np.uint8))
            io.imsave('outputs/{0:05d}-i1-rgb'.format(idx+j)+'.png', np_yuv2rgb(inps[j][...,0:3]*255).astype(np.uint8))
            io.imsave('outputs/{0:05d}-i2-rgb'.format(idx+j)+'.png', np_yuv2rgb(inps[j][...,3:6]*255).astype(np.uint8))
            io.imsave('outputs/{0:05d}-fl-rgb'.format(idx+j)+'.png', flowToColor(flows[j]))
            if(gts is not None):
                io.imsave('outputs/{0:05d}-gt-rgb'.format(idx+j)+'.png', flowToColor(gts[j,...,0:2]))

    def run_experiment(self, mode='eval'):
            self.runtime.load()
            all_acc = [] ; all_ape = []
            r_err = []
            for i in range(0, self.n_imgs, 1):
                print('\nrunning experiment idx',i)
                if mode == 'eval':
                    flows, gts, inps, outs, val, orig_shape = self.runtime.inference([i, i+1])
                    print(flows.shape)
                elif mode == 'train_eval':
                    flows, gts, inps, outs = self.runtime.train_inference(i) #LUCA!
                acc, ape = self.eval_flow(flows, gts, inps, outs, i, val, orig_shape)
                ii = np_yuv2rgb((1+inps[0][...,3:6])/2*255).astype(np.uint8)
                oo = np_yuv2rgb((1+outs[0])/2*255).astype(np.uint8)
                err = self.eval_reconstruction(ii,oo)
                r_err.append(err)
                print 'RECONSTRUCTION ERROR:', err
                all_acc.extend(acc) ; all_ape.extend(ape)
                all_acc_bil.extend(acc_bil) ; all_ape_bil.extend(ape_bil)
            print('\nExperiment completed')
            print('\tACC@5   ', np.mean(all_acc[0:self.n_imgs]))
            print('\tAPE     ', np.mean(all_ape[0:self.n_imgs]))
            print('\tRECONSTRUCTION ERR   ', np.mean(r_err))
    
    def eval_reconstruction(self, inp, gt):
        print np.min(inp), np.max(inp)
        print np.min(gt), np.max(gt)
        return np.mean(np.abs(inp.astype(np.float32)-gt.astype(np.float32)))

    def load_reconstruction_data(self, i, data, imgs):

        fl = data[i]
        print fl
        flow = np.load(fl)['arr_0']
        im_id = int(fl.split('/')[-1][:-4])

        seq = fl.split('/')[5]
        print self.config.mode
        if self.config.mode == 'eval-vkitti':
            i1 = join('VirtualKitti/vkitti_1.3.1_rgb', seq, 'morning', '{0:05d}'.format(im_id)+'.png')
            i2 = join('VirtualKitti/vkitti_1.3.1_rgb', seq, 'morning', '{0:05d}'.format(im_id+1)+'.png')
        else:
            i1 = join('/WindowsShares/MAJINBU/DREYEVE/DATA/', seq, 'frames', '{0:06d}'.format(im_id)+'.jpg')
            i2 = join('/WindowsShares/MAJINBU/DREYEVE/DATA/', seq, 'frames', '{0:06d}'.format(im_id+1)+'.jpg')
        im1 = io.imread(i1)
        im2 = io.imread(i2)
        h,w,_ = flow.shape

        im1 = resize(im1, (h,w),preserve_range=True, mode='edge')
        im2 = resize(im2, (h,w),preserve_range=True, mode='edge')

        return im1, im2, flow

    def run_experiment_reconstruction(self, competitor='deepflow'):
            self.runtime.load()

            self.root = '/media/nascalde/transflow_competitors'
            self.data_list = join(self.root, competitor, 'data.lst')
            self.imgs_list = '/WindowsShares/MAJINBU/DREYEVE/DATA/imgs.lst'

            data = [line.rstrip() for line in open(self.data_list)]
            imgs = [line.rstrip() for line in open(self.imgs_list)]

            n_imgs = len(data)
            error = []
            for i in range(0, n_imgs,1):
                print('\nrunning experiment idx',i)
                im1, im2, flow = self.load_reconstruction_data(i, data, imgs)
                flow = flow.astype(np.float32)
                interp = self.runtime.stn_only(np.expand_dims(im1, axis=0), np.expand_dims(flow,axis=0))
                print im2.shape, interp[0].shape
    
                rgbflow = flowToColor(flow)
                io.imsave('reconstruction_flownetv2/{0:05d}-ccflow'.format(i)+'.png', rgbflow)    
                io.imsave('reconstruction_flownetv2/{0:05d}-i2'.format(i)+'.png', im2.astype(np.uint8))
                io.imsave('reconstruction_flownetv2/{0:05d}-interp'.format(i)+'.png', interp[0][0].astype(np.uint8))    
                err = self.eval_reconstruction(im2, interp[0])
                print 'Reconstruction error', err
                error.append(err)
            print('\nExperiment completed')
            print('\tReconstruction error: ', np.mean(error))
    
    def run_eval_vkitti(self, competitor, mode='eval'):
        self.mode = mode
        all_acc = [] ; all_ape = []
        self.root = '/media/nascalde/transflow_vkitti_comp'
        self.data_list = join(self.root, self.runtime.config.competitor, 'data.lst')
        self.imgs_list = 'VirtualKitti/imgs.lst'

        data = [line.rstrip() for line in open(self.data_list)]
        imgs = [line.rstrip() for line in open(self.imgs_list)]

        n_imgs = len(data)
        error = []
        for i in range(0, n_imgs):
            print('\nrunning experiment idx',i)
            im1, im2, flow = self.load_reconstruction_data(i, data, imgs)
            print imgs[i]
            u, v, val = self.load_flow_gt_virtualkitti(imgs[i])
            gt = np.expand_dims(np.stack((u,v,val), axis=2),axis=0)
            flow = np.expand_dims(flow, axis=0)
            acc, ape = self.eval_flow(flow, gt, None, None, i, val)
            all_acc.extend(acc) ; all_ape.extend(ape)
        print('\nExperiment completed')
        print('\tACC@5 ', np.mean(all_acc[0:self.n_imgs]))
        print('\tAPE   ', np.mean(all_ape[0:self.n_imgs]))

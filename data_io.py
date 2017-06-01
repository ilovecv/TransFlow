import cv2
import sys
import copy
import glob
from os.path import join
import tensorflow as tf
import numpy as np
from skimage.transform import resize
import skimage.io as io
import struct
import itertools
import png
import numpy as np
import tensorflow as tf

RGB_TO_YUV = np.array([
    [ 0.299,     0.587,     0.114],
    [-0.168736, -0.331264,  0.5],
    [ 0.5,      -0.418688, -0.081312]])
YUV_TO_BGR = np.array([
    [1.0,  0.0,      1.402],
    [1.0, -0.34414, -0.71414],
    [1.0,  1.772,    0.0]])
YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)

def np_rgb2yuv(im):
    im = im[...,::-1]
    return np.tensordot(im, RGB_TO_YUV, ([2], [1])) + YUV_OFFSET

def np_yuv2rgb(im):
    return np.tensordot(im.astype(float) - YUV_OFFSET, YUV_TO_BGR, ([2], [1]))[...,::-1]#BGR->RGB

class data_io(object):
    def __init__(self, config, colorspace='yuv',epochs=100):

        self.db_name = config.dbname
        self.config = config
        if self.db_name == 'KITTI_RAW':
                self.orig_h = 370
                self.orig_w = 1224
                self.h = 128
                self.w = 384
                print('Setting up the tfrecord queue')
                if config.where == 'modena':
                    self.name = 'KITTI_RAW/i1i2raw_yuv.tfrecord'
                else:
                    self.name = ''

                self.filename_queue = tf.train.string_input_producer(
                                        [self.name], num_epochs=epochs, shuffle=False)
        elif self.db_name == 'KITTI_FLOW':
            self.name = ''
            self.orig_h = 370
            self.orig_w = 1224
            self.h = 128
            self.w = 384
            if config.where == 'modena':
                self.root = 'KITTI_FLOW/training/'
            else:
                self.root = ''
            self.img_folder = 'colored_0'
            self.flow_folder = 'flow_noc'
            self.n_imgs = 194
            imgs_10 = [x.split('/')[-1] for x in sorted(glob.glob(join(self.root, self.img_folder, '*.png'))) if '_10' in x]
            imgs_11 = [x.split('/')[-1] for x in sorted(glob.glob(join(self.root, self.img_folder, '*.png'))) if '_11' in x]
            self.img_pairs = list(zip(imgs_10, imgs_11))
        elif self.db_name == 'KITTI_FLOW_TEST':
            self.name = ''
            self.orig_h = 370
            self.orig_w = 1224
            self.h = 128
            self.w = 384
            if config.where == 'modena':
                self.root = 'KITTI_FLOW/testing/'
            else:
                self.root = ''
            self.img_folder = 'colored_0'
            self.flow_folder = 'flow_noc'
            self.n_imgs = 195
            imgs_10 = [x.split('/')[-1] for x in sorted(glob.glob(join(self.root, self.img_folder, '*.png'))) if '_10' in x]
            imgs_11 = [x.split('/')[-1] for x in sorted(glob.glob(join(self.root, self.img_folder, '*.png'))) if '_11' in x]
            self.img_pairs = list(zip(imgs_10, imgs_11))
        elif self.db_name == 'KITTI_2015_TEST':
            self.name = ''
            self.orig_h = 370
            self.orig_w = 1224
            self.h = 128
            self.w = 384
            if config.where == 'modena':
                self.root = 'KITTI_2015/testing/'
            else:
                self.root = ''
            self.img_folder = 'image_2'
            self.flow_folder = 'flow_noc'
            self.n_imgs = 200
            imgs_10 = [x.split('/')[-1] for x in sorted(glob.glob(join(self.root, self.img_folder, '*.png'))) if '_10' in x]
            imgs_11 = [x.split('/')[-1] for x in sorted(glob.glob(join(self.root, self.img_folder, '*.png'))) if '_11' in x]
            self.img_pairs = list(zip(imgs_10, imgs_11))
        elif self.db_name == 'KITTI_2015':
            if config.where == 'modena':
                self.name = 'KITTI_2015/train.tfrecord'
            else:
                self.name = ''
            self.orig_h = 370
            self.orig_w = 1224
            self.h = 128
            self.w = 384
            if config.where == 'modena':
                self.root = 'KITTI_2015/training/'
            else:
                self.root = ''
            self.img_folder = 'image_2'
            self.flow_folder = 'flow_noc'
            self.n_imgs = 200
            imgs_10 = [x for x in sorted(glob.glob(join(self.root, self.img_folder, '*.png'))) if '_10' in x]
            imgs_11 = [x for x in sorted(glob.glob(join(self.root, self.img_folder, '*.png'))) if '_11' in x]
            self.img_pairs = list(zip(imgs_10, imgs_11))
            self.filename_queue = tf.train.string_input_producer(
                                    [self.name], num_epochs=epochs, shuffle=False)
        elif self.db_name == 'VirtualKitti':
            if config.where == 'modena':
                self.name = 'VirtualKitti/vkitti.tfrecord'
            else:
                self.name = ''
            self.orig_h = 375
            self.orig_w = 1242
            self.h = 128
            self.w = 384
            if config.where == 'modena':
                self.root = 'VirtualKitti/'
            else:
                self.root = ''
            self.img_folder = 'vkitti_1.3.1_rgb'
            self.flow_folder = 'vkitti_1.3.1_flowgt'
            imgs = []
            name = 'imgs.lst'
            with open(join(self.root, name)) as f:
                tmp = []
                cur_seq = '0001'
                for line in f:
                    if line.split('/')[2] == cur_seq:
                        tmp.append(line.rstrip())
                    else:
                        imgs.append(copy.deepcopy(tmp))
                        tmp = []
                        cur_seq = line.split('/')[2]
            self.img_pairs = []
            for s in imgs:
                for fr_id in range(0, len(s)-1):
                    self.img_pairs.append((s[fr_id], s[fr_id+1]))
            self.n_imgs = len(self.img_pairs)
            self.filename_queue = tf.train.string_input_producer(
                                    [self.name], num_epochs=epochs, shuffle=False)
        elif self.db_name == 'Sintel':
            if config.where == 'modena':
                self.name = 'Sintel/sintel_full.tfrecord'
            else:
                self.name = ''
            self.orig_h = 436
            self.orig_w = 1024
            self.h = 128
            self.w = 256
            if config.where == 'modena':
                self.root = 'Sintel/'
            else:
                self.root = '' 
            self.img_folder = 'training/final'
            self.flow_folder = 'training/flow'
            imgs = []
            if self.config.mode == 'eval':
                name = 'training.lst'
            else:
                name = 'images.lst'
            with open(join(self.root, name)) as f:
                tmp = []
                cur_seq = 'alley_1'
                for line in f:
                    if line.split('/')[3] == cur_seq:
                        tmp.append(line.rstrip())
                    else:
                        imgs.append(copy.deepcopy(tmp))
                        tmp = []
                        cur_seq = line.split('/')[3]
            self.img_pairs = []
            for s in imgs:
                for fr_id in range(0, len(s)-1):
                    self.img_pairs.append((s[fr_id], s[fr_id+1]))
            self.n_imgs = len(self.img_pairs)
            self.filename_queue = tf.train.string_input_producer(
                                    [self.name], num_epochs=epochs, shuffle=True)
        elif self.db_name == 'dreyeve':
            if config.where == 'modena':
                self.name = '/home/stefano/workspace/DeepMatch/dreyeve/dreyeve1.tfrecord' 
            else:
                self.name = ''
            self.orig_h = 370
            self.orig_w = 1224
            self.h = 128 
            self.w = 384
            if config.where == 'modena':
                self.root = '/WindowsShares/MAJINBU/DREYEVE/DATA/'
            else:
                self.root = ''
            self.img_folder = 'frames'
            self.flow_folder = 'no'
            imgs = []
            name = 'imgs.lst'
            with open(join(self.root, name)) as f:
                tmp = []
                cur_seq = '04'
                for line in f:
                    if line.split('/')[5] == cur_seq:
                        tmp.append(line.rstrip())
                    else:
                        imgs.append(copy.deepcopy(tmp))
                        tmp = []
                        cur_seq = line.split('/')[5]
            self.img_pairs = []
            for s in imgs:
                for fr_id in range(0, len(s)-1):
                    self.img_pairs.append((s[fr_id], s[fr_id+1]))
            self.n_imgs = len(self.img_pairs)
            self.filename_queue = tf.train.string_input_producer(
                                    [self.name], num_epochs=epochs, shuffle=True)
        elif self.db_name == 'DAVIS':
            if config.where == 'modena':
                self.name = '/majinbu/public/DAVIS/davis.tfrecord'
            else:
                self.name = '/u/big/trainingdata/DAVIS/davis.tfrecord'
            self.orig_h = 480
            self.orig_w = 854
            self.h = 256
            self.w =448
            if config.where == 'modena':
                self.root = '/majinbu/public/DAVIS'
            else:
                self.root = '/u/big/trainingdata/DAVIS/'
            self.img_folder = '480p'
            self.flow_folder = 'no'
            imgs = []
            name = 'train.lst'
            with open(join(self.root, name)) as f:
                tmp = []
                cur_seq = 'bear'
                for line in f:
                    if line.split('/')[6] == cur_seq:
                        tmp.append(line.rstrip())
                    else:
                        imgs.append(copy.deepcopy(tmp))
                        tmp = []
                        cur_seq = line.split('/')[6]
            self.img_pairs = []
            for s in imgs:
                for fr_id in range(0, len(s)-1):
                    self.img_pairs.append((s[fr_id], s[fr_id+1]))
            self.n_imgs = len(self.img_pairs)
            self.filename_queue = tf.train.string_input_producer(
                                    [self.name], num_epochs=epochs, shuffle=True)
        elif self.db_name == 'template': #template to be used to create new db entries, all these fields must be filled
            self.name = '/path/to/tfrecord'
            self.orig_h = 0
            self.orig_w = 0
            self.h = 0
            self.w = 0
            self.root = '/path/to/the/dataset/root/'
            self.img_folder = 'image_folder_name'
            self.flow_folder = 'flow_folder_name'
            self.img_pairs = [] # pairs of image names used for testing
            self.n_imgs = 0
            self.finelame_queue = tf.train.string_input_producer(
                                    [self.name], num_epochs=epochs, shuffle=True)
        else:
            print('self.db_name not recognized, the options are: [KITTI_RAW, Sintel, KITTI_FLOW, demo]')
            sys.exit(0)

    def load_flow_gt_kitti(self, img_pair):
        if self.db_name == 'KITTI_FLOW':
            name = join(self.root, self.flow_folder, img_pair[0])
        else:
            n = img_pair[0].split('/')[-1]
            name = join(self.root, self.flow_folder, n)

        r = png.Reader(name)
        data = r.asDirect()
        im = np.vstack(itertools.imap(np.uint16, data[2]))
        im3 = np.reshape(im, (im.shape[0],im.shape[1]/3, 3))

        fu = (im3[...,0]-np.power(2.,15))/64.
        fv = (im3[...,1]-np.power(2.,15))/64.
        val = im3[...,2]
        return fu, fv, val
    
    def load_flow_gt_virtualkitti(self, img_pair):
        name = join(self.root, self.flow_folder, img_pair[0].split('/')[2], 'morning', img_pair[0].split('/')[4])
        bgr = cv2.imread(name, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        h, w, _c = bgr.shape
        assert bgr.dtype == np.uint16 and _c == 3
        # b == invalid flow flag == 0 for sky or other invalid flow
        invalid = bgr[..., 0] == 0
        out_flow = 2.0 / (2**16 - 1.0) * bgr[..., 2:0:-1].astype('f4') - 1
        out_flow[..., 0] *= w - 1
        out_flow[..., 1] *= h - 1
        out_flow[invalid] = 0  # or another value (e.g., np.nan)
        return out_flow[...,0], out_flow[...,1],  invalid == False

    def load_flow_gt_sintel(self, img_pair):
        name = join(self.root, self.flow_folder, img_pair[0].split('/')[3], img_pair[0].split('/')[4][:-4]+'.flo')
        f = open(name, 'rb')
        
        occl = join(self.root, 'training/occlusions', img_pair[0].split('/')[3], img_pair[0].split('/')[4])
        invalid = join(self.root, 'training/invalid', img_pair[0].split('/')[3], img_pair[0].split('/')[4])
        print 'occlusion:', occl
        print 'invalid:', invalid

        oc = io.imread(occl, as_gray=True)
        inv = io.imread(invalid, as_gray=True)

        val = oc + inv
        val[val==0] = 1 
        val[val!=1] = 0 

        head = struct.unpack('<f', f.read(4))[0]
        assert head == 202021.25
        w = struct.unpack('<i', f.read(4))[0]
        h = struct.unpack('<i', f.read(4))[0]
        data = struct.unpack('<'+'f'*w*h*2, f.read(w*h*2*4))

        fmap = np.zeros((w*h,2))
        pos=0
        for i in range(0, len(data), 2):
            fmap[pos,0] = data[i]
            fmap[pos,1] = data[i+1]
            pos += 1
        fmap = fmap.reshape((h,w,2))
        return fmap[...,0],fmap[...,1], val


    def get_symbolic_batch(self, idx=0):
        if self.db_name in ['KITTI_RAW', 'Sintel', 'VirtualKitti', 'KITTI_2015', 'dreyeve', 'DAVIS']:
            reader = tf.TFRecordReader()

            _, serialized_example = reader.read(self.filename_queue)
            features = tf.parse_single_example(serialized_example,
                        features={'images_raw':tf.FixedLenFeature([], tf.string)})

            img_pair = tf.decode_raw(features['images_raw'], tf.uint8)
            pair_shape = tf.stack([self.orig_h, self.orig_w, 6])
            img_pair = tf.cast(tf.reshape(img_pair, pair_shape), tf.float32)

            img_pair = img_pair * (2./255) - 1. #scale back to [-1,1]
            imgs = tf.train.shuffle_batch([img_pair],
                                          batch_size=self.config.batch_size,
                                          capacity=100+self.config.batch_size*3,
                                          num_threads=2,
                                          min_after_dequeue=100)
            return imgs, None #don't use gt for now 

        elif self.config.mode == 'train_eval' or self.db_name == 'KITTI_FLOW':
            b_1, g_1, _ = self.get_testing_pair([idx,idx+1])
            self.batch_1 = b_1 
            return tf.convert_to_tensor(self.batch_1), None
        else:
            print('db_name not recognized in get_symbolic_batch')
            sys.exit(0)

    def get_testing_pair(self, idx, reverse=False):
        """returns a testing pair (input, gt) with shape ((h,w,6),(h,w,2)) """
        if type(idx) == list: # if called using a list, batch the data
            if idx[1] >= len(self.img_pairs):
                idx[1] = len(self.img_pairs)
            im_batch = np.zeros((self.config.batch_size, self.orig_h, self.orig_w, 6))
            fl_batch = np.zeros((self.config.batch_size, self.orig_h, self.orig_w, 3))
            for ii in range(idx[0], idx[1]):
                imgs = []
                for ind in range(2):
                    i = self.img_pairs[ii][ind] #read the images
                    if self.db_name == 'KITTI_FLOW':
                        i = join(self.root, self.img_folder, i)
                    im = io.imread(i)
                    orig_shape = im.shape
                    im = resize(im, (self.orig_h, self.orig_w), preserve_range=True)
                    im = np_rgb2yuv(im) / 128. - 1.0
                    imgs.append(im)
                if self.db_name == 'KITTI_FLOW' or self.db_name == 'KITTI_2015': #read the gt flows
                    u, v, val = self.load_flow_gt_kitti(self.img_pairs[ii])
                elif self.db_name == 'Sintel':
                    u, v, val = self.load_flow_gt_sintel(self.img_pairs[ii])
                elif self.db_name  == 'VirtualKitti':
                    u, v, val = self.load_flow_gt_virtualkitti(self.img_pairs[ii])
                else:
                    u = v = val = np.zeros((self.orig_h, self.orig_w))
                if reverse:
                    im_batch[ii-idx[0]] = np.concatenate(imgs[::-1], axis=2)
                else:
                    im_batch[ii-idx[0]] = np.concatenate(imgs, axis=2)

                fl_batch[ii-idx[0]] = np.stack((u,v,val), axis=2)[0:self.orig_h, 0:self.orig_w,:]
            return im_batch, fl_batch, np.zeros((self.orig_h, self.orig_w)), orig_shape
        else:
            imgs = []
            for ind in range(2):
                i = self.img_pairs[idx][ind] #read the images
                im = resize(io.imread(join(self.root, self.img_folder, i)), (self.orig_h, self.orig_w))*255.
                im -= 128
                im /= np.max(np.abs(im), axis=0)
                imgs.append(im.astype(np.float32))

                if self.db_name == 'KITTI_FLOW': #read the gt flows
                    u, v, _ = self.load_flow_gt_kitti(self.img_pairs[idx])
                elif self.db_name == 'Sintel':
                    u, v = load_flow_gt_sintel(self.img_pairs[idx])
            return np.expand_dims(np.concatenate(imgs, axis=2),0), np.expand_dims(np.stack((u,v), axis=2)[0:self.orig_h, 0:self.orig_w,:],0), 1

    def get_batch(self):
        """ unified interface to ask for data"""
        self.inputs_big = tf.placeholder(tf.float32, [None,self.orig_h, self.orig_w, 6], name='inputs_big')
        self.gt_flow = tf.placeholder(tf.float32, [None,self.orig_h, self.orig_w, 2], name='gt_flow')
        if(self.config.augment):
            self.output = self.inputs_big[...,3:6] # I2
            scale_max = 2 
            jitter_h_max = 50 
            jitter_w_max = 50 
            return self.augmenter._crop_random(self.inputs_big, jitter_h_max=jitter_h_max, jitter_w_max=jitter_w_max, scale_max=scale_max)
        else:
            self.inputs_small = tf.image.resize_images(self.inputs_big, [self.h, self.w])
            print 'inputs small size', self.inputs_small.get_shape()
            return self.inputs_small, self.inputs_small[...,3:6]


import struct
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import skimage.io as io
from skimage.transform import resize
from utils import pp, visualize, to_json
from os.path import join
import copy
from yuv import *
import matplotlib.pyplot as plt

np.random.seed(23)
# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) #flush

flags = tf.app.flags
flags.DEFINE_string("colorspace", 'yuv', "colorspace to be used [yuv, rgb]")
flags.DEFINE_string("png_list", 'KITTI_RAW/images.lst', "Input PNG file list")
flags.DEFINE_string("seq_list", 'KITTI_RAW/sequences.lst', "Input sequence list")
flags.DEFINE_string("seq_index", 'KITTI_RAW/seq_index.pkl', "Input//Output transitory sequence//image list")
flags.DEFINE_string("output_db", 'KITTI_RAW/i1i2raw_yuv.tfrecord', "TFRecords database")
flags.DEFINE_string("dbname", 'KITTI_RAW', "Name of the dataset [KITTI_RAW|Sintel]")
flags.DEFINE_boolean("shuffle", True, "Apply Real-Time data augmentation [False]")

class tfrWriter():
    def __init__(self, config, shape=(1080/2., 1920/2.)):
        self.h = shape[0]
        self.w = shape[1]
        self.config=config
        # self.tfr_name = outname
        # print 'Loading the listone'
        # self.all_imgs = np.load('/u/big/trainingdata/KITTI_RAW/raw_db.npy')

    # nicely split file list in list of list by sequence
    def make_sequence_index(self):
        lst = [line.rstrip() for line in open(self.config.png_list)]
        seq = [line.rstrip() for line in open(self.config.seq_list)]
        seq_ll_ff = [[ff for ff in lst if ss in ff] for ss in seq]
        with open(self.config.seq_index, "wb") as f:
            pickle.dump(seq_ll_ff, f)

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def build_tfrecord(self):
        tmp = np.zeros((self.h, self.w, 6))

        with open(self.config.seq_index, "rb") as f:
            seq_ll_ff = pickle.load(f)

        print 'Building the tfrecord', len(seq_ll_ff), 'sequences'

        list_of_i1i2=[]
        # for each sequence
        for i in range(len(seq_ll_ff)):
            print 'Processing sequence', i, 'of lenght',len(seq_ll_ff[i])
            # for each image in the sequence, build a pair
            for j in range(0, len(seq_ll_ff[i])-1, 1):
                i1fn=seq_ll_ff[i][j]
                i2fn=seq_ll_ff[i][j+1]
                list_of_i1i2.append([i1fn, i2fn])

        print 'Fetched', len(list_of_i1i2), 'i1/i2 couples'

        writer = tf.python_io.TFRecordWriter(self.config.output_db)

        indexes = np.arange(len(list_of_i1i2))
        if(self.config.shuffle):
            np.random.shuffle(indexes)
        for i in indexes:
             i1fn=list_of_i1i2[i][0]
             i2fn=list_of_i1i2[i][1]
             print 'couple', i
             print 'i1', i1fn
             print 'i2', i2fn
             if self.config.colorspace == 'yuv':
                 print 'yuv'
		 im1 = io.imread(i1fn)
		 print im1.shape
                 tmp[...,0:3] = np_rgb2yuv(io.imread(i1fn)[0:self.h,0:self.w])
                 tmp[...,3:6] = np_rgb2yuv(io.imread(i2fn)[0:self.h,0:self.w])
             else:
                 tmp[...,0:3] = io.imread(i1fn)[0:self.h,0:self.w]
                 tmp[...,3:6] = io.imread(i2fn)[0:self.h,0:self.w]
             raw = tmp.astype(np.uint8).tostring()
             example = tf.train.Example(features=tf.train.Features(feature={'images_raw': self._bytes_feature(raw)}))
             writer.write(example.SerializeToString())
        writer.close()

    def build_tfrecord_sintel(self):
        orig_h = 436
        orig_w = 1024
        h = 128
        w = 256
        root = 'Sintel/test/'
        img_folder = 'final'
        putgt = False
        if putgt:
            flow_folder = 'flow'
        # Prepare the images list
        imgs = []
        with open('Sintel/full_movie.lst') as f:
            tmp = []
            cur_seq = '000'
            for line in f:
                if line.split('/')[2] == cur_seq:
                    tmp.append(line.rstrip())
                else:
                    imgs.append(copy.deepcopy(tmp))
                    tmp = []
                    cur_seq = line.split('/')[2]
        img_pairs = []
        for s in imgs:
            print s
            for fr_id in range(0, len(s)-1, 2):
                img_pairs.append((s[fr_id], s[fr_id+1]))
        writer = tf.python_io.TFRecordWriter(self.config.output_db)
        indexes = np.arange(len(img_pairs))
        for repeat in range(10):
            if(self.config.shuffle):
                np.random.shuffle(indexes)
            tmp = np.zeros((orig_h, orig_w, 6))
            for i in indexes:
                i1fn=img_pairs[i][0]
                i2fn=img_pairs[i][1]
                print 'couple', i
                print 'i1', i1fn
                print 'i2', i2fn
                if self.config.colorspace=='yuv':
                    tmp[...,0:3] = np_rgb2yuv(io.imread(i1fn))
                    tmp[...,3:6] = np_rgb2yuv(io.imread(i2fn))
                else:
                    tmp[...,0:3] = io.imread(i1fn)
                    tmp[...,3:6] = io.imread(i2fn)

                raw_imgs = tmp.astype(np.uint8).tostring()

                if putgt:
                    name = join(root, flow_folder, i1fn.split('/')[7], i1fn.split('/')[8][:-4]+'.flo')
                    f = open(name, 'rb')

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

                    raw_of = fmap.astype(np.float32).tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={'images_raw': self._bytes_feature(raw_imgs), 'flow_raw': self._bytes_feature(raw_of)}))
                else:
                    example = tf.train.Example(features=tf.train.Features(feature={'images_raw': self._bytes_feature(raw_imgs)}))

                writer.write(example.SerializeToString())
        writer.close()

    def build_tfrecord_virtualkitti(self):
        orig_h = 375
        orig_w = 1242
        h = 128
        w = 384
        root = 'VirtualKitti'
        img_folder = 'vkitti_1.3.1_rgb'
        putgt = False
        if putgt:
            flow_folder = 'flow'
        # Prepare the images list
        imgs = []
        with open('VirtualKitti/imgs.lst') as f:
            tmp = []
            cur_seq = '0001'
            for line in f:
                print line
                if line.split('/')[2] == cur_seq:
                    tmp.append(line.rstrip())
                else:
                    imgs.append(copy.deepcopy(tmp))
                    tmp = []
                    cur_seq = line.split('/')[2]
        img_pairs = []
        for s in imgs:
            for fr_id in range(0, len(s)-1, 2):
                img_pairs.append((s[fr_id], s[fr_id+1]))
        print img_pairs
        writer = tf.python_io.TFRecordWriter(self.config.output_db)
        indexes = np.arange(len(img_pairs))
        for repeat in range(10):
            if(self.config.shuffle):
                np.random.shuffle(indexes)
            tmp = np.zeros((orig_h, orig_w, 6))
            for i in indexes:
                i1fn=img_pairs[i][0]
                i2fn=img_pairs[i][1]
                print 'couple', i
                print 'i1', i1fn
                print 'i2', i2fn
                if self.config.colorspace=='yuv':
                    tmp[...,0:3] = np_rgb2yuv(io.imread(i1fn))
                    tmp[...,3:6] = np_rgb2yuv(io.imread(i2fn))
                else:
                    tmp[...,0:3] = io.imread(i1fn)
                    tmp[...,3:6] = io.imread(i2fn)

                raw_imgs = tmp.astype(np.uint8).tostring()

                if putgt:
                    name = join(root, flow_folder, i1fn.split('/')[7], i1fn.split('/')[8][:-4]+'.flo')
                    f = open(name, 'rb')

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

                    raw_of = fmap.astype(np.float32).tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={'images_raw': self._bytes_feature(raw_imgs), 'flow_raw': self._bytes_feature(raw_of)}))
                else:
                    example = tf.train.Example(features=tf.train.Features(feature={'images_raw': self._bytes_feature(raw_imgs)}))

                writer.write(example.SerializeToString())
        writer.close()

    def build_tfrecord_kitti2015(self):
        orig_h = 370
        orig_w = 1224
        h = 128
        w = 384
        root = 'KITTI_2015'
        img_folder = 'asdada'
        putgt = False
        if putgt:
            flow_folder = 'flow'
        # Prepare the images list
        imgs = []
        with open('KITTI_2015/imgs.lst') as f:
            tmp = []
            cur_seq = 'image_2'
            for line in f:
                print line.split('/')[2]
                if line.split('/')[2] == cur_seq:
                    tmp.append(line.rstrip())
                else:
                    imgs.append(copy.deepcopy(tmp))
                    tmp = []
                    cur_seq = line.split('/')[2]
        imgs.append(copy.deepcopy(tmp))
        print imgs
        img_pairs = []
        for s in imgs:
            print s
            for fr_id in range(0, len(s)-1, 2):
                img_pairs.append((s[fr_id], s[fr_id+1]))
        print img_pairs
        writer = tf.python_io.TFRecordWriter(self.config.output_db)
        indexes = np.arange(len(img_pairs))
        for repeat in range(100):
            if(self.config.shuffle):
                np.random.shuffle(indexes)
            tmp = np.zeros((orig_h, orig_w, 6))
            for i in indexes:
                i1fn=img_pairs[i][0]
                i2fn=img_pairs[i][1]
                print 'couple', i
                print 'i1', i1fn
                print 'i2', i2fn
                if self.config.colorspace=='yuv':
                    tmp[...,0:3] = np_rgb2yuv(io.imread(i1fn)[0:self.h,0:self.w])
                    tmp[...,3:6] = np_rgb2yuv(io.imread(i2fn)[0:self.h,0:self.w])
                else:
                    tmp[...,0:3] = io.imread(i1fn)
                    tmp[...,3:6] = io.imread(i2fn)

                raw_imgs = tmp.astype(np.uint8).tostring()

                if putgt:
                    name = join(root, flow_folder, i1fn.split('/')[7], i1fn.split('/')[8][:-4]+'.flo')
                    f = open(name, 'rb')

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

                    raw_of = fmap.astype(np.float32).tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={'images_raw': self._bytes_feature(raw_imgs), 'flow_raw': self._bytes_feature(raw_of)}))
                else:
                    example = tf.train.Example(features=tf.train.Features(feature={'images_raw': self._bytes_feature(raw_imgs)}))

                writer.write(example.SerializeToString())
        writer.close()

    def build_tfrecord_dreyeve(self):
        orig_h = 1080/2
        orig_w = 1920/2
        h = 135
        w = 240
        root = '/WindowsShares/MAJINBU/DREYEVE/DATA'
        img_folder = 'frames'
        putgt = False
        if putgt:
            flow_folder = 'flow'
        # Prepare the images list
        imgs = []
        with open(join(root, 'train.lst')) as f:
            tmp = []
            cur_seq = '04'
            for line in f:
                if line.split('/')[5] == cur_seq:
                    tmp.append(line.rstrip())
                else:
                    imgs.append(copy.deepcopy(tmp))
                    tmp = []
                    cur_seq = line.split('/')[5]
        imgs.append(copy.deepcopy(tmp))
        img_pairs = []
        for s in imgs:
            for fr_id in range(0, len(s)-1, 2):
                img_pairs.append((s[fr_id], s[fr_id+1]))
        writer = tf.python_io.TFRecordWriter(self.config.output_db)
        indexes = np.arange(len(img_pairs))
        #for repeat in range():
        if True:
            if(self.config.shuffle):
                np.random.shuffle(indexes)
            tmp = np.zeros((orig_h, orig_w, 6))
            for i in indexes:
                i1fn=img_pairs[i][0]
                i2fn=img_pairs[i][1]
                print 'couple', i
                print 'i1', i1fn
                print 'i2', i2fn
                if self.config.colorspace=='yuv':
                    tmp[...,0:3] = np_rgb2yuv(resize(io.imread(i1fn),[self.h,self.w], preserve_range=True))
                    tmp[...,3:6] = np_rgb2yuv(resize(io.imread(i2fn),[self.h,self.w], preserve_range=True))
                else:
                    tmp[...,0:3] = io.imread(i1fn)
                    tmp[...,3:6] = io.imread(i2fn)
                print tmp
                raw_imgs = tmp.astype(np.uint8).tostring()

                if putgt:
                    name = join(root, flow_folder, i1fn.split('/')[7], i1fn.split('/')[8][:-4]+'.flo')
                    f = open(name, 'rb')

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

                    raw_of = fmap.astype(np.float32).tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={'images_raw': self._bytes_feature(raw_imgs), 'flow_raw': self._bytes_feature(raw_of)}))
                else:
                    example = tf.train.Example(features=tf.train.Features(feature={'images_raw': self._bytes_feature(raw_imgs)}))

                writer.write(example.SerializeToString())
        writer.close()
    def build_tfrecord_davis(self):
        orig_h = 480
        orig_w = 854
        h = 240
        w = 427
        root = '/majinbu/public/DAVIS/JPEGImages/'
        img_folder = '480p'
        putgt = False
        if putgt:
            flow_folder = 'flow'
        # Prepare the images list
        imgs = []
        with open(join(root, 'train.lst')) as f:
            tmp = []
            cur_seq = 'bear'
            for line in f:
                if line.split('/')[6] == cur_seq:
                    tmp.append(line.rstrip())
                else:
                    imgs.append(copy.deepcopy(tmp))
                    tmp = []
                    cur_seq = line.split('/')[6]
        imgs.append(copy.deepcopy(tmp))
        img_pairs = []
        for s in imgs:
            for fr_id in range(0, len(s)-1, 2):
                img_pairs.append((s[fr_id], s[fr_id+1]))
        writer = tf.python_io.TFRecordWriter(self.config.output_db)
        indexes = np.arange(len(img_pairs))
        #for repeat in range():
        if True:
            if(self.config.shuffle):
                np.random.shuffle(indexes)
            tmp = np.zeros((orig_h, orig_w, 6))
            for i in indexes:
                i1fn=img_pairs[i][0]
                i2fn=img_pairs[i][1]
                print 'couple', i
                print 'i1', i1fn
                print 'i2', i2fn
                if self.config.colorspace=='yuv':
                    tmp[...,0:3] = np_rgb2yuv(resize(io.imread(i1fn),[orig_h,orig_w], preserve_range=True))
                    tmp[...,3:6] = np_rgb2yuv(resize(io.imread(i2fn),[orig_h,orig_w], preserve_range=True))
                else:
                    tmp[...,0:3] = io.imread(i1fn)
                    tmp[...,3:6] = io.imread(i2fn)
                raw_imgs = tmp.astype(np.uint8).tostring()

                if putgt:
                    name = join(root, flow_folder, i1fn.split('/')[7], i1fn.split('/')[8][:-4]+'.flo')
                    f = open(name, 'rb')

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

                    raw_of = fmap.astype(np.float32).tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={'images_raw': self._bytes_feature(raw_imgs), 'flow_raw': self._bytes_feature(raw_of)}))
                else:
                    example = tf.train.Example(features=tf.train.Features(feature={'images_raw': self._bytes_feature(raw_imgs)}))

                writer.write(example.SerializeToString())
        writer.close()
def main(_):
    pp.pprint(flags.FLAGS.__flags)
    writer = tfrWriter(flags.FLAGS)
    writer.make_sequence_index()
    if flags.FLAGS.dbname == 'Sintel':
        writer.build_tfrecord_sintel()
    elif flags.FLAGS.dbname == 'VirtualKitti':
        writer.build_tfrecord_virtualkitti()
    elif flags.FLAGS.dbname == 'KITTI_2015':
        writer.build_tfrecord_kitti2015()
    elif flags.FLAGS.dbname == 'dreyeve':
        writer.build_tfrecord_dreyeve()
    elif flags.FLAGS.dbname == 'DAVIS':
        writer.build_tfrecord_davis()
    else:
        writer.build_tfrecord()

if __name__ == '__main__':
    tf.app.run()

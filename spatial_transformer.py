# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf

def meshgrid(height, width, flatten=True):
    with tf.variable_scope('meshgrid'):
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')

        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.range(0, width, dtype='float32'), 1), [1, 0])) * 2.0 / width_f - 1.0
        y_t = tf.matmul(tf.expand_dims(tf.range(0, height, dtype='float32'), 1),
                                        tf.ones(shape=tf.stack([1, width]))) * 2.0 / height_f - 1.0

        if not flatten:
                                        grid = tf.stack([x_t, y_t], axis=2)
                                        return grid

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))
        ones = tf.ones_like(x_t_flat)
        grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
        return grid


def transformer(U, theta=None, flow=None, out_size=None, name='SpatialTransformer', mode='Affine2D', **kwargs):

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output


    def _transform(inp_img, out_size, theta=None, flow=None):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(inp_img)[0]
            height = tf.shape(inp_img)[1]
            width = tf.shape(inp_img)[2]
            num_channels = tf.shape(inp_img)[3]

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]

            grid = meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

            x_t = tf.slice(grid, [0, 0, 0], [-1, 1, -1])
            y_t = tf.slice(grid, [0, 1, 0], [-1, 1, -1])

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            if(theta is not None):
                dim = 2 if mode == 'Affine2D' else 3 # 'Projective2D'
                theta = tf.reshape(theta, (-1, dim, 3))
                theta = tf.cast(theta, 'float32')
                T_g = tf.matmul(theta, grid)
                x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
                y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
                if(mode=='Projective2D'):
                    print("%% setting up Projective2D")
                    w_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])
                    # w_s = tf.clip_by_value(w_s, 1e-3, 1e+3)
                    w_s = tf.sign(w_s) * tf.clip_by_value(tf.abs(w_s), 1e-3, 1e+3) # clip-pn -- clip pos/neg values
                    # w_s = tf.sign(w_s) * (tf.abs(w_s) + 1e-3)
                    x_s = x_s / w_s
                    y_s = y_s / w_s
            elif(flow is not None):
                print("%% setting up direct flow transformer")
                # relative flow w/ rebasing
                x_s = tf.reshape(flow[...,0], [num_batch, 1, -1]) * 2.0 + x_t
                y_s = tf.reshape(flow[...,1], [num_batch, 1, -1]) * 2.0 + y_t

            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                inp_img, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))

            # infer flow
            if(theta is not None):
                x = (x_s - x_t) / 2.0
                y = (y_s - y_t) / 2.0
                # x = -x * width_f
                # y = -y * height_f
                flow = tf.concat( axis=3, values=[tf.reshape( x, [num_batch, height, width, 1]),
                                      tf.reshape( y, [num_batch, height, width, 1])] )
                return output, flow
            else:
                return output


    with tf.variable_scope(name):
        if(mode=='Flow'):
            output = _transform(flow=flow, inp_img=U, out_size=out_size)
        else:
            output = _transform(theta=theta, inp_img=U, out_size=out_size)
        return output


def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer'):
    """Batch Spatial Transformer Layer
    Parameters
    ----------
    U : float
        tensor of inputs [num_batch,height,width,num_channels]
    thetas : float
        a set of transformations for each input [num_batch,num_transforms,6]
    out_size : int
        the size of the output [out_height,out_width]
    Returns: float
        Tensor of size [num_batch*num_transforms,out_height,out_width,num_channels]
    """
    with tf.variable_scope(name):
        num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
        indices = [[i]*num_transforms for i in xrange(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
        return transformer(input_repeated, thetas, out_size)

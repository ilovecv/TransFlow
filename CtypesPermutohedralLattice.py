import numpy as np
from ctypes import *
import ctypes.util
import cv2

from concurrent.futures import ThreadPoolExecutor, wait

permdll = ctypes.CDLL('./ctypes_shared_lib/permuto.so', mode=ctypes.RTLD_LOCAL)

c_constr = permdll.PermBuild
c_constr.restype = c_void_p

c_init = permdll.PermInit
c_init.argtypes = [c_void_p, POINTER(c_float), c_int, c_int]

c_compute = permdll.PermCompute
c_compute.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float), c_int, c_int]

c_delete = permdll.PermDelete
c_delete.argtypes = [c_void_p]


class CtypesPermutohedralLattice(object):
    def __init__(self, features):
        self.obj = c_constr()

        cf, h, w = features.shape

        features = np.reshape(features, (cf, h*w))
        features = np.ascontiguousarray(features, dtype='float32')

        f_p = features.ctypes.data_as(POINTER(c_float))
        c_init(self.obj, f_p, cf, h*w)

        self.norm_factor = self.compute(np.ones((1, h, w)), normalize=False)
        print np.min( self.norm_factor), np.max( self.norm_factor)

    def compute(self, x, normalize=True):
        cx, h, w = x.shape

        x = np.reshape(x, (cx, h*w))
        x = np.ascontiguousarray(x, dtype='float32')

        out = np.ones_like(x)
        out = np.ascontiguousarray(out, dtype='float32')

        x_p = x.ctypes.data_as(POINTER(c_float))
        out_p = out.ctypes.data_as(POINTER(c_float))

        c_compute(self.obj, x_p, out_p, cx, h*w)

        out = np.reshape(out, (cx, h, w))

        if normalize:
            out /= self.norm_factor

        return out

    def compute_stacked(self, x_list, normalize=True):
        c_vec = np.array([x.shape[0] for x in x_list])

        stacked_tensor = x_list[0]
        for i in range(1, len(x_list)):
            stacked_tensor = np.concatenate([stacked_tensor, x_list[i]], axis=0)

        stacked_tensor_filtered = self.compute(stacked_tensor, normalize)

        output_list = []
        c_counter = 0
        for i in range(0, c_vec.shape[0]):
            output_list.append(stacked_tensor_filtered[c_counter:c_counter + c_vec[i]])
            c_counter += c_vec[i]

        return output_list

    def __del__(self):
        c_delete(self.obj)



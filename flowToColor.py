import numpy as np
import scipy
#import np

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def computeColor(u, v):
    img = np.zeros((u.shape[0], u.shape[1], 3))
    #nanIdx = np.logical_or(np.isnan(u), np.isnan(v))
    #u[int(nanIdx)-1] = 0.
    #v[int(nanIdx)-1] = 0.
    colorwheel, ncols = makeColorwheel()
    rad = np.sqrt((u**2.+v**2.))
    a = np.arctan2((-v), (-u))/ np.pi
    fk = np.dot((a+1.)/2., ncols-1.)
    #% -1~1 maped to 1~ncols
    k0 = np.floor(fk).astype(np.int32)
    #% 1, 2, ..., ncols
    k1 = k0+1
    k1[k1 == ncols] = 1
    f = fk-k0

    #print k0

    for i in np.arange(colorwheel.shape[-1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0]/255.
        col1 = tmp[k1]/255.
        col = (1.-f)*col0+f*col1
        idx = rad<=1.
        col[idx] = 1.-rad[idx]*(1.-col[idx])
        #% increase saturation with radius
        col[rad>1] = col[rad>1]*0.75
        #% out of range
        img[:,:,i] = np.floor(255.*col)
    return img

def makeColorwheel():

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY+YG+GC+CB+BM+MR
    colorwheel = np.zeros((int(ncols), 3))
    #% r g b
    col = 0
    #%RY
    colorwheel[0:RY,0] = 255.
    colorwheel[0:RY,1] = np.floor(255.*np.arange(0., RY)/RY)
    col = col+RY
    #%YG
    colorwheel[col:col+YG,0] = 255.-np.floor(255.*np.arange(0.,YG)/YG)
    colorwheel[col:col+YG,1] = 255.
    col = col+YG
    #%GC
    colorwheel[col+0:col+GC,1] = 255.
    colorwheel[col+0:col+GC,2] = np.floor(255.*np.arange(0., GC)/GC)
    col = col+GC
    #%CB
    colorwheel[col+0:col+CB,1] = 255.-np.floor(255.*np.arange(0., CB)/CB)
    colorwheel[col+0:col+CB,2] = 255.
    col = col+CB
    #%BM
    colorwheel[col+0:col+BM,2] = 255.
    colorwheel[col+0:col+BM,0] = np.floor(255.*np.arange(0., BM)/BM)
    col = col+BM
    #%MR
    colorwheel[col+0:col+MR,2] = 255.-np.floor(255.*np.arange(0., MR)/MR)
    colorwheel[col+0:col+MR,0] = 255.
    return colorwheel, ncols

def flowToColor(flow, varargin=None):
    # TODO: cleanup all the translator crap
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10
    [height, widht, nBands] = flow.shape
    if nBands != 2.:
        np.error('flowToColor: image must have two bands')


    u = flow[:,:,0]
    v = flow[:,:,1]
    #print u.shape,v.shape
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1.
    #% fix unknown flow
    #idxUnknown = np.logical_or(np.abs(u) > UNKNOWN_FLOW_THRESH, np.abs(v) > UNKNOWN_FLOW_THRESH)
    #print np.array(idxUnknown)
    #u[int(idxUnknown)-1] = 0.
    #v[int(idxUnknown)-1] = 0.
    maxu = max(maxu, np.max(u))
    minu = max(minu, np.max(u))
    maxv = max(maxv, np.max(v))
    minv = max(minv, np.max(v))
    rad = np.sqrt((u**2.+v**2.))
    maxrad = max(maxrad, np.max(rad))
    #print 'max flow:',maxrad, ' flow range: u =', minu, maxu, 'v =', minv, maxv
    #if isempty(varargin) == 0.:
    #    maxFlow = varargin.cell[0]
    #    if maxFlow > 0.:
    #        maxrad = maxFlow
    u = u/( maxrad+1e-5)
    v = v/( maxrad+1e-5)
    #% compute color
    img = computeColor(u, v)
    #% unknown flow
    #IDX = np.repmat(idxUnknown, np.array(np.hstack((1., 1., 3.))))
    #img[int(IDX)-1] = 0.
    return img/255.

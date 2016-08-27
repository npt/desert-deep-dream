# Based on dreamify.py from image-dreamer by Dhar (Gary Arnold)
# https://github.com/Dhar/image-dreamer

from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format

import caffe

import sys
import time

def load_net(name, bgr=False):
    model_path = 'models/%s/' % name
    net_fn = model_path + 'deploy.prototxt'
    param_fn = model_path + 'model.caffemodel'

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    kwargs = {}
    if not bgr: # the reference model has channels in BGR order instead of RGB
        kwargs['channel_swap'] = (2,1,0)
    return caffe.Classifier('tmp.prototxt', param_fn,
                           mean = np.float32([104.0, 116.0, 122.0]),
                           # ImageNet mean, training set dependent
                           **kwargs)

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32,
              clip=True):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)
    # apply jitter shift

    net.forward(end=end)
    dst.diff[:] = dst.data  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)
    # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4,
              end='inception_4c/output', clip=True, callback=None,
              **step_params):
    starttime = time.time()

    h0, w0 = base_img.shape[:2]

    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1],
                               (1, 1.0/octave_scale,1.0/octave_scale),
                               order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1])
    # allocate image for network-produced details

    progress = 0
    total_pixel_factor = sum(octave_scale ** (-2*i) for i in xrange(octave_n))

    for octave, octave_base in enumerate(octaves[::-1]):
        octave_pixel_factor = octave_scale ** (-2 * (octave_n - octave - 1))
        progress_incr = octave_pixel_factor / total_pixel_factor / iter_n

        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            stepstarttime = time.time()
            make_step(net, end=end, clip=clip, **step_params)

            # visualization
            #vis = deprocess(net, src.data[0])
            #if not clip: # adjust image contrast if clipping is disabled
            #    vis = vis*(255.0/np.percentile(vis, 99.98))

            detail = src.data[0] - octave_base
            h1, w1 = detail.shape[-2:]
            scale_detail = nd.zoom(detail, (1, 1.0*h0/h1, 1.0*w0/w1), order=1)
            vis = deprocess(net, scale_detail + octaves[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))

            progress += progress_incr
            if callback:
                callback(progress=progress, image=vis)

            #showarray(vis)
            stependtime = time.time()
            print octave, i, end, detail.shape, stependtime-stepstarttime, \
                  progress

        # extract details produced on the current octave
        #detail = src.data[0]-octave_base
    # returning the resulting image
    rv = deprocess(net, src.data[0])

    endtime = time.time()
    print
    print endtime-starttime
    print

    return rv

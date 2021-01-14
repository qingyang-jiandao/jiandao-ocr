import sys
import numpy as np
import os
import time
caffe_root = './caffe_t/python'
sys.path.append(caffe_root)
import caffe
from caffe.proto import caffe_pb2

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = './model/stnpt-fnn-vgg-stnfeature-xyxy2032_iter_600000.caffemodel'

# init
caffe.set_mode_gpu()
caffe.set_device(1)

solver_param = caffe_pb2.SolverParameter()
# solver = caffe.SGDSolver('models/darknet_yolov3/solver.prototxt')
solver = caffe.AdamSolver('solver.prototxt')
solver.net.copy_from(weights)

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

timer = Timer()
while solver.iter < 600000:
    # Make one SGD update
    timer.tic()
    solver.step(1)
    timer.toc()

    # filename = (solver_param.snapshot_prefix +
    #             '_iter_{:d}'.format(solver.iter) + '.caffemodel')
    # filename = os.path.join(output_dir, filename)
    # solver.net.save(str(filename))

    # if solver.iter % (100) == 0:
    #     print('speed: {:.3f}s / iter'.format(timer.average_time))

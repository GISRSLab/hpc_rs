import pycuda.driver as drv
from numba import cuda

 
def query_device():
    drv.init()
    print("%d device(s) found." % drv.Device.count())
    for ordinal in range(drv.Device.count()):
        dev = drv.Device(ordinal)
        print('Device #%d: %s' % (ordinal, dev.name()))
        print(' Compute Capability: %d.%d' % dev.compute_capability())
        print(' Total Memory: %s KB' % (dev.total_memory()//(1024)))
    
        atts = [ (str(att), value) for att, value in list(dev.get_attributes().items())]
        atts.sort()
    
        for att, value in atts:
            print(' %s : %s' % (att, value))

@cuda.jit
def gpu_mulply(a, b, result):
    x, y = cuda.grid(2);

    if x < result.shape[0] and y < result.shape[1]:
        for k in range(a.shape[1]):
            result[x, y] += a[x, k] * b[k, y];

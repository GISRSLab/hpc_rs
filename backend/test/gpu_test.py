from numba import cuda
import numpy as np

@cuda.jit
def test(output_data:cuda.device_array, row:int, col:int):
    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x;
    j = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y;

    if j < col and i < row:
        output_data[i, j] = j;

if __name__ == "__main__":
    device_data = cuda.device_array((3,10), dtype=int)
    test[(1,1),(3,10)](device_data, 3,10)
    print(device_data.copy_to_host())


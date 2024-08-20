import numpy as np
from numba import cuda


@cuda.jit
def mean_filter(
    data: cuda.device_array,
    outData: cuda.device_array,
    row: int,
    col: int,
    half_cell_size: int,
):
    """邻域均值滤波
    gpu 版本
    Returns:
        输出栅格: 经过均值滤波后的栅格数据
    """
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if j < row and i < col:
        # 控制线程边界

        # 确定边界
        if j - half_cell_size < 0:
            up_bound = 0
        else:
            up_bound = j - half_cell_size
        if j + half_cell_size > row - 1:
            down_bound = row
        else:
            down_bound = j + half_cell_size + 1
        if i - half_cell_size < 0:
            left_bound = 0
        else:
            left_bound = i - half_cell_size
        if i + half_cell_size > col - 1:
            right_bound = col
        else:
            right_bound = i + half_cell_size + 1

        # # 确定边界
        # up_bound = max(0, i - half_cell_size)
        # down_bound = min(row, i + half_cell_size + 1)
        # left_bound = max(0, j - half_cell_size)
        # right_bound = min(col, j + half_cell_size + 1)

        # 计算邻域均值
        sum_value = 0.0
        count = 0
        for k in range(up_bound, down_bound):
            for l in range(left_bound, right_bound):
                sum_value += data[k, l]
                count += 1

        # 存储均值到输出数组
        outData[j, i] = sum_value / count


def main(raster: np.ndarray) -> np.ndarray:
    """
    应用均值滤波器到输入影像,计算失败返回空数组
    :param raster: 输入的多波段影像，存储在numpy的三维数组中
    :return: 应用均值滤波后的影像，存储在numpy的三维数组中
    """
    try:
        if raster.ndim != 3:
            raise ValueError("输入数组必须是三维数组")

        # 获取影像的尺寸
        num_bands, image_height, image_width = raster.shape

        # 定义均值滤影像的尺寸
        half_size = 1

        # 计算每个线程块的尺寸
        threads_per_block = (16, 16)
        blocks_per_grid_x = int(np.ceil(image_height / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil(image_width / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # 创建一个结果数组
        result = np.zeros_like(raster)

        # 遍历进行滤波计算
        for band in range(num_bands):
            # 初始化图像和结果数组
            input_image = raster[band, :, :]
            output_image = np.zeros_like(input_image)

            # 将图像和结果数组传递到GPU
            d_image = cuda.to_device(input_image)
            d_result = cuda.to_device(output_image)

            # 调用CUDA内核
            mean_filter[blocks_per_grid, threads_per_block](d_image, d_result, image_height, image_width, half_size)

            # 将结果传回cpu
            result_cpu = d_result.copy_to_host()

            # 将计算值传入输出值中
            result[band, :, :] = result_cpu

        return result

    except cuda.CudaSupportError as e:
        print(f"CUDA不支持: {e}")
    except MemoryError as e:
        print(f"内存错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")
    return np.array([])


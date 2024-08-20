import numpy as np
from numba import cuda


@cuda.jit
def gauss_filter(
    data: cuda.device_array,
    out_data: cuda.device_array,
    kernel: cuda.device_array,
    row: int,
    col: int,
    cell_size: int,
    half_cell_size: int
):
    """邻域均值滤波
    gpu 版本
    Returns:
        输出栅格: 经过均值滤波后的栅格数据
    """

    i, j = cuda.grid(2)

    if i < row and j < col:
        # 控制线程边界

        # 确定边界
        up_bound = max(0, i - half_cell_size)
        down_bound = min(row, i + half_cell_size + 1)
        left_bound = max(0, j - half_cell_size)
        right_bound = min(col, j + half_cell_size + 1)

        # 计算高斯卷积
        sum_value = 0.0
        kernel_sum = 0.0
        for k in range(up_bound, down_bound):
            for m in range(left_bound, right_bound):
                # 计算高斯核对应的索引
                kernel_i = k - i + half_cell_size
                kernel_j = m - j + half_cell_size
                if 0 <= kernel_i < cell_size and 0 <= kernel_j < cell_size:
                    weight = kernel[kernel_i, kernel_j]
                    sum_value += data[k, m] * weight
                    kernel_sum += weight

        # 存储卷积值到输出数组
        if kernel_sum > 0:
            out_data[i, j] = sum_value / kernel_sum
        else:
            out_data[i, j] = data[i, j]


def gaussian_kernel(size, sigma=1):
    """生成高斯核"""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)


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
        cell_size = 3
        half_cell_size = cell_size // 2

        # 创建卷积核
        kernel = gaussian_kernel(cell_size)

        # 计算每个线程块的尺寸
        threads_per_block = (16, 16)
        blocks_per_grid_x = int(np.ceil(image_height / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil(image_width / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # 创建一个结果数组
        result = np.zeros_like(raster)

        # 创建cuda stream
        stream_list = list()
        for u in range(num_bands):
            stream = cuda.stream()
            stream_list.append(stream)

        # 遍历进行滤波计算
        for band in range(num_bands):
            # 初始化图像和结果数组
            input_image = raster[band, :, :]
            output_image = np.zeros_like(input_image)

            # 将图像和结果数组传递到GPU
            d_kernel = cuda.to_device(kernel, stream=stream_list[band])
            d_image = cuda.to_device(input_image, stream=stream_list[band])
            d_result = cuda.to_device(output_image, stream=stream_list[band])

            # 调用CUDA内核
            (gauss_filter[blocks_per_grid, threads_per_block, stream_list[band]]
             (d_image, d_result, d_kernel, image_height, image_width, cell_size,half_cell_size))

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
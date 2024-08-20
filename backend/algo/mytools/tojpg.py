# -*- coding: UTF-8 -*-
import numpy as np
import cupy as cp
from PIL import Image
from osgeo import gdal

"""
读取影像并将其转换为jpg影像
"""


def readTif(original: str, bandsOrder=[4, 3, 2]) -> np.ndarray:
    """
    Args:
        original: 影像存储位置
        bandsOrder: 转jpg使用波段默认输入4, 3, 2

    Returns: 包含指定波段的二维numpy数组，如果读取失败返回空数组
    """
    try:
        gdal.DontUseExceptions()
        driver = gdal.GetDriverByName('GTiff')
        driver.Register()
        ds = gdal.Open(original, gdal.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError(f"无法打开遥感影像: {original}")

        cols = ds.RasterXSize
        rows = ds.RasterYSize
        data = np.empty([rows, cols, 3], dtype=float)
        num_bands = ds.RasterCount

        if num_bands < 3:
            raise ValueError(f"波段数为{num_bands}，过小")

        if num_bands == 3:
            bandsOrder = [3, 2, 1]

        for i in range(3):
            if bandsOrder[i] > num_bands or bandsOrder[i] <= 0:
                raise ValueError(f"波段索引 {bandsOrder[i]} 超出范围")

            band = ds.GetRasterBand(bandsOrder[i])
            data1 = band.ReadAsArray()
            if data1 is None:
                raise IOError(f"无法读取波段 {bandsOrder[i]}")
            data[:, :, i] = data1

        return data

    except FileNotFoundError as fnf_error:
        print(f"文件错误: {fnf_error}")
    except ValueError as val_error:
        print(f"值错误: {val_error}")
    except IOError as io_error:
        print(f"IO错误: {io_error}")
    except Exception as e:
        print(f"发生未知错误: {e}")
    return np.array([])


# 百分比拉伸
def stretchImg(imgPath, resultPath, lower_percent=0.6, higher_percent=99.4, block_size=1024):
    try:
        data = readTif(imgPath)
        if data.size == 0:
            print("数据读取失败，无法进行拉伸")
            return

        rows, cols, bands = data.shape
        out_gpu = cp.zeros_like(data, dtype=cp.uint8)
        list_max = []
        list_min = []

        for i in range(bands):
            for row in range(0, rows, block_size):
                row_end = min(row + block_size, rows)
                data_block = data[row:row_end, :, i]
                data_block = cp.array(data_block)
                c = cp.percentile(data_block, lower_percent)
                d = cp.percentile(data_block, higher_percent)
                list_min.extend(c[c != 0].tolist())
                list_max.extend(d[d != 0].tolist())

            if list_min:
                bottom = np.max(list_min)
            else:
                bottom = 0

            if list_max:
                top = np.max(list_max)
            else:
                raise ValueError("无法计算数据的最大值")
            for row in range(0, rows, block_size):
                row_end = min(row + block_size, rows)
                data_block = data[row:row_end, :, i]
                data_block = cp.array(data_block)
                t = (data_block - bottom) * 255 / (top - bottom)
                t[t < 0] = 0
                t[t > 255] = 255
                out_gpu[row:row_end, :, i] = t

            list_max = []
            list_min = []

        out = cp.asnumpy(out_gpu)
        outImg = Image.fromarray(np.uint8(out))
        out_img = outImg.resize((500, 500), resample=Image.LANCZOS)
        out_img.save(resultPath)

    except ValueError as val_error:
        print(f"值错误: {val_error}")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    gdal.DontUseExceptions()
    path = r"F:\学习资料\汤俊大三下\实习\data\rs01_noise_800.TIF"
    out_path = r"F:\学习资料\汤俊大三下\实习\data\03.jpg"
    stretchImg(path, out_path)

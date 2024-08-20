from osgeo import gdal
import numpy as np

gdal.AllRegister()

def open_raster(filepath: str) -> np.ndarray:
    """
    读取所有波段并将其写入数组

    :param filepath: 遥感影像文件的路径
    :return: 包含所有波段数据的三维Numpy数组。如果读取失败，返回一个空数组
    """
    try:
        # 打开影像并且创建空间
        dataset = gdal.Open(filepath)
        if dataset is None:
            raise FileNotFoundError(f"无法打开遥感影像: {filepath}")

        # 获取影像的宽度、高度以及波段数
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        num_bands = dataset.RasterCount
        # 定义一个空数组
        image_array = np.zeros((num_bands, height, width), dtype=np.float32)

        # 将每个波段填入数组
        for num in range(1, num_bands + 1):
            band = dataset.GetRasterBand(num)
            if band is None:
                raise ValueError(f"无法读取指定的波段: {num}")

            band_data = band.ReadAsArray()
            image_array[num - 1, :, :] = band_data

        return image_array

    except FileNotFoundError as fnf_error:
        print(f"文件错误: {fnf_error}")
    except ValueError as val_error:
        print(f"值错误: {val_error}")
    except Exception as e:
        print(f"发生未知错误: {e}")
    return np.array([])

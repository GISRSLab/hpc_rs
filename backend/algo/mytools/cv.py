# -*- coding: UTF-8 -*-
import numpy as np
from PIL import Image
from osgeo import gdal



def readTif(original: str, bandsOrder=[4, 3, 2]):
    """

    Args:
        original: 影像存储位置
        bandsOrder: 转jpg使用波段默认输入4, 3, 2

    Returns:包含指定波段的二维numpy数组，如果读取失败返回空数组

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
            raise ValueError(f"波段数为{num_bands}过小")
        if num_bands == 3:
            bandsOrder = [3, 2, 1]
        for i in range(3):
            band = ds.GetRasterBand(bandsOrder[i])
            data1 = band.ReadAsArray()
            data[:, :, i] = data1
        return data

    except FileNotFoundError as fnf_error:
        print(f"文件错误: {fnf_error}")
    except ValueError as val_error:
        print(f"值错误: {val_error}")
    except Exception as e:
        print(f"发生未知错误: {e}")
    return np.array([])

# 百分比拉伸
def stretchImg(imgPath, resultPath, lower_percent=0.6, higher_percent=99.4):
    try:
        data = readTif(imgPath)
        n = data.shape[2]
        out = np.zeros_like(data, dtype=np.uint8)
        for i in range(n):
            a = 0
            b = 255
            c = np.percentile(data[:, :, i], lower_percent)
            d = np.percentile(data[:, :, i], higher_percent)
            t = a + (data[:, :, i] - c) * (b - a) / (d - c)
            t[t < a] = a
            t[t > b] = b
            out[:, :, i] = t
        outImg = Image.fromarray(np.uint8(out))
        out_img = outImg.resize((500, 500), resample=Image.LANCZOS)
        out_img.save(resultPath)
    except Exception as e:
        print(f"cv.py-->{e}")
        return False
    return True


if __name__ == "__main__":
    gdal.DontUseExceptions()
    input_file = r"F:\学习资料\汤俊大三下\实习\数据\initial_data\data\rs01.TIF"
    output_file = r"F:\学习资料\汤俊大三下\实习\数据\initial_data\data\01.jpg"
    stretchImg(input_file, output_file)

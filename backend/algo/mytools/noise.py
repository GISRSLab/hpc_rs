from osgeo import gdal
from typing import Literal
import numpy as np


async def add_noise_to_tif(input_tif: str, output_tif: str, mean=0, std:Literal[200, 400, 600, 800]=400) -> bool:
    try:
        # 打开输入的 TIF 文件
        dataset = gdal.Open(input_tif)
        if dataset is None:
            raise FileNotFoundError(f"无法打开文件 {input_tif}")

        # 获取影像的元数据
        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.Create(
            output_tif,
            dataset.RasterXSize,
            dataset.RasterYSize,
            dataset.RasterCount,
            gdal.GDT_Float32
        )
        if output_dataset is None:
            raise IOError(f"无法创建文件 {output_tif}")

        # 复制投影和地理转换信息
        output_dataset.SetProjection(dataset.GetProjection())
        output_dataset.SetGeoTransform(dataset.GetGeoTransform())

        # 遍历每一个波段
        for band_index in range(1, dataset.RasterCount + 1):
            # 读取波段数据
            band = dataset.GetRasterBand(band_index)
            array = band.ReadAsArray()

            # 生成高斯噪声
            noise = np.random.normal(mean, std, array.shape)

            # 添加噪声到影像数据
            noisy_array = array + noise

            # 创建新的波段并写入带噪声的影像数据
            output_band = output_dataset.GetRasterBand(band_index)
            output_band.WriteArray(noisy_array)

        # 关闭数据集
        dataset = None
        output_dataset = None

        return True

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return False
    except IOError as e:
        print(f"IO错误: {e}")
        return False
    except Exception as e:
        print(f"发生错误: {e}")
        return False


from osgeo import gdal
import numpy as np


def write_raster(filepath: str, image_array: np.ndarray, refer_file: str) -> None:
    """
    将多波段数据写入影像文件,文件格式GeoTiff

    :param filepath: 输出影像文件的路径
    :param image_array: 包含多波段数据的三维Numpy数组
    :param refer_file: 参考影像，获取投影与地理变换参数
    """
    try:
        if image_array.ndim != 3:
            raise ValueError("输入数组必须是三维数组")

        # 获取影像的维度信息
        num_bands, height, width = image_array.shape

        # 创建驱动
        driver = gdal.GetDriverByName('GTiff')
        if driver is None:
            raise RuntimeError("无法获取GTiff驱动程序")

        # 创建一个新的影像文件
        dataset = driver.Create(filepath, width, height, num_bands, gdal.GDT_Float32)
        if dataset is None:
            raise RuntimeError(f"无法创建影像文件: {filepath}")

        # 获取地理变换参数与投影
        # 打开输入文件，以只读模式读取
        dataset_input = gdal.Open(refer_file, gdal.GA_ReadOnly)
        if dataset_input is None:
            raise FileNotFoundError(f"无法打开遥感影像: {refer_file}")

        # 获取输入文件的地理变换信息与投影
        geo_transform = dataset_input.GetGeoTransform()
        if geo_transform is None:
            raise ValueError("无法获取输入文件的地理变换信息")

        projection = dataset_input.GetProjection()  # 原影像坐标系
        if projection is None:
            raise ValueError("无法获取输入文件的投影信息")

        # 设置地理变换参数
        dataset.SetGeoTransform(geo_transform)

        # 设置投影信息（如果有）
        if projection is not None:
            dataset.SetProjection(projection)

        # 将每个波段的数据写入影像
        for i in range(num_bands):
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(image_array[i, :, :])

        # 关闭影像文件
        dataset.FlushCache()
        dataset = None

    except RuntimeError as run_error:
        print(f"运行时错误: {run_error}")
    except Exception as e:
        print(f"发生未知错误: {e}")

import numpy as np
from osgeo import gdal

gdal.AllRegister()

def read_img(filename): #读取影像
    dataset: gdal.Dataset = gdal.Open(filename)  # 打开文件
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data: np.ndarray = dataset.ReadAsArray(0, 0, im_width, im_height).astype(np.float32)  # 将数据写成数组，对应栅格矩阵

    return dataset,im_proj, im_geotrans, im_data, im_height, im_width #输出文件、坐标系、仿射矩阵、栅格数据、栅格的高和宽

def GetPointsData(ds,Colname): #获取点上的数据
    global df_values #设置df_values为全局变量，因而在每次使用时都需要确保df_values为所需要的数据
    #获取放射变换信息
    bands = ds.RasterCount #获取栅格的波段数
    transform = ds.GetGeoTransform() #获取栅格的仿射矩阵
    xOrigin = transform[0] #获取栅格左上角栅格的点位信息
    yOrigin = transform[3] #获取栅格左上角栅格的点位信息
    pixelWidth = transform[1] #获取栅格的宽度
    pixelHeight = transform[5] #获取栅格的高度
    for i in df_values.index: #对df_values的每个数据进行数据提取
        xOffset = int((df_values.loc[i,"Longitude"]-xOrigin)/pixelWidth) #获取点所在的位置
        yOffset = int((df_values.loc[i,"Latitude"]-yOrigin)/pixelHeight) #获取点所在的位置
        df_values.loc[i,"Tif_Xs"] = xOffset 
        df_values.loc[i,"Tif_Ys"] = yOffset
        for  k in range(bands): #分别求不同波段的数据
            band = ds.GetRasterBand(k+1) #获取k+1波段的数据，由于python是0开始，gdal是1开始，因而在python中使用gdal时需要加1
            data = band.ReadAsArray(xOffset, yOffset,1,1) #读取栅格位置的数据 #xsize,ysize,是栅格的列数和行数
            value = data[0,0] #data为二维的数据，需要提取出来
            if value != 1e+20: #CMIP6使用1e+20作为缺省值，因而对于非缺省值，应该保存他的数据
                df_values.loc[i,Colname] = value
            else:
                df_values.loc[i,Colname] = " " #如果是缺省值，直接填空格来占位
    del ds #删除内存占用栅格
    return df_values #返回df_values


if __name__ == '__main__':
    dataset,im_proj, im_geotrans, im_data, im_height, im_width = read_img('./data/dem06.img');
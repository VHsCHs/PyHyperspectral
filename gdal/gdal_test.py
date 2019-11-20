from osgeo import gdal

dataset = gdal.Open('D:\\Desktop\\gap\\RAW\\356.raw', gdal.GA_ReadOnly)
print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                             dataset.GetDriver().LongName))
print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                    dataset.RasterYSize,
                                    dataset.RasterCount))
print("Projection is {}".format(dataset.GetProjection()))
geotransform = dataset.GetGeoTransform()
if geotransform:
    print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
band = dataset.GetRasterBand(1)
print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

print("Description={}".format(dataset.GetDescription()))
print("RasterCount={}".format(dataset.RasterCount))

min = band.GetMinimum()
max = band.GetMaximum()
if not min or not max:
    (min, max) = band.ComputeRasterMinMax(True)
print("Min={:.3f}, Max={:.3f}".format(min, max))

if band.GetOverviewCount() > 0:
    print("Band has {} overviews".format(band.GetOverviewCount()))

if band.GetRasterColorTable():
    print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

scanline = band.ReadRaster(xoff=0, yoff=0,
                           xsize=band.XSize, ysize=band.YSize,
                           buf_xsize=band.XSize, buf_ysize=band.YSize,
                           buf_type=gdal.GDT_Float32)

import struct
tuple_of_floats = struct.unpack('f' * band.XSize * band.YSize, scanline)

print(tuple_of_floats)
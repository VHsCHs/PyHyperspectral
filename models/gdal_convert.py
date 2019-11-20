#!/usr/bin/env python
# -*- coding: utf-8 -*-

from osgeo import gdal, gdalconst
import numpy as np
import os


class ENVI_RAW():
    def __init__(self, filename):
        '''打开文件'''
        self.dataset = gdal.Open(filename, gdal.GA_ReadOnly)
        if self.dataset == None:
            print('数据无法打开')

        self.driver_name, self.driver_long_name = self.get_driver()
        self.Xsize, self.Ysize = self.get_size()
        self.band_type = self.get_bandtype()
        self.bands = self.get_bands()
        self.cols = self.dataset.RasterXSize
        self.rows = self.dataset.RasterYSize
        self.desc = self.get_desc()
        self._r_band = 73
        self._g_band = 46
        self._b_band = 15
        self._gray_band = 169
        self._r_scan = False
        self._g_scan = False
        self._b_scan = False
        self._gray_scan = False
        self.get_hdr()

    def get_hdr(self):
        self.info = self.get_info()
        self._wavelength = self.info['wavelength']
        self._samples = self.info['samples']
        self._lines = self.info['lines']
        self._bands = self.info['bands']
        self._headeroffset = self.info['headeroffset']
        self._filetype = self.info['filetype']
        self._datatype = self.info['datatype']
        self._interleave = self.info['interleave']
        self._wavelengthunits = self.info['wavelengthunits']
        self._binning = self.info['binning']

    def get_driver(self):
        return self.dataset.GetDriver().ShortName, self.dataset.GetDriver().LongName

    def get_size(self):
        return self.dataset.RasterXSize, self.dataset.RasterYSize

    def get_bands(self):
        return self.dataset.RasterCount

    def get_bandtype(self):
        band = self.dataset.GetRasterBand(1)
        return gdal.GetDataTypeName(band.DataType)

    def get_desc(self):
        return self.dataset.GetDescription()

    def scanband(self, band):
        band = self.dataset.GetRasterBand(band)
        return np.array(band.ReadAsArray(0, 0, band.XSize, band.YSize))

    def cut(self, band, xpoint, ypoint, xoffset, yoffset):
        band = self.dataset.GetRasterBand(band)
        return np.array(band.ReadAsArray(xpoint, ypoint, xoffset, yoffset))

    def output(self):
        output_array = []
        for num, bandnum in enumerate(range(self.bands)):
            band = self.dataset.GetRasterBand(bandnum + 1)
            scanband = band.ReadAsArray(0, 0, band.XSize, band.YSize)
            output_array.append(scanband)
        return np.array(output_array)

    def get_rgb(self, r_band=73, g_band=46, b_band=15):

        if r_band == self._r_band:
            if not self._r_scan:
                self.r_layer = self.scanband(self._r_band)
                self._r_scan = True
            else:
                pass
        else:
            self._r_band = r_band
            self.r_layer = self.scanband(r_band)

        if g_band == self._g_band:
            if not self._g_scan:
                self.g_layer = self.scanband(self._g_band)
                self._g_scan = True
            else:
                pass
        else:
            self._g_band = g_band
            self.g_layer = self.scanband(g_band)

        if b_band == self._b_band:
            if not self._b_scan:
                self.b_layer = self.scanband(self._b_band)
                self._b_scan = True
            else:
                pass
        else:
            self._b_band = b_band
            self.b_layer = self.scanband(b_band)
        rgb = np.hstack([self.r_layer.reshape(-1, 1), self.g_layer.reshape(-1, 1), self.b_layer.reshape(-1, 1)])
        rgb = rgb.reshape(self.Ysize, self.Xsize, 3)
        return rgb

    def get_gray(self, gray_band=169):
        if gray_band == self._gray_band:
            if not self._gray_scan:
                self.gray_layer = self.scanband(self._gray_band)
                self._gray_scan = True
            else:
                pass
        else:
            self._gray_band = gray_band
            self.gray_layer = self.scanband(gray_band)
        return self.gray_layer

    def get_info(self):
        hdr_file = os.path.splitext(self.desc)[0] + '.hdr'
        with open(hdr_file, 'r') as hdr_to_read:
            text_array = []
            curve_mark = False
            while True:
                text_line = hdr_to_read.readline()
                if text_line:
                    if curve_mark != True:
                        text_array.append(text_line.replace('\n', ''))
                    else:
                        text_array[-1] = text_array[-1] + text_line.replace('\n', '')
                    if '{' in text_line:
                        curve_mark = not curve_mark
                        if '}' in text_line:
                            curve_mark = not curve_mark
                    if curve_mark == True:
                        if '}' in text_line:
                            curve_mark = not curve_mark
                else:
                    break
        text_array = [x.replace(' ', '') for x in text_array if x.strip() != '']
        text_array = [x.replace('{', '[') for x in text_array if x.strip() != '']
        text_array = [x.replace('}', ']') for x in text_array if x.strip() != '']
        # text_array = [x.replace(':', '') for x in text_array if x.strip() != '']
        text_array = [x.replace('[[', '\'') for x in text_array if x.strip() != '']
        text_array = [x.replace(']]', '\'') for x in text_array if x.strip() != '']
        info = {}
        for i in text_array:
            # print(i)
            if i.find('=') != -1:
                if '[' in i:
                    info[i[0:i.find('=')]] = eval(i[i.find('=') + 1: len(i)])
                else:
                    info[i[0:i.find('=')]] = i[i.find('=') + 1: len(i)]
            else:
                info[i] = i
        return info

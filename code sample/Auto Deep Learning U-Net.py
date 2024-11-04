import numpy as np
from osgeo import gdal
import os
import rasterio
import time
import h5py
import rasterio
from rasterio import plot
from rasterio.mask import mask
from rasterio.plot import show
import numpy as np
from osgeo import gdal
from sklearn.metrics import r2_score,mean_absolute_percentage_error,mean_absolute_error,mean_squared_error
import os
from  builtins import any as b_any
import time
import h5py
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import initializers, activations

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as k
import tensorflow.keras as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, Dense, Dropout, Activation, Layer,Flatten, add, Add, multiply,GlobalAveragePooling2D, GlobalMaxPooling2D,\
    Reshape,Permute, Concatenate, Lambda,BatchNormalization,Dot,dot
from tensorflow.keras.models import load_model
from numpy.random import uniform
import random
import math
import time

modis_shapefile=input("please insert shapefile address for modis data eg 'F:/shapefile.shp': ")
ET_shapefile=input("please insert shapefile address for ET data eg 'F:/shapefile.shp': ")
address_src_modis=input("please insert modis data address eg 'F:/data/': ")
address_dest_modis=input("please insert destination for modis data eg 'F:/data/': ")
address_src_ET=input("please insert ET data address eg 'F:/data/': ")
address_dest_ET=input("please insert destination for ET data eg 'F:/data/': ")

adress_src=[address_src_modis,address_src_ET]
address_dest=[address_dest_modis,address_dest_ET]
shapefile_list=[modis_shapefile,ET_shapefile]
lb=[1,1,1,.2,.00001,1,1,1]
ul=[4.99,4.99,2.99,.5,.005,3.99,4.99,4.99]

Resolution=0.009
#resampling and cutting the images based on shapefile
for i in range(len(adress_src)):
    bandlist = [band for band in os.listdir(adress_src[i]) if band[-4:] == '.tif']
    for bands in bandlist:
        options = gdal.WarpOptions(cutlineDSName=shapefile_list[i], cropToCutline=True, xRes=Resolution, yRes=Resolution, resampleAlg='bilinear',)
        outband = gdal.Warp(srcDSOrSrcDSTab=adress_src[i] + bands,destNameOrDestDS=address_dest[i] + bands[:-4] + '_masked_resampled.tif',options=options)

#filling missing values for modis data
bandlist = [band for band in os.listdir(address_dest_modis) if band[-4:] == '.tif']
for bands in bandlist:
    #ds2 = rasterio.open('D:/download/GLDAS 2.2/masked/'+band)
    ds2 = rasterio.open(address_dest_modis + bands)
    nparray1 = ds2.read(1).astype('int16')
    nparray2 = nparray1.flatten()
    if bands[21:23]=='b0':
        cc=np.where(nparray2==-28672)
        if np.min(nparray2) == -28672:
            for i in range(5):
                np.put(nparray2,cc[0],np.average(nparray2))
    elif bands[12:15]=='Emi' or bands[12:15]=='LST' or bands[21:24]=='vze' or  bands[21:24]=='sze':
        cc = np.where(nparray2 == 0)
        for i in range(5):
            if np.min(nparray2) == 0:
                np.put(nparray2, cc[0], np.average(nparray2))

    first_array=nparray2.reshape(nparray1.shape[0],nparray1.shape[1])
    first_array=first_array[1:833,:1728]
    image = rasterio.open(address_dest_modis+bands[:-4]+'_revised.tif', 'w', width=first_array.shape[1], height=first_array.shape[0], count=1,
                      crs=ds2.crs, transform=ds2.transform
                      , dtype='int16', driver='GTiff')
    image.write(first_array, 1)

#filling missing values for ET data
bandlist=[band for band in os.listdir(address_dest_ET) if band[-4:]=='.tif']
for bands in bandlist:
    #ds2 = rasterio.open('D:/download/GLDAS 2.2/masked/'+band)
    ds2 = rasterio.open(address_dest_ET + bands)
    nparray1 = ds2.read(1).astype('int16')
    nparray2 = nparray1.flatten()
    cc=np.where(nparray2==9999)
    if np.max(nparray2) == 9999:
        for i in range(5):
            np.put(nparray2,cc[0],np.average(nparray2))
    cd = np.where(nparray2 == 0)
    if np.min(nparray2) == 0:
        for i in range(5):
            np.put(nparray2, cd[0], np.average(nparray2))
    first_array=nparray2.reshape(nparray1.shape[0],nparray1.shape[1])
    first_array = first_array[1:833, :1728]
    image = rasterio.open(address_dest_ET+bands[:-4]+'_revised.tif', 'w', width=first_array.shape[1], height=first_array.shape[0], count=1,
                      crs=ds2.crs, transform=ds2.transform, dtype='int16', driver='GTiff')
    image.write(first_array, 1)

#putting data in shape X_size, Y_size in numpy array : [number of years, number of days in month, X_size, Y_size, Number of Features]
Start=2004
End=2020
Modis_day_list=[['081', '089', '097', '105',],
                ['113', '121', '129', '137',],
                ['145', '153', '161', '169',],
                ['177', '185', '193', '201',],
                ['209', '217,' '225', '233',],
                ['241', '249', '257', '265']]
address_DEM=input("please insert address for DEM derived data eg 'F:/data/': ")
Y_test_mse_N, Y_test_mae_N, Y_valid_mse_N, Y_valid_mae_N, Y_train_mse_N, Y_train_mae_N = [], [], [], [], [], [],
valllosss, combination = [], []
Leader_scores=[]
Leader_positions=[]

#Main function
lb=[1,1,1,.2,.00001,1,3,]
ub=[4.99,4.99,2.99,.5,.005,3.99,6.99,]
SearchAgents_no=input('please inter number of search agents')
Max_iter=input('please inter number of maximun iteration')
Season=0

def fitness(X):
    for season in range (len(Modis_day_list)):
        year_list=[str(a) for a in range(Start,End)]
        number_of_days=len(Modis_day_list[season])
        Pixel_resolution=2**int(X[6])
        bandlist = [band for band in os.listdir(address_dest_modis) if band[-4:] == '.tif']
        ds2 = rasterio.open(address_dest_modis + bandlist[0])
        nparray = ds2.read(1).astype('int16')
        Number_Samples=nparray[0]*nparray[1]/Pixel_resolution
        #DEM?

        stack_modis=np.zeros((End-Start,number_of_days,Number_Samples,Pixel_resolution,Pixel_resolution,15)).astype(dtype='int16')
        stack_modis_prime=np.zeros((Number_Samples,Pixel_resolution,Pixel_resolution,1)).astype(dtype='int16')
        stack_ssebop_prime=np.zeros((Number_Samples,Pixel_resolution,Pixel_resolution,1)).astype(dtype='int16')
        stack_prop_prime=np.zeros((Number_Samples,Pixel_resolution,Pixel_resolution,1)).astype(dtype='int16')
        stack_dem_prime = np.zeros((Number_Samples, Pixel_resolution, Pixel_resolution, 1)).astype(dtype='int16')

        address_modis=address_dest_modis
        bandlist_modis=[band for band in os.listdir(address_modis) if band[-10:]=='masked.tif']
        c = np.zeros((Number_Samples,Pixel_resolution,Pixel_resolution)).astype(dtype='int16')
        y=0
        day_list=Modis_day_list[season]
        for year in year_list:
            bands_years = [band for band in bandlist_modis if band[-34:-30] == year]
            #bands_years = [band for band in bandlist_modis if band[-27:-23] == year]

            d=0
            for day in day_list:
                f=0
                band_day=[band for band in bands_years if band[-30:-27] == day and band[19:21]!='No' and band[15:18]!='Pro']
                #band_day=[band for band in bands_years if band[-23:-20] == day and band[19:21]!='No' and band[15:18]!='Pro']
                #print(band_day)
                for bands in band_day:
                    #print(bands)
                    path = os.path.join(address_modis, bands)
                    rasterBand = rasterio.open(path, driver='GTiff')
                    num = rasterBand.read(1).astype('int16')
                    #num = num[:896, :1600]
                    g = 0
                    for i in range(int((num.shape[-1]) / Pixel_resolution)):
                        for j in range(int((num.shape[-2]) / Pixel_resolution)):
                            b = num[ j * Pixel_resolution:j * Pixel_resolution + Pixel_resolution, i * Pixel_resolution:i * Pixel_resolution + Pixel_resolution]
                            b=b.reshape(1, Pixel_resolution,Pixel_resolution)
                            c[g, :, :] = b
                            g += 1
                    c = c.reshape(Number_Samples,Pixel_resolution,Pixel_resolution,1)
                    #print(c.shape,'c')
                    stack_modis_prime=np.append(stack_modis_prime,c,axis=-1)
                    c = np.zeros((Number_Samples,Pixel_resolution,Pixel_resolution))
                for i in range(2):
                    stack_modis_prime=np.expand_dims(stack_modis_prime, axis=0)
                stack_modis[y,d,:, :, :,:]=stack_modis_prime[:,:,:,:,:,1:]
                stack_modis_prime = np.zeros((Number_Samples,Pixel_resolution,Pixel_resolution, 1)).astype(dtype='int16')
                d+=1
            y+=1

        print('finish')

        #ssebop
        year_list=[str(a) for a in range(Start,End)]
        stack_ssebop=np.zeros((End-Start,number_of_days,Number_Samples,Pixel_resolution,Pixel_resolution,1)).astype(dtype='int16')
        address_ssebop=address_dest_ET
        bandlist_ssebop=[band for band in os.listdir(address_ssebop) if band[-10:]=='masked.tif']
        c = np.zeros((Number_Samples,Pixel_resolution,Pixel_resolution)).astype(dtype='int16')
        y=0
        for year in year_list:
            bands_years = [band for band in bandlist_ssebop if band[3:7] == year]
            d=0
            for day in day_list:
                f=0
                band_day=[band for band in bands_years if band[7:10] == day]
                for bands in band_day:
                    #print(bands)
                    path = os.path.join(address_ssebop, bands)
                    rasterBand = rasterio.open(path, driver='GTiff')
                    num = rasterBand.read(1).astype('int16')
                    #num = num[:896, :1600]
                    g = 0
                    for i in range(int((num.shape[-1]) / Pixel_resolution)):
                        for j in range(int((num.shape[-2]) / Pixel_resolution)):
                            b = num[j * Pixel_resolution:j * Pixel_resolution + Pixel_resolution, i * Pixel_resolution:i * Pixel_resolution + Pixel_resolution]
                            b = b.reshape(1, Pixel_resolution, Pixel_resolution)
                            c[g, :, :] = b
                            g += 1
                    c = c.reshape(Number_Samples,Pixel_resolution,Pixel_resolution, 1)
                        # print(c.shape,'c')
                    stack_ssebop_prime = np.append(stack_ssebop_prime, c, axis=-1)
                    c = np.zeros((Number_Samples,Pixel_resolution,Pixel_resolution))
                for i in range(2):
                    stack_ssebop_prime = np.expand_dims(stack_ssebop_prime, axis=0)
                stack_ssebop[y, d, :, :, :, :] = stack_ssebop_prime[:, :, :, :, :, 1:]
                stack_ssebop_prime = np.zeros((Number_Samples,Pixel_resolution,Pixel_resolution, 1)).astype(dtype='int16')
                d+=1
            y+=1

        print('finish')

        #DEM data
        f=0
        c = np.zeros((Number_Samples,Pixel_resolution,Pixel_resolution)).astype(dtype='int16')
        bandlist_modis=[band for band in os.listdir(address_DEM) if band[-4:]=='.tif']
        for band in bandlist_modis:
            path = os.path.join(address_DEM, band)
            rasterBand = rasterio.open(path, driver='GTiff')
            num = rasterBand.read(1).astype('int16')
            g = 0
            for i in range(int((num.shape[-1]) / Pixel_resolution)):
                for j in range(int((num.shape[-2]) / Pixel_resolution)):
                    b = num[j * Pixel_resolution:j * Pixel_resolution + Pixel_resolution, i * Pixel_resolution:i * Pixel_resolution + Pixel_resolution]
                    b = b.reshape(1, Pixel_resolution, Pixel_resolution)
                    c[g, :, :] = b
                    g += 1
            c = c.reshape(Number_Samples,Pixel_resolution,Pixel_resolution,1)
            stack_dem_prime = np.append(stack_dem_prime, c, axis=-1)
            c = np.zeros((Number_Samples,Pixel_resolution,Pixel_resolution)).astype(dtype='int16')
        b=stack_dem_prime[:,:,:,1:]
        b = np.repeat(b[np.newaxis,:, :,:,:], number_of_days, axis=0)
        stack_dem=np.repeat(b[np.newaxis,:, :,:,:,:], End-Start, axis=0)
        DEM=stack_dem

        #prope_extra
        address_extra_modis=address_dest_modis
        stack_prop=np.zeros((End-Start,number_of_days,Number_Samples,Pixel_resolution,Pixel_resolution,3)).astype(dtype='int16')
        h=0
        c = np.zeros((Number_Samples,Pixel_resolution,Pixel_resolution)).astype(dtype='int16')
        bandlist_modis=[band for band in os.listdir(address_extra_modis) if band[-10:]=='masked.tif']
        bandlist_modis=[band for band in bandlist_modis if band[15:20]=='Prop2' or band[11:18]=='Percent']
        yearlist=[str(years) for years in range(Start,End)]
        for years in yearlist:
            band_year = [band for band in bandlist_modis if band[-34:-30] == years]
            #band_year=[band for band in bandlist_modis if band[-27:-23]==years]
            for band in band_year:
                path = os.path.join(address_extra_modis, band)
                rasterBand = rasterio.open(path, driver='GTiff')
                num = rasterBand.read(1).astype('int16')

                g = 0
                for i in range(int((num.shape[-1]) / Pixel_resolution)):
                    for j in range(int((num.shape[-2]) / Pixel_resolution)):
                        b = num[j * Pixel_resolution:j * Pixel_resolution + Pixel_resolution, i * Pixel_resolution:i * Pixel_resolution + Pixel_resolution]
                        b = b.reshape(1, Pixel_resolution, Pixel_resolution)
                        c[g, :, :] = b
                        g += 1
                c = c.reshape(Number_Samples,Pixel_resolution,Pixel_resolution,1)
                stack_prop_prime= np.append(stack_prop_prime, c, axis=-1)
                c = np.zeros((Number_Samples,Pixel_resolution,Pixel_resolution)).astype(dtype='int16')
            stack_prop_1=np.repeat(stack_prop_prime[np.newaxis,:, :,:,1:], number_of_days, axis=0)
            stack_prop[h,:,:,:,:,:]=stack_prop_1.reshape(1,number_of_days,Number_Samples,Pixel_resolution,Pixel_resolution,len(band_year))
            stack_prop_prime=np.zeros((Number_Samples,Pixel_resolution,Pixel_resolution,1)).astype(dtype='int16')
            h+=1
        Prop=stack_prop

        print(stack_modis.shape,Prop.shape,stack_ssebop.shape)
        data=np.concatenate((stack_modis,Prop,DEM,stack_ssebop),axis=-1)
        print(data.shape)
        print('start')

        with h5py.File(f'D:/h5/case_study_month {season+1}.h5', 'w') as hf:
            hf.create_dataset("evapotranspiration",  data=data,dtype='int16',compression='gzip')

        #Data Normalization
        hf_1 = h5py.File(f'D:/h5/case_study_month {season+1}.h5', 'r')
        Data = hf_1['evapotranspiration']
        Data = (np.array(Data)).astype('float32')

        print(Data.dtype,Data.shape)
        Empty=np.zeros((End-Start,number_of_days,Number_Samples,Pixel_resolution,Pixel_resolution,Data.shape[-1])).astype(dtype='float32')
        print('start')
        start_time = time.time()
        for i in range(Data.shape[-1]):
            max=np.max(Data[:int((End-Start)*.8),:,:,:,:,i])
            min=np.min(Data[:int((End-Start)*.8),:,:,:,:,i])
            if abs(min)>=max:
                max=abs(min)
            Empty[:,:,:,:,:,i]=(Data[:,:,:,:,:,i].astype(dtype='float32'))/max
        print("data_normalization" ,(time.time() - start_time))

        start_time = time.time()
        with h5py.File(f'D:/h5/case_study_month {season+1}.h5', 'w') as hf:
            hf.create_dataset("evapotranspiration",  data=Empty,dtype='float32',compression='gzip')
        print("save time" ,(time.time() - start_time))

    #Training the Model
    hf_1 = h5py.File(f'D:/h5/case_study_month {Season+1}.h5', 'r')
    Data = hf_1['evapotranspiration']
    Data = (np.array(Data))

    X_train=Data[:int((End-Start)*.8),:,:,:,:,:-1]
    Y_train=Data[:int((End-Start)*.8),:,:,:,:,-1]
    X_valid=Data[int((End-Start)*.8):int((End-Start)*.9),:,:,:,:,:-1]
    Y_valid=Data[int((End-Start)*.8):int((End-Start)*.9),:,:,:,:,-1]
    X_test=Data[int((End-Start)*.9):,:,:,:,:,:-1]

    Y_train=np.reshape((Y_train),(Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],Y_train.shape[3],Y_train.shape[4],1))
    Y_valid=np.reshape((Y_valid),(Y_valid.shape[0],Y_valid.shape[1],Y_valid.shape[2],Y_valid.shape[3],Y_valid.shape[4],1))

    X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1]*X_train.shape[2], X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])
    X_valid = X_valid.reshape(X_valid.shape[0]*X_valid.shape[1]*X_valid.shape[2], X_valid.shape[-3], X_valid.shape[-2], X_valid.shape[-1])
    X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1]*X_test.shape[2], X_test.shape[-3], X_test.shape[-2], X_test.shape[-1])

    Y_train = Y_train.reshape(Y_train.shape[0]*Y_train.shape[1]*Y_train.shape[2], Y_train.shape[-3], Y_train.shape[-2], Y_train.shape[-1])
    Y_valid = Y_valid.reshape(Y_valid.shape[0]*Y_valid.shape[1]*Y_valid.shape[2], Y_valid.shape[-3], Y_valid.shape[-2], Y_valid.shape[-1])

    print(X)
    combination.append([X])
    Y_test = Data[14:, :, :, :, :, -1]
    Y_test = np.reshape((Y_test),
                        (Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], Y_test.shape[3], Y_test.shape[4], 1))
    Y_test = Y_test.reshape(Y_test.shape[0] * Y_test.shape[1] * Y_test.shape[2], Y_test.shape[-3], Y_test.shape[-2],
                            Y_test.shape[-1])
    input_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])
    num_chan_out = 1
    dropout=round(X[3],3)
    param_kernal=int(X[2])*2+1
    fms=int(X[0])*16
    fms2=int(X[1])*16
    batch_size=128
    learning_rate=X[4]
    value=int(X[5])

    adam = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam", )
    inputs = K.layers.Input(input_shape, name="MRImages")

    # Convolution parameters
    params = dict(kernel_size=(param_kernal, param_kernal), activation="tanh",
              padding="same",)

    # Transposed convolution parameters
    params_trans = dict(kernel_size=(2, 2), strides=(2, 2),
                    padding="same")

    encodeAAa = K.layers.Conv2D(name="encodeAAa", filters=fms*2, **params)(inputs)
    encodeAAb = K.layers.Conv2D(name="encodeAAb", filters=fms2, **params)(encodeAAa)

    #A #32*32*16
    encodeAa = K.layers.Conv2D(name="encodeAa", filters=fms, **params)(encodeAAb)
    encodeAb = K.layers.Conv2D(name="encodeAb", filters=fms, **params)(encodeAa)
    encodeAb = K.layers.SpatialDropout2D(dropout)(encodeAb)

    # 16*16*16
    poolA = K.layers.MaxPooling2D(name="poolA", pool_size=(2, 2))(encodeAb)
    #B 16*16*32
    encodeBa = K.layers.Conv2D(name="encodeBa", filters=fms*2, **params)(poolA)
    encodeBb = K.layers.Conv2D(name="encodeBb", filters=fms*2, **params)(encodeBa)
    encodeBb = BatchNormalization() (encodeBb)
    encodeBb = Activation(activation='tanh') (encodeBb)

    # 8*8*32
    poolB = K.layers.MaxPooling2D(name="poolB", pool_size=(2, 2))(encodeBb)
    #C 8*8*64
    encodeCC = K.layers.Conv2D(name="encodeCa", filters=fms*4, **params)(poolB)
    encodeCb = K.layers.Conv2D(name="encodeCb", filters=fms*4, **params)(encodeCC)
    encodeCb = K.layers.SpatialDropout2D(dropout)(encodeCb)

    if value == 2:
        # bottle_neck
        # 4*4*64
        poolC = K.layers.MaxPooling2D(name="poolC", pool_size=(2, 2))(encodeCb)
        # 4*4*128
        encodeDa = K.layers.Conv2D(name="encodeDa", filters=fms * 8, **params)(poolC)
        encodeDb = K.layers.Conv2D(name="encodeDb", filters=fms * 8, **params)(encodeDa)

        # 8*8*64
        upa = K.layers.Conv2DTranspose(name="transconvE", filters=fms * 4, **params_trans)(encodeDb)
        K_encodeCb = calculat_K()(encodeCb)
        Q_upa = calculate_Q()(upa)
        layera = self_attention(k=K_encodeCb, q=Q_upa, v=upa)
        concatE = K.layers.concatenate([encodeCb,layera ], axis=-1, name="concatE")

        # 8*8*256
        decodeCa = K.layers.Conv2D(name="decodeCa", filters=fms * 4, **params)(concatE)
        decodefinal = K.layers.Conv2D(name="decodeCb", filters=fms * 4, **params)(decodeCa)
        # 16*16*256
        upb = K.layers.Conv2DTranspose(name="transconvC", filters=fms * 2, **params_trans)(decodefinal)
    elif value == 1:
        # C 4*4*64
        poolC = K.layers.MaxPooling2D(name="poolC", pool_size=(2, 2))(encodeCb)
        # C 4*4*128
        encodeDa = K.layers.Conv2D(name="encodeDa", filters=fms * 8, **params)(poolC)
        encodeDb = K.layers.Conv2D(name="encodeDb", filters=fms * 8, **params)(encodeDa)
        # C 2*2*128
        poolD = K.layers.MaxPooling2D(name="poolD", pool_size=(2, 2))(encodeDb)
        # C 2*2*256
        encodeEa = K.layers.Conv2D(name="encodeEa", filters=fms * 16, **params)(poolD)
        encodeEb = K.layers.Conv2D(name="encodeEb", filters=fms * 16, **params)(encodeEa)
        encodeEb = BatchNormalization()(encodeEb)
        encodeEb = Activation(activation='tanh')(encodeEb)
        encodeEb = K.layers.SpatialDropout2D(dropout)(encodeEb)
        # C 4*4*128
        upaa = K.layers.Conv2DTranspose(name="transconvE", filters=fms * 8, **params_trans)(encodeEb)
        K_encodeDc = calculat_K()(encodeDb)
        Q_upaa = calculate_Q()(upaa)
        layeraa = self_attention(k=K_encodeDc, q=Q_upaa, v=upaa)
        concatE = K.layers.concatenate([layeraa, encodeDb ,], axis=-1, name="concatE")
        # C 4*4*128
        decodeDa = K.layers.Conv2D(name="decodeDa", filters=fms * 8, **params)(concatE)
        decodeDa = K.layers.Conv2D(name="decodeDb", filters=fms * 8, **params)(decodeDa)
        # C 8*8*64
        upaaa = K.layers.Conv2DTranspose(name="transconvEE", filters=fms * 4, **params_trans)(decodeDa)
        K_encodeCb = calculat_K()(encodeCb)
        Q_upaaa = calculate_Q()(upaaa)
        layeraaa = self_attention(k=K_encodeCb, q=Q_upaaa, v=upaaa)
        concatF = K.layers.concatenate([ encodeCb,layeraaa], axis=-1, name="concatF")

        # C 8*8*64
        decodeCa = K.layers.Conv2D(name="decodeCaa", filters=fms * 4, **params)(concatF)
        decodefinal = K.layers.Conv2D(name="decodeCbbb", filters=fms * 4, **params)(decodeCa)
        # C 16*16*32
        upb = K.layers.Conv2DTranspose(name="transconvC", filters=fms * 2, **params_trans)(decodefinal)
    else:
        upb = K.layers.Conv2DTranspose(name="transconvC", filters=fms * 2, **params_trans)(encodeCb)

    # C 16*16*32
    K_encodeBb = calculat_K()(encodeBb)
    Q_upb = calculate_Q()(upb)
    layerb = self_attention(k=K_encodeBb, q=Q_upb, v=upb, )
    concatC = K.layers.concatenate([encodeBb,layerb,], axis=-1, name="concatC")

    # 16*16*128
    decodeBa = K.layers.Conv2D(name="decodeBa", filters=fms * 2, **params)(concatC)
    decodeBb = K.layers.Conv2D(name="decodeBb", filters=fms * 2, **params)(decodeBa)
    decodeBb = BatchNormalization()(decodeBb)
    decodeBb = Activation(activation='tanh')(decodeBb)
    decodeBb = K.layers.SpatialDropout2D(dropout)(decodeBb)

    # 32*32*16
    upc = K.layers.Conv2DTranspose(name="transconvB", filters=fms * 1, **params_trans)(decodeBb)
    K_encodeAb = calculat_K()(encodeAb)
    Q_upc = calculate_Q()(upc)
    layerbb = self_attention(k=K_encodeAb, q=Q_upc, v=upc, )
    concatD = K.layers.concatenate([ upc,layerbb ], axis=-1, name="concatD")

    # 32*32*16
    convOuta = K.layers.Conv2D(name="convOuta", filters=fms, **params)(concatD)
    convOuta = K.layers.SpatialDropout2D(dropout)(convOuta)
    convOutb = K.layers.Conv2D(name="convOutb", filters=fms/4, **params)(convOuta)
    convOutc = K.layers.Conv2D(name="convOutd", filters=fms/8, **params)(convOutb)

    prediction = K.layers.Conv2D(name="PredictionMask",
                                 filters=num_chan_out, kernel_size=(1, 1),
                                 activation="linear")(convOutc)


    model = K.models.Model(inputs=[inputs], outputs=[prediction], name="2DUNet_Brats_Decathlon")
    model.compile(loss='mean_absolute_error', # one may use 'mean_absolute_error' as  mean_squared_error
                      optimizer=adam,
                      metrics=[tf.keras.metrics.MeanSquaredError(),])

    #model.summary()
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='saved_model_Unet_attention.h5',
        monitor='val_loss',
        mode='min',
        save_best_only=True,)
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=200, verbose=1,
              callbacks=[model_checkpoint_callback,tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25,baseline=0.08,restore_best_weights=True)],validation_data=(X_valid,Y_valid))


    saved_model = load_model('saved_model_Unet_attention.h5',
                             custom_objects={'calculat_K':calculat_K,'calculate_Q':calculate_Q,'self_attention':self_attention,'channel_attention':channel_attention,})

    saved_model.compile(loss='mean_absolute_error', # one may use 'mean_absolute_error' as  mean_squared_error
                      optimizer=adam,
                      metrics=[ tf.keras.metrics.MeanSquaredError()])

    score_test = saved_model.evaluate(X_test, Y_test, verbose=0,batch_size=batch_size)
    print(score_test,'= score_test')
    Y_test_mse_N.append(score_test[-1])
    Y_test_mae_N.append(score_test[0])
    score_train = saved_model.evaluate(X_train, Y_train, verbose=0,batch_size=batch_size)
    print(score_train, '= score_train')
    Y_train_mse_N.append(score_train[-1])
    Y_train_mae_N.append(score_train[0])
    score_valid = saved_model.evaluate(X_valid, Y_valid, verbose=0,batch_size=batch_size)
    print(score_valid, '= score_valid')
    Y_valid_mse_N.append(score_valid[-1])
    Y_valid_mae_N.append(score_valid[0])

    valloss = score_valid[0]
    valllosss.append(valloss)


    preds = saved_model.predict(X_test, batch_size=128)
    print(preds.shape, Y_test.shape)
    Y_test = np.reshape((Y_test), (Y_test.shape[0], Y_test.shape[1] * Y_test.shape[2]))
    preds = np.reshape((preds), (preds.shape[0], preds.shape[1] * preds.shape[2]))

    preds = preds * 11435
    Y_test = Y_test * 11435
    list = []
    for i in range(Y_test.shape[0]):
        for j in range(Y_test.shape[1]):
            if Y_test[i, j] != 0 and preds[i, j] != 0:
                list.append((np.absolute((Y_test[i, j] - preds[i, j]) / Y_test[i, j])).item())
    print(sum(list) / len(list))

    print(r2_score(Y_test, preds),
          mean_absolute_error(Y_test, preds), (mean_squared_error(Y_test, preds)) ** .5)
    print(np.average(Y_test), np.max(Y_test), np.min(Y_test))

    del model
    tensorflow.keras.backend.clear_session()
    k.clear_session()
    tf.compat.v1.reset_default_graph()
    return valloss


class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual=[]
        self.convergence = []
        self.optimizer=""
        self.objfname=""
        self.startTime=0
        self.endTime=0
        self.executionTime=0
        self.lb=0
        self.ub=0
        self.dim=0
        self.popnum=0
        self.maxiers=0


#attention modules
class Self_attention(Layer):
    def __init__(self,activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(Self_attention, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
    def build(self, input_shape,):
        #optimize second variable input_shape[-3]
        self.Wq = self.add_weight(shape=(input_shape[-1],input_shape[-2],input_shape[-3],),
                                        name='Wq',
                                        initializer=self.kernel_initializer,)
        self.Wk = self.add_weight(shape=(input_shape[-1], input_shape[-2], input_shape[-3],),
                                        name='Wk',
                                        initializer=self.kernel_initializer, )
        self.built = True

    def call(self, input,trainable=True):
        input = (tf.transpose(input, perm=[0,3,1, 2]))
        print(input.shape,'shapex')
        q = self.recurrent_activation(tf.matmul(input, self.Wq))
        K=self.activation(tf.matmul(input, self.Wk))
        q_k = self.recurrent_activation(tf.matmul(q, tf.transpose(K, perm=[0,1, 3, 2])))
        q_k = (k.softmax(q_k, axis=-1))
        x=tf.matmul(input, q_k)
        x = (tf.transpose(x, perm=[0,2, 3, 1]))
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = {
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
              }
        base_config = super(Self_attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class calculat_K(Layer):
    def __init__(self, activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(calculat_K, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)

    def build(self, input_shape, ):
        self.Wk = self.add_weight(shape=(input_shape[-1], input_shape[-2], input_shape[-3],),
                                  name='Wk',
                                  initializer=self.kernel_initializer, )

        self.built = True

    def call(self, x, trainable=True):
        x = (tf.transpose(x, perm=[0, 3, 1, 2]))
        K=self.activation(tf.matmul(x, self.Wk))
        K=tf.transpose(K, perm=[0,1, 3, 2])
        return K

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
        }
        base_config = super(calculat_K, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class calculate_Q(Layer):
    def __init__(self, activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(calculate_Q, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)

    def build(self, input_shape, ):
        self.Wq = self.add_weight(shape=(input_shape[-1], input_shape[-2], input_shape[-3],),
                                  name='Wq',
                                  initializer=self.kernel_initializer, )

        self.built = True

    def call(self, x, trainable=True):
        x = (tf.transpose(x, perm=[0, 3, 1, 2]))
        q = self.recurrent_activation(tf.matmul(x, self.Wq))
        #print(x.shape, 'shapex')
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3], input_shape[1], input_shape[2])

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
        }
        base_config = super(calculate_Q, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
def self_attention(q, k,v ):
    v = Permute((3, 1, 2))(v)
    Q_K=tf.matmul(q,k)
    Q_K=Activation('sigmoid')(Q_K)
    Q_K=tf.keras.activations.softmax(Q_K,axis=-1)
    #badesh ya ghablesh? ghablesh
    #mean ba halate ghabl?
    #badesh ba yek activation?

    """Q_K=tf.reduce_mean(Q_K, axis=1,keepdims=True)
    Q_K = Activation('tanh')(Q_K)"""

    x = tf.matmul(v, Q_K)
    #x=K.layers.concatenate([x1, x2], axis=-3,)
    #print(x.shape,'x')
    #x=Permute((2,3,1))(x)
    # Transposed convolution parameters
    #x=K.layers.Conv2D(filters=x.shape[-1]/2,**params)(x)
    x = Permute((2, 3, 1))(x)
    return x
def channel_attention(input_feature, ratio=4):
    channel_axis = -1
    channel = input_feature.shape[channel_axis]
    shared_layer_one = Dense(channel // ratio,
                             activation='tanh',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = add([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


#Harris Hawk Optimization Algorithm
def HHO(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # dim=30
    # SearchAgents_no=50
    # lb=-100
    # ub=100
    # Max_iter=500

    # initialize the location and Energy of the rabbit
    Rabbit_Location = np.zeros(dim)
    Rabbit_Energy = float("inf")  # change this to -inf for maximization problems

    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = np.asarray(lb)
    ub = np.asarray(ub)

    # Initialize the locations of Harris' hawks
    X = np.asarray(
        [x * (ub - lb) + lb for x in np.random.uniform(0, 1, (SearchAgents_no, dim))]
    )

    # Initialize convergence
    convergence_curve = np.zeros(Max_iter)

    ############################
    s = solution()

    print('HHO is now tackling  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################

    t = 0  # Loop counter

    # Main loop
    while t < Max_iter:
        for i in range(0, SearchAgents_no):

            # Check boundries

            X[i, :] = np.clip(X[i, :], lb, ub)

            # fitness of locations
            fitness = objf(X[i, :])

            # Update the location of Rabbit
            if fitness < Rabbit_Energy:  # Change this to > for maximization problem
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()

        E1 = 2 * (1 - (t / Max_iter))  # factor to show the decreaing energy of rabbit

        # Update the location of Harris' hawks
        for i in range(0, SearchAgents_no):

            E0 = 2 * random.random() - 1  # -1<E0<1
            Escaping_Energy = E1 * (
                E0
            )  # escaping energy of rabbit Eq. (3) in the paper

            # -------- Exploration phase Eq. (1) in paper -------------------

            if abs(Escaping_Energy) >= 1:
                # Harris' hawks perch randomly based on 2 strategy:
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no * random.random())
                X_rand = X[rand_Hawk_index, :]
                if q < 0.5:
                    # perch based on other family members
                    X[i, :] = X_rand - random.random() * abs(
                        X_rand - 2 * random.random() * X[i, :]
                    )

                elif q >= 0.5:
                    # perch on a random tall tree (random site inside group's home range)
                    X[i, :] = (Rabbit_Location - X.mean(0)) - random.random() * (
                        (ub - lb) * random.random() + lb
                    )

            # -------- Exploitation phase -------------------
            elif abs(Escaping_Energy) < 1:
                # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                # phase 1: ----- surprise pounce (seven kills) ----------
                # surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r = random.random()  # probablity of each event

                if (
                    r >= 0.5 and abs(Escaping_Energy) < 0.5
                ):  # Hard besiege Eq. (6) in paper
                    X[i, :] = (Rabbit_Location) - Escaping_Energy * abs(
                        Rabbit_Location - X[i, :]
                    )

                if (
                    r >= 0.5 and abs(Escaping_Energy) >= 0.5
                ):  # Soft besiege Eq. (4) in paper
                    Jump_strength = 2 * (
                        1 - random.random()
                    )  # random jump strength of the rabbit
                    X[i, :] = (Rabbit_Location - X[i, :]) - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i, :]
                    )

                # phase 2: --------performing team rapid dives (leapfrog movements)----------

                if (
                    r < 0.5 and abs(Escaping_Energy) >= 0.5
                ):  # Soft besiege Eq. (10) in paper
                    # rabbit try to escape by many zigzag deceptive motions
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i, :]
                    )
                    X1 = np.clip(X1, lb, ub)

                    if objf(X1) < fitness:  # improved move?
                        X[i, :] = X1.copy()
                    else:  # hawks perform levy-based short rapid dives around the rabbit
                        X2 = (
                            Rabbit_Location
                            - Escaping_Energy
                            * abs(Jump_strength * Rabbit_Location - X[i, :])
                            + np.multiply(np.random.randn(dim), Levy(dim))
                        )
                        X2 = np.clip(X2, lb, ub)
                        if objf(X2) < fitness:
                            X[i, :] = X2.copy()
                if (
                    r < 0.5 and abs(Escaping_Energy) < 0.5
                ):  # Hard besiege Eq. (11) in paper
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X.mean(0)
                    )
                    X1 = np.clip(X1, lb, ub)

                    if objf(X1) < fitness:  # improved move?
                        X[i, :] = X1.copy()
                    else:  # Perform levy-based short rapid dives around the rabbit
                        X2 = (
                            Rabbit_Location
                            - Escaping_Energy
                            * abs(Jump_strength * Rabbit_Location - X.mean(0))
                            + np.multiply(np.random.randn(dim), Levy(dim))
                        )
                        X2 = np.clip(X2, lb, ub)
                        if objf(X2) < fitness:
                            X[i, :] = X2.copy()

        convergence_curve[t] = Rabbit_Energy
        if t % 1 == 0:
            print(["At iteration "+ str(t)+ " the best fitness is "+ str(Rabbit_Energy)])
            print(["At iteration " + str(t) + " the best set is " + str(Rabbit_Location)])
            Leader_scores.append(Rabbit_Energy)
            Leader_positions.append(Rabbit_Location)
        t = t + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "HHO"
    s.objfname = objf.__name__
    s.best = Rabbit_Energy
    s.bestIndividual = Rabbit_Location

    return s
def Levy(dim):
    beta=1.5
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u= 0.01*np.random.randn(dim)*sigma
    v = np.random.randn(dim)
    zz = np.power(np.absolute(v),(1/beta))
    step = np.divide(u,zz)
    return step

#Run the Optimization
HHO(objf=fitness, lb=lb, ub=ub, dim=7, SearchAgents_no=SearchAgents_no, Max_iter=Max_iter)
print('Y_test_mse_N,Y_test_mae_N ', Y_test_mse_N, Y_test_mae_N)
print('Y_valid_mse_N,Y_valid_mae_N ', Y_valid_mse_N, Y_valid_mae_N)
print('Y_train_mse_N,Y_train_mae_N ', Y_train_mse_N, Y_train_mae_N)
print('Leader_scores: ', Leader_scores)
print('Leader_positions: ', Leader_positions)

val, idx = min((val, idx) for (idx, val) in enumerate(valllosss))
print('val,idx ', val, idx)
print(Y_test_mse_N[idx], Y_test_mae_N[idx], Y_train_mse_N[idx],
      Y_train_mae_N[idx], Y_valid_mse_N[idx], Y_valid_mae_N[idx], )
print('valllosss: ', valllosss)
with open('florida_farvardin_attention_self_attention_new', "w") as text_file:
    print(combination,
          file=text_file)

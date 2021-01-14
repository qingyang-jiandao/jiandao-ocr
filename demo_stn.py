# -*- coding=utf-8 -*-
import sys
sys.path.append('G:/fqy-jiandao/svn/jiandao-caffe/python')
import caffe
from PIL import Image
import numpy as np
import matplotlib
from matplotlib import cm
from ConvertRGBA2RGB import alpha_composite
np.set_printoptions(threshold=np.nan)

# wight_file = './model-stnpt-fnn/stnpt_fnn_vgg_lw0.05-64x256-t48_iter_1500000.caffemodel'
wight_file = './model-stnpt-fnn/stnpt_fnn_vgg_lw0.7-64x256-t48-nd_iter_1500000.caffemodel'
deploy_file = './model-stnpt-fnn/deploy-stnpt-fnn-lossweight-vgg-64x256-t48.prototxt'

test_img = 'test-images/data_25_26.jpg'

# set device
caffe.set_device(0)
caffe.set_mode_gpu()

# load model
net = caffe.Net(deploy_file, wight_file, caffe.TEST)

#####################set blob data ###########################

#================PIL=================================
img = Image.open(test_img).convert("RGB")
width, height = img.size
target_height = 32
target_width = 128
# if target_width<128:
#     target_width = 128
init_img = img.resize((target_width, target_height), Image.ANTIALIAS)
matplotlib.image.imsave(str("output/resize")+str(".jpg"), init_img)

in_ = np.array(init_img, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ = in_/255.
in_ = in_.transpose((2,0,1))

net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_

#####################blob stn_data ###########################
target_height = 64
target_width = int(1.0*target_height*width/height)
print('target_height target_width:', target_height, target_width)
stn_input_img = img.resize((target_width, target_height), Image.ANTIALIAS)
in_stn = np.array(stn_input_img, dtype=np.float32)
in_stn = in_stn[:,:,::-1]
in_stn = in_stn/255.
in_stn = in_stn.transpose((2,0,1))

# np.set_printoptions(linewidth=6400)
# array_b = in_stn[0, :, :]
# print ('array_b.shape:', array_b.shape)
# print(array_b)

net.blobs['stn_data'].reshape(1, *in_stn.shape)
net.blobs['stn_data'].data[...] = in_stn

#----------------------------------
# run net
net.forward()
#----------------------------------

thetas = net.blobs['st/theta'].data[0]
print ('thetas.shape:', thetas.shape, type(thetas))
for i in range(thetas.shape[0]):
    theta = thetas[i]
    print('theta[%d]: %f' % (i, theta))

thetas = net.blobs['iter2_st/theta'].data[0]
print ('iter2_st/thetas.shape:', thetas.shape, type(thetas))
for i in range(thetas.shape[0]):
    theta = thetas[i]
    print('theta[%d]: %f' % (i, theta))
    
featuremaps = net.blobs['st/st_output'].data[0]
#featuremaps = net.blobs['st/tps_output'].data[0]
print ('st_output.shape:', featuremaps.shape, type(featuremaps))
for i in range(featuremaps.shape[0]):
    map = featuremaps[i]
    matplotlib.image.imsave("output/stn_out_" + str(i) +".jpg", map, cmap = cm.gray)
    #np.savetxt('output/stn_out_' + str(i) + '.csv', map, delimiter=',')

featuremaps = net.blobs['seg_output'].data[0]
print ('seg_output.shape:', featuremaps.shape, type(featuremaps))
for i in range(featuremaps.shape[0]):
    map = featuremaps[i]
    matplotlib.image.imsave("output/seg_output_" + str(i) +".jpg", map, cmap = cm.gray)
    #np.savetxt('output/seg_output_' + str(i) + '.csv', map, delimiter=',')

featuremaps = net.blobs['iter2_st/st_output'].data[0]
print ('iter2_st/st_output.shape:', featuremaps.shape, type(featuremaps))
for i in range(featuremaps.shape[0]):
    map = featuremaps[i]
    matplotlib.image.imsave("output/iter2_st_seg_output_" + str(i) +".jpg", map, cmap = cm.gray)
    #np.savetxt('output/seg_output_' + str(i) + '.csv', map, delimiter=',')

# get result
res = net.blobs['reshape2'].data
print('res shape is:', res.shape, type(res))

char_set = []
with open('dict_6862_ex_ex.list', 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        line = line.strip('\n\r')
        if line is not "":
            char_set.append(str(line))
        line = f.readline()

print('char num:', len(char_set))

small_add = 1e-20 * np.ones(res.shape, dtype=np.float32)
res = res + small_add
output = np.log(res)
print('output shape is:', output.shape)

output_char_list = ''
pre_char_index = 7134
for i in range(output.shape[0]):
    data = output[i, :output.shape[1]]
    indexMap = np.argmax(data, axis=0)
    if indexMap != 7134 and pre_char_index != indexMap:
        output_char_list = output_char_list + char_set[indexMap]

    pre_char_index = indexMap

print(output_char_list)
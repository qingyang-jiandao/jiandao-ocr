# -*- coding=utf-8 -*-
import sys
sys.path.append('G:/fqy-jiandao/svn/jiandao-caffe/python')
import caffe
from PIL import Image
import numpy as np
import cv2
import matplotlib
from matplotlib import cm
from ConvertRGBA2RGB import alpha_composite
# np.set_printoptions(threshold=np.nan)

## ctc decode.
# wight_file = './model-jiandao/stnpt-fnn-vgg-pro-stnfeature-last45_iter_600000.caffemodel'
# wight_file = './model-jiandao/stnpt-fnn-vgg-pro-stnfeature-xyxy2032_iter_600000.caffemodel'
wight_file = './model-jiandao/stnpt-fnn-vgg-pro-stnfeature-breakthrough_iter_420000.caffemodel'
deploy_file = './model-jiandao/deploy-stnpt-fnn-vgg-pro-stnfeature-64x256.prototxt'

# wight_file = './model-jiandao/stnpt-fnn-vgg-pro-stnfeature-sumcoefficient2_iter_600000.caffemodel'
# deploy_file = './model-jiandao/deploy-stnpt-fnn-vgg-pro-stnfeature-sumcoefficient2-64x256.prototxt'

## attention mechanism decode.
# wight_file = './model-jiandao/stnpt-fnn-vgg-stnfeature-att_iter_270000.caffemodel'
# # deploy_file = './model-jiandao/deploy-stnpt-fnn-vgg-pro-stnfeature-attention-64x256.prototxt'
# deploy_file = './model-jiandao/deploy-stnpt-fnn-vgg-pro-stnfeature-attention-64x256-optim.prototxt'

# wight_file = './model-jiandao/stnpt-fnn-vgg-seg-attlstmnode_iter_600000.caffemodel'
# deploy_file = './model-jiandao/deploy-stnpt-segnet-vgg-attention-lstmnode-64x256.prototxt'

# wight_file = './model-jiandao/stnpt-fnn-vgg-seg-attlstmunit_iter_600000.caffemodel'
# deploy_file = './model-jiandao/deploy-stnpt-segnet-vgg-attention-lstmunit-64x256.prototxt'


images = []
# test_img = 'test-images2/139_2.jpg'
# test_img = 'test-images2/21_1.jpg'
# test_img = 'test-images2/48_1.jpg'
# test_img = 'test-images2/71_0.jpg'
# test_img = 'test-images2/131_2.jpg'
# test_img = 'test-images2/118_1.jpg'
test_img = 'test-images2/42_1.jpg'
# test_img = 'test-images2/237_0.jpg'

# set device
caffe.set_device(0)
caffe.set_mode_gpu()

# caffe.set_mode_cpu()

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
init_img.save(str("output/resize.jpg"))

in_ = np.array(init_img, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ = in_/255.
in_ = in_.transpose((2,0,1))

net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_

#####################set blob stn_data ###########################
target_height = 64
target_width = int(1.0*target_height*width/height)
# target_width = 256
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

# thetas = net.blobs['iter2_st/theta'].data[0]
# print ('iter2_st/thetas.shape:', thetas.shape, type(thetas))
# for i in range(thetas.shape[0]):
#     theta = thetas[i]
#     print('theta[%d]: %f' % (i, theta))
    
featuremaps = net.blobs['st/st_output'].data[0]
#featuremaps = net.blobs['st/tps_output'].data[0]
print ('st_output.shape:', featuremaps.shape, type(featuremaps))
for i in range(featuremaps.shape[0]):
    map = featuremaps[i]
    matplotlib.image.imsave("output/stn_out_" + str(i) +".jpg", map, cmap = cm.gray)
    #np.savetxt('output/stn_out_' + str(i) + '.csv', map, delimiter=',')

# featuremaps = net.blobs['seg_conv3'].data[0]
# featuremaps = net.blobs['iter2_st_output'].data[0]
featuremaps = net.blobs['interp_iter2_st_seg_output'].data[0]
print ('iter2_st_output.shape:', featuremaps.shape, type(featuremaps))
for i in range(featuremaps.shape[0]):
    map = featuremaps[i]
    matplotlib.image.imsave("output/iter2_st_output_" + str(i) +".jpg", map, cmap = cm.gray)
    #np.savetxt('output/seg_output_' + str(i) + '.csv', map, delimiter=',')

# get result
res = net.blobs['reshape2'].data
print('res shape is:', res.shape, type(res))

char_set = []
with open('1s2n.list', 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        # line = line.strip('\n\r')
        line = line.replace('\n', '')
        if line is not "":
            char_set.append(str(line))
        line = f.readline()

print('char num:', len(char_set))

small_add = 1e-20 * np.ones(res.shape, dtype=np.float32)
res = res + small_add
output = np.log(res)
print('output shape is:', output.shape)

output_char_list = ''
pre_char_index = 6862
padding_char_index = 6862
space_char_index = 62
for i in range(output.shape[0]):
    data = output[i, :output.shape[1]]
    indexMap = np.argmax(data, axis=0)
    if indexMap != padding_char_index and pre_char_index!=indexMap: # and space_char_index!=indexMap:
        output_char_list = output_char_list + char_set[indexMap]

    pre_char_index = indexMap

# top_k = 10
# top_k_mat = np.zeros((output.shape[0], top_k), dtype=np.int32)
# for i in range(output.shape[0]):
#     data = output[i, :output.shape[1]]
#     top_k_idx = data.argsort()[::-1][0:top_k]
#     top_k_mat[i, :] = top_k_idx
#     # if indexMap != padding_char_index and pre_char_index!=indexMap: # and space_char_index!=indexMap:
#     #     output_char_list = output_char_list + char_set[indexMap]
#
#     # pre_char_index = indexMap
#
# for i in range(top_k_mat.shape[1]):
#     data_index = top_k_mat[:, i]
#     print(data_index.shape)


print(output_char_list)
#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
from multiprocessing import Process
import h5py
import sys
import argparse
import cv2
import os.path as osp
from multiprocessing import Pool

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = osp.dirname(__file__)

def write_image_info_into_file(file_name, data,alp):
    with open(file_name, 'w') as f:
        for k in range(0,len(data)):
            img = data[k][0]
            text = data[k][1]
            numbers = []
            for i in text:
                if alp.get(i) is not None:
                    numbers.append(str(alp[i]))
                else:
                    numbers.append(str(len(alp)))
            f.write(img  + "|" + ','.join(numbers) + "\n")

'''
def write_image_info_into_hdf5(file_name, data, phase,args,alp):
    total_size = len(data)
    print ('[+] total image for {0} is {1}'.format(file_name, len(data)))
    single_size = 10000
    groups = total_size // single_size
    if total_size % single_size:
        groups += 1
    def process(file_name, data):
        img_data = np.zeros((len(data), args.numChan, args.imgH, args.imgW), dtype = np.float32)
        label_seq = len(alp)*np.ones((len(data), args.seqLen), dtype = np.int)
        ratio = args.imgW/float(args.imgH)
        for k in range(0,len(data)):
            imgpath = data[k][0]
            text = data[k][1]
            numbers = []
            for i in text:
                if alp.get(i) is not None:
                    numbers.append(alp[i])
                else:
                    numbers.append(len(alp))
            labels = np.array(numbers)
            label_seq[k, :len(labels)] = labels
            img = caffe.io.load_image(imgpath)
            img = caffe.io.resize(img, (args.imgH,args.imgW, 3))
            img = np.transpose(img, (2, 0, 1))
            img_data[k] = img

        with h5py.File(file_name, 'w') as f:
            f.create_dataset('data', data = img_data)
            f.create_dataset('label', data = label_seq)

    with open(file_name, 'w', encoding='utf-8') as f:
        workspace = os.path.split(file_name)[0]
        process_pool = []
        for g in range(groups):
            h5_file_name = os.path.join(workspace, '%s_%d.h5' % (phase, g))
            f.write(h5_file_name + '\n')
            start_idx = g * single_size
            end_idx = start_idx + single_size
            if g == groups - 1:
                end_idx = len(data)
            p = Process(target=process, args=(h5_file_name, data[start_idx:end_idx]))
            p.start()
            process_pool.append(p)
        for p in process_pool:
            p.join()
'''


def process(args, alp, file_name_h5, data):
    img_data = np.zeros((len(data), args.numChan, args.imgH, args.imgW), dtype=np.float32)
    # label_seq = len(alp) * np.ones((len(data), args.seqLen), dtype=np.int)
    input_words = len(alp) * np.ones((len(data), args.seqLen), dtype=np.int)
    target_words = len(alp) * np.ones((len(data), args.seqLen), dtype=np.int)
    mask_data = np.zeros((len(data), 1, args.imgH, args.imgW), dtype=np.float32)
    loss_weight = np.ones((len(data), 1), dtype=np.float32)
    ratio = args.imgW / float(args.imgH)
    index = 0
    for k in range(0, len(data)):
        imgpath = data[k][0]
        text = data[k][1]
        maskpath = data[k][2]
        numbers = []
        for i in text:
            if alp.get(i) is not None:
                numbers.append(alp[i])
            else:
                print ('[+] |{0}| is not found in dict'.format(i))
                numbers.append(len(alp))
        labels = np.array(numbers)
        if args.numChan == 1:
            img = cv2.imread(imgpath)
            if img is None:
                print('cv2.imread is None', imgpath)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            #img = 255 - img
            flag = 0
            if float(img.shape[1]) / img.shape[0] < ratio:
                img = cv2.resize(img, (int(args.imgH * img.shape[1] / float(img.shape[0])), args.imgH))
                flag = 1
            else:
                img = cv2.resize(img, (args.imgW, args.imgH))
            if flag == 1:
                xzeros = np.zeros((args.imgH, args.imgW - img.shape[1]))
                img = np.hstack((img, xzeros))
            if img.ndim == 2:
                img = img[:, :, np.newaxis]
        else:
            img = cv2.imread(imgpath)
            if img is None:
                print('cv2.imread is None', imgpath)
                continue
            flag = 0
            if float(img.shape[1]) / img.shape[0] < ratio:
                img = cv2.resize(img, (int(args.imgH * img.shape[1] / float(img.shape[0])), args.imgH))
                flag = 1
            else:
                img = cv2.resize(img, (args.imgW, args.imgH))
            if flag == 1:
                xones = 0 * np.ones((args.imgH, args.imgW - img.shape[1], 3))
                img = np.hstack((img, xones))

            mask_img = cv2.imread(maskpath)
            if mask_img is None:
                print('cv2.imread is None', maskpath)
                continue
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
            #np.savetxt("mask_img.csv", mask_img, delimiter=',')
            #print('mask_img.shape', mask_img.shape)
            if mask_img is None:
                print('cv2.imread is None', maskpath)
                continue
            flag = 0
            if float(mask_img.shape[1]) / mask_img.shape[0] < ratio:
                mask_img = cv2.resize(mask_img, (int(args.imgH * mask_img.shape[1] / float(mask_img.shape[0])), args.imgH), interpolation = cv2.INTER_NEAREST)
                flag = 1
            else:
                mask_img = cv2.resize(mask_img, (args.imgW, args.imgH), interpolation = cv2.INTER_NEAREST)
            if flag == 1:
                xzeros = np.zeros((args.imgH, args.imgW - mask_img.shape[1]))
                mask_img = np.hstack((mask_img, xzeros))
            assert mask_img.ndim == 2
            mask_img = mask_img[:, :, np.newaxis]

        if args.imgH != img.shape[0] or args.imgW != img.shape[1]:
            print(img.shape)
            print("####")
            continue
        img = img / 255.
        img = np.transpose(img, (2, 0, 1))
        img_data[index] = img
        # label_seq[index, :len(labels)] = labels
        input_words[index, 1:min(args.seqLen, len(labels) + 1)] = labels[:args.seqLen-1]
        target_words[index, 0:min(args.seqLen, len(labels))] = labels[:args.seqLen]
        mask_img = np.transpose(mask_img, (2, 0, 1))
        mask_data[index] = mask_img
        loss_weight[index] = 1.0
        if imgpath.find('n10_cf') != -1:
            loss_weight[index] = 0.80
        index = index + 1

    img_seq = img_data[:index, :, :, :]
    # label_seq = label_seq[:index, :]
    input_words = input_words[:index, :]
    target_words = target_words[:index, :]
    mask_seq = mask_data[:index, :]
    loss_weight_seq = loss_weight[:index,:]
    with h5py.File(file_name_h5, 'w') as f:
        f.create_dataset('data', data=img_seq)
        # f.create_dataset('label', data=label_seq)
        f.create_dataset('input_words', data=input_words)
        f.create_dataset('target_words', data=target_words)
        f.create_dataset('seg_data', data=mask_seq)
        f.create_dataset('loss_weight', data=loss_weight_seq)

def write_image_info_into_hdf5_cv2(file_name, data, phase, args, alp):
    total_size = len(data)
    print ('[+] total image for {0} is {1}'.format(file_name, len(data)))
    single_size = 32000
    groups = total_size // single_size
    if total_size % single_size:
        groups += 1

    with open(file_name, 'a', encoding='utf-8') as f:
        workspace = os.path.split(file_name)[0]
        process_pool = []
        for g in range(groups):
            h5_file_name = os.path.join(workspace, '%d_%d.h5' % (phase, g))
            h5_file_name = h5_file_name.replace("\\", "/")
            f.write(h5_file_name + '\n')
            start_idx = g * single_size
            end_idx = start_idx + single_size
            if g == groups - 1:
                end_idx = len(data)
            p = Process(target=process, args=(args, alp, h5_file_name, data[start_idx:end_idx]))
            p.start()
            process_pool.append(p)
            #process(args, alp, h5_file_name, data[start_idx:end_idx])
        for p in process_pool:
            p.join()

def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataPath', type=str, required=True)
    parser.add_argument('--imgW', type=int,default=256)
    parser.add_argument('--imgH', type=int,default=64)
    parser.add_argument('--seqLen', type=int, default=16)
    parser.add_argument('--labelPath', type=str, default="annotations.list")
    parser.add_argument('--imgPath', type=str, default="images.list")
    parser.add_argument('--maskPath', type=str, default="segObjects.list")
    parser.add_argument('--dicPath', type=str, default="1s2n.list")
    parser.add_argument('--samplesPath', type=str, default="samples.list")
    parser.add_argument('--numChan', type=int, default=3)
    args = parser.parse_args(arguments)

    img_path = args.dataPath
    IMAGE_WIDTH, IMAGE_HEIGHT = args.imgW, args.imgH
    LABEL_SEQ_LEN = args.seqLen

    images, labels, masks = [], [], []
    # fi1 = open(args.imgPath, 'r', encoding='utf-8')
    # fi2 = open(args.labelPath, 'r', encoding='utf-8')
    # fi3 = open(args.maskPath, 'r', encoding='utf-8')
    # for (li1,li2,li3) in zip(fi1,fi2,fi3):
    #     li1 = li1.strip()
    #     li2 = li2.strip()
    #     li3 = li3.strip()
    #     #isExists = os.path.exists(li2)
    #     #if not isExists:
    #     #    continue
    #     label_txt = open(li2, 'r', encoding='utf-8')
    #     if label_txt is None:
    #         continue
    #     content = label_txt.read().strip()
    #     #print(content)
    #     if len(content)>args.seqLen:
    #         continue
    #     images.append(li1)
    #     labels.append(content)
    #     masks.append(li3)
    #     label_txt.close()
    # fi1.close()
    # fi2.close()
    # fi3.close()
    #
    # if len(images)!=len(labels):
    #     print ("labels length!=images length")
    #     raise AssertionError()
    #
    # if len(images)!=len(masks):
    #     print ("masks length!=images length")
    #     raise AssertionError()

    with open(args.samplesPath, 'r', encoding='utf-8') as f:
        lines = [line.strip('\n') for line in f.readlines()]

    for i, file_path in enumerate(lines):
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = [line.strip('\n') for line in f.readlines()]

        for k, sample in enumerate(samples):
            splits = sample.split('$--$')
            images.append(splits[0])
            labels.append(splits[1])
            masks.append(splits[2])

    fi4 = open(args.dicPath, 'r', encoding='utf-8')
    ind = 0
    alp = dict()
    for li in fi4:
        #li = li.strip()
        li = li.replace("\n", "")
        alp[li] = ind
        if li == " ":
            print(li, ind, alp[li])
        ind = ind + 1
    fi4.close()

    print ('[+] total image number: {}'.format(len(images)))
    print ('[+] total charater number: {}'.format(len(alp)))

    wdata = [(i,j,k) for (i,j,k) in zip(images,labels,masks)]
    np.random.shuffle(wdata)
    trainsize = 99*len(wdata)//100   # number of images for trainning
    train = wdata[:trainsize]
    test = wdata[trainsize:]
    isExists=os.path.exists(img_path)
    if not isExists:
        os.makedirs(img_path)

    # train_1 = train[:32000*40]
    # train_2 = train[32000 * 40:]
    # write_image_info_into_hdf5_cv2(os.path.join(img_path, 'training_1.list'), train_1, 'train_1', args, alp)
    # write_image_info_into_hdf5_cv2(os.path.join(img_path, 'training_2.list'), train_2, 'train_2', args, alp)

    single_size = 32000*7
    batch = len(train) // single_size
    if len(train) % single_size:
        batch += 1
    print('batch: ', batch)
    for b in range(batch):
        start_idx = b * single_size
        end_idx = start_idx + single_size
        if b == batch - 1:
            end_idx = len(train)
        train_set = train[start_idx:end_idx]
        write_image_info_into_hdf5_cv2(os.path.join(img_path, 'training.list'), train_set, b, args, alp)

    write_image_info_into_hdf5_cv2(os.path.join(img_path, 'testing.list'), test, 312, args, alp)
    # write_image_info_into_file(os.path.join(img_path, 'testing-images.list'), test, alp)

if __name__=="__main__":
    main(sys.argv[1:])

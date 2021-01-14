# coding: utf-8
#import utils
#import sys
#import imp
#imp.reload(sys)
#sys.setdefaultencoding('utf8')

from six import iteritems, itervalues, string_types
import os  
import glob
import logging
import numpy as np
from multiprocessing import Process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gather_samples(label_dir, image_dir, sample_num, extname='.jpg'):
    """TODO: Docstring for gather_samples.

    Args:
        label_dir: String. Directory save label txts
        image_dir: String. Directory save images.

    Returns: return samples List. Each element is (image_path, label_path)

    """
    labels = glob.glob(os.path.join(label_dir, '*.txt'))
    samples_paths = []
    num = 0
    for label_path in labels:
        path, label_name = os.path.split(label_path)
        image_name = label_name.replace('.txt', extname)
        image_path = os.path.join(image_dir, image_name)
        image_path = image_path.replace("\\", "/")
        label_path = label_path.replace("\\", "/")
        #isExists=os.path.exists(image_path)
        if num < sample_num:
            samples_paths.append((image_path, label_path))
            num = num + 1
    logger.info('gather samples of number:%d', len(samples_paths))
    return samples_paths
    
def gather_samples(label_dir, image_dir, mask_dir, sample_num, extname='.jpg'):
    """TODO: Docstring for gather_samples.

    Args:
        label_dir: String. Directory save label txts
        image_dir: String. Directory save images.

    Returns: return samples List. Each element is (image_path, label_path)

    """
    labels = glob.glob(os.path.join(label_dir, '*.txt'))
    samples_paths = []
    num = 0
    for label_path in labels:
        path, label_name = os.path.split(label_path)
        image_name = label_name.replace('.txt', extname)
        image_path = os.path.join(image_dir, image_name)
        mask_name = label_name.replace('.txt', '.png')
        mask_path = os.path.join(mask_dir, mask_name)
        image_path = image_path.replace("\\", "/")
        label_path = label_path.replace("\\", "/")
        mask_path = mask_path.replace("\\", "/")
        #isExists=os.path.exists(image_path)
        if num<sample_num:
            samples_paths.append((image_path, label_path, mask_path))
            num = num + 1
    logger.info('gather samples of number:%d', len(samples_paths))
    return samples_paths

def write_samples(samples, labelPath, imgPath): 
    outfile_label = open(labelPath, 'w', encoding='utf-8')
    outfile_img = open(imgPath, 'w', encoding='utf-8')
    for i, sample_path in enumerate(samples):
            outfile_img.write(sample_path[0]+'\n')
            outfile_label.write(sample_path[1]+'\n')
    outfile_label.close()
    outfile_img.close()

def write_samples(samples, labelPath, imgPath, maskPath):
    outfile_label = open(labelPath, 'w', encoding='utf-8')
    outfile_img = open(imgPath, 'w', encoding='utf-8')
    outfile_mask = open(maskPath, 'w', encoding='utf-8')
    for i, sample_path in enumerate(samples):
            outfile_img.write(sample_path[0]+'\n')
            outfile_label.write(sample_path[1]+'\n')
            outfile_mask.write(sample_path[2] + '\n')
    outfile_label.close()
    outfile_img.close()
    outfile_mask.close()

def output_label2cls(label2cls, outfilename):
    if len(label2cls)>0:
        outrect_list = open(outfilename, 'w', encoding='utf-8')
        for label_name, index in iteritems(label2cls):
            label_info = '%s\n' % (label_name)
            outrect_list.write(label_info)
        outrect_list.close()

def sortedDictValues1(adict): 
    items = adict.items() 
    items.sort() 
    return [value for key, value in items] 
    
    
def output_sorted_label2cls(label2cls, outfilename):
    if len(label2cls)>0:
        outrect_list = open(outfilename, 'w', encoding='utf-8')
        label_list = sorted(label2cls.items(), key=lambda d:d[0])
        for label, index in label_list:
            label_info = '%s\n' % (label)
            outrect_list.write(label_info)
        outrect_list.close()

def process(data, file_name):
    #input_images, input_labels, input_masks = [], [], []
    with open(file_name, 'w', encoding='utf-8') as f:
        for k in range(0, len(data)):
            li1 = data[k][0]
            li2 = data[k][1]
            li3 = data[k][2]
            li1 = li1.strip()
            li2 = li2.strip()
            li3 = li3.strip()
            # isExists = os.path.exists(li2)
            # if not isExists:
            #    continue
            label_txt = open(li2, 'r', encoding='utf-8')
            if label_txt is None:
                continue
            # content = label_txt.read().strip()
            content = label_txt.read().replace("\n", "")
            if len(content) > 16:
                continue
            # input_images.append(li1)
            # input_labels.append(content)
            # input_masks.append(li3)
            label_txt.close()

            sample_info = '%s$--$%s$--$%s\n' % (li1, content, li3)
            f.write(sample_info)

if __name__ == '__main__':
    args_imagenet = {
        'image_dir': './synthetic_chinese_dataset/synthetic_data_6862_n10_imagenet/train_part_images',
        'annotation_dir': './synthetic_chinese_dataset/synthetic_data_6862_n10_imagenet/annotations',
        'mask_dir': './synthetic_chinese_dataset/synthetic_data_6862_n10_imagenet/segObjects',
        'img_shape': [64, 256],
        'label_length': 16,
        'save_path': './h5_space',
        'phase': 'train',
        'sample_num': 400000
    }
    sample_paths = gather_samples(
        label_dir=args_imagenet.get('annotation_dir'), image_dir=args_imagenet.get('image_dir'),
        mask_dir=args_imagenet.get('mask_dir'), sample_num=args_imagenet.get('sample_num'))
    # sample_paths.extend(sample_paths_imagenet)

    args_coco = {
        'image_dir': './synthetic_chinese_dataset/synthetic_data_6862_n10_coco/train_part_images',
        'annotation_dir': './synthetic_chinese_dataset/synthetic_data_6862_n10_coco/annotations',
        'mask_dir': './synthetic_chinese_dataset/synthetic_data_6862_n10_coco/segObjects',
        'img_shape': [64, 256],
        'label_length': 16,
        'save_path': './h5_space',
        'phase': 'train',
        'sample_num': 400000
    }
    sample_paths_coco = gather_samples(
        label_dir=args_coco.get('annotation_dir'), image_dir=args_coco.get('image_dir'),
        mask_dir=args_coco.get('mask_dir'), sample_num=args_coco.get('sample_num'))
    sample_paths.extend(sample_paths_coco)
    
    np.random.shuffle(sample_paths)
    write_samples(sample_paths, labelPath="annotations.list", imgPath="images.list", maskPath="segObjects.list")

    images, labels, masks = [], [], []
    fi1 = open("images.list", 'r', encoding='utf-8')
    fi2 = open("annotations.list", 'r', encoding='utf-8')
    fi3 = open("segObjects.list", 'r', encoding='utf-8')

    raw_data = [(i, j, k) for (i, j, k) in zip(fi1, fi2, fi3)]
    total_size = len(raw_data)
    single_size = 150000
    groups = total_size // single_size
    if total_size % single_size:
        groups += 1

    sample_dir = os.path.join("sample_space")
    if os.path.exists(sample_dir) is False:
        os.mkdir(sample_dir)

    process_pool = []
    with open('samples.list', 'w', encoding='utf-8') as f:
        for g in range(groups):
            sample_file_name = os.path.join("sample_space", 'sample_%d.list' % (g))
            sample_file_name = sample_file_name.replace("\\", "/")
            f.write(sample_file_name + '\n')

            start_idx = g * single_size
            end_idx = start_idx + single_size
            if g == groups - 1:
                end_idx = len(raw_data)
            p = Process(target=process, args=(raw_data[start_idx:end_idx], sample_file_name))
            p.start()
            process_pool.append(p)
            # process(args, alp, h5_file_name, data[start_idx:end_idx])
        for p in process_pool:
            p.join()

    fi1.close()
    fi2.close()
    fi3.close()

    # for (li1, li2, li3) in zip(fi1, fi2, fi3):
    #     li1 = li1.strip()
    #     li2 = li2.strip()
    #     li3 = li3.strip()
    #     # isExists = os.path.exists(li2)
    #     # if not isExists:
    #     #    continue
    #     label_txt = open(li2, 'r', encoding='utf-8')
    #     if label_txt is None:
    #         continue
    #     content = label_txt.read().strip()
    #     # print(content)
    #     if len(content) > 16:
    #         continue
    #     images.append(li1)
    #     labels.append(content)
    #     masks.append(li3)
    #     label_txt.close()
    # fi1.close()
    # fi2.close()
    # fi3.close()

    # if len(images)!=len(labels):
    #     print ("labels length!=images length")
    #     raise AssertionError()
    #
    # if len(images)!=len(masks):
    #     print ("masks length!=images length")
    #     raise AssertionError()
    #
    # wdata = [(i, j, k) for (i, j, k) in zip(images, labels, masks)]
    # with open('samples.list', 'w', encoding='utf-8') as out_list:
    #     for k in range(0, len(wdata)):
    #         imgpath = wdata[k][0]
    #         text = wdata[k][1]
    #         maskpath = wdata[k][2]
    #         sample_info = '%s$$%s$$%s\n' % (imgpath, text, maskpath)
    #         #print(sample_info)
    #         out_list.write(sample_info)

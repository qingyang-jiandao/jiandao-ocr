#/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import glob
import random
import codecs
import pathlib
import math
import PIL
from PIL import Image, ImageDraw
import numpy as np
import cv2
import Aug_Operations as aug
from ConvertRGBA2RGB import alpha_composite, alpha_to_color

def augmentImage_vp(txt_img, points, rot_degree=0, skew=aug.Skew(1, 'TILT', 0.5), borderValue=(255, 255, 255, 0), type=-1):
    # Augment rate for eatch type
    if type == -1:
        type = random.randint(0, 3)

    if type == 0:
        #degee = random.randint(0, 10)
        #rot_degree = random.randint(-degee, degee)
        txt_img, points = rotate_img(txt_img, rot_degree, borderValue)
    elif type == 1:
        #skew = aug.Skew(1, 'RANDOM', 0.5)
        txt_img, points = skew.perform_operation(txt_img, borderValue)
    elif type == 2:
        # shear = aug.Shear(1., 5, 5)
        shear = aug.Shear(1., 2, 2)
        txt_img, points = shear.perform_operation(txt_img, borderValue)
    #elif type == 3:
    #    distort = aug.Distort(1.0, 4, 4, 1)
    #    txt_img = distort.perform_operation(txt_img)
    return txt_img, points 

def augmentImage(txt_img, points, rot_degree=0, skew=aug.Skew(1, 'TILT', 0.5), borderValue=(255, 255, 255, 0), type=-1):
    # Augment rate for eatch type
    if type == -1:
        type = random.randint(0, 5)

    if type == 0:
        txt_img, points = rotate_img(txt_img, rot_degree, borderValue)
    elif type == 1:
        txt_img, points = skew.perform_operation(txt_img, borderValue)
    elif type == 2:
        shear = aug.Shear(1., 5, 5)
        txt_img, points = shear.perform_operation(txt_img, borderValue)
    #elif type == 3 :
    #    distort = aug.Distort(1.0, 4, 4, 1)
    #    txt_img = distort.perform_operation(txt_img)
    return txt_img, points

def getTopNCharacters2Dict(characters_file_path, top_n):
    id_cha_dict = {}
    characters = getAllCharactersFromFile(characters_file_path)
    print('character number: %s' % len(characters))
    for index, char in enumerate(characters):
        id_cha_dict[index] = char
        #if index == top_n-1: break
    return id_cha_dict

def getAllCharactersFromFile(characters_file_path):
    #characters_set = set()
    characters_set = []
    with codecs.open(characters_file_path, encoding="utf-8") as file:
        lines = file.readlines()
        count = 0
        for oneline in lines:
            cont = oneline
            #oneline.replace('\r\n', '')
            #oneline.replace('\n', '')
            #oneline.replace('\t', '')
            oneline = oneline.strip('\n')
            if oneline is not '':
                count = count+1
                #characters_set.update(list(oneline))
                characters_set.append(oneline)
            else:
                print('[error]cont: %s index: %s' % (cont, count))
        print('lines: %s valid: %s' % (len(lines), count))
        print('character num: %s' % (len(characters_set)))
    #return list(characters_set)
    return characters_set

def makeDirectory(path):
    if os.path.exists(path):
        print('The path exists: %s'%path)
    else:
        os.makedirs(path)
        print('The path maked: %s'%path)

def getBackgroundListFromDir(background_dir, ext_name = '*.jpg'):
    image_path_list = []
    image_path_list.extend([path for path in pathlib.Path(background_dir).rglob(ext_name)])
    # image_path_list.extend([path for path in pathlib.Path(background_dir).rglob('*.JPEG')])
    print('Load backgroudn image: %d'%len(image_path_list))
    return image_path_list

def getFontListFromDir(font_dir):
    font_path_list = []
    font_path_list.extend(glob.glob(os.path.join(font_dir, '*.[t,T][t,T][f,F,c,C]')))
    print('Load font files: %d'%len(font_path_list))
    return font_path_list

def get_content(id_character_dict, length_range_tuple):
    length = len(id_character_dict)
    rand_len = random.randint(length_range_tuple[0], length_range_tuple[1])
    content = u''
    content_index = []
    for i in range(rand_len):
        rand_index = random.randint(0, length-1)
        content += id_character_dict[rand_index]
        content_index.append(rand_index)
    return content, content_index

def get_contents(id_character_dict, length_range_tuple, line_number=2):
    contents, contents_index = [], []
    for i in range(line_number):
        content, content_index = get_content(id_character_dict, length_range_tuple)
        contents.append(content)
        contents_index.append(content_index)
    return contents, contents_index

def saveImage2Dir(image, image_save_dir, image_name='test_image'):
    if type(image) == list:
        for index, one_image in enumerate(image):
            saveImage2Dir(one_image, image_save_dir, image_name=image_name+'_'+str(index))
    else:
        image_save_path = os.path.join(image_save_dir, image_name+'.jpg')
        image.save(image_save_path)

def rotate_img(image, degree, border_color):
    img = np.array(image)
    height, width = img.shape[:2]

    heightNew = int(width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
    widthNew = int(height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))
    #print('heightNew: %d, widthNew: %d' % (heightNew, widthNew))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2  
    matRotation[1, 2] += (heightNew - height) / 2
    #print('matRotation', matRotation)
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=border_color)
    #imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0, 0))

    w = width
    h = height
    points = np.matrix([[-w / 2, -h / 2, 1], [-w / 2, h / 2, 1], [w / 2, h / 2, 1], [w / 2, -h / 2, 1]])
    matRotation = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)
    matRotation[0, 2] = widthNew / 2
    matRotation[1, 2] = heightNew / 2
    #print('matRotation', matRotation)

    p = matRotation * points.T

    if imgRotation.ndim==4:
        for row in imgRotation:
            for element in row:
                if element[3] == 0:
                    for i in range(3):
                        element[i] = 0
                        #element[i] = 255
    image = Image.fromarray(imgRotation)
    points = np.array(p.T, int)
    return image, points

def mergeImageAtPoint(image, txt_img, left_top_point):
    left, top = left_top_point

    image = image.convert('RGBA')
    image.paste(txt_img, (left, top), txt_img)
    res_img = image.convert('RGB')
    return res_img

def mergeBgimgAndTxtimgPoints(left_center, points):
    left, center_line = left_center
    top = center_line - (max(points[:, 1]) - min(points[:, 1]))/2
    for k in range(0, 4):
        points[k, :] += np.array([left, top], dtype=np.int32)
    return points

def setColor(roi_img):
    if type(roi_img) != np.ndarray:
        roi_img = np.array(roi_img)
    color1 = cv2.mean(roi_img)
    color = np.zeros(3, dtype=np.int)
    color[0] = math.ceil(color1[2])
    color[2] = math.ceil(color1[1])
    color[1] = math.ceil(color1[0])
    for _ in range(0,3):
        s = random.randint(0, 2)
        if color[s] > 150:
            color[s] = random.randint(0,20)
        else:
            color[s] = random.randint(230,255)
    return color

def saveIdCharacterDict2File(id_character_dict, save_path):
    with codecs.open(save_path, 'w', encoding='utf-8') as file:
        #character_list = sorted(id_character_dict.items(), key=lambda d: d[1])
        #for key, val in character_list:
        for (key, val) in id_character_dict.items():
            if val is not '\n':
                #file.write(val + ' ' + str(key) + '\n')  # the first index must be zero
                file.write(val+'\n')  # the first index must be zero
    return

def drawMulContentsRectangle(image, mulcontents_points):
    draw = ImageDraw.Draw(image)
    for content_points in mulcontents_points:
        for point in content_points:
            draw.line([tuple(point[0]), tuple(point[1]), tuple(point[2]), tuple(point[3]), tuple(point[0])])
            # draw.rectangle((tuple(point[0]), tuple(point[2])))
    del draw
    return image

def getRandomOneFromList(list):
    return list[random.randint(0, len(list)-1)]

def getPointByCenterLine(center_line, left_top_point, width, height):
    left, _ = left_top_point
    points = [(left, center_line - height/2), 
           (left + width, center_line - height/2), 
           (left + width, center_line + height/2), 
           (left, center_line + height/2)]
    return points

def getNewLeftCenterPointByContentPoints(content_points):
    left = content_points[0][0][0]
    bottom_line = getTopOrBottomLineInPoints(content_points, is_top=0)
    top_line = getTopOrBottomLineInPoints(content_points, is_top=1)
    height = bottom_line - top_line
    return (left, bottom_line+height/2)

def getTopOrBottomLineInPoints(points, is_top):
    list = []
    if is_top:
        for i in points:
            list.append(min(i[:, 1]))
        return min(list)
    else:
        for i in points:
            list.append(max(i[:, 1]))
        return max(list)
    
def getPointsOfImageRectangle(w, h):
    #return np.array([[-w / 2, -h / 2], [-w / 2, h / 2], [w / 2, h / 2], [w / 2, -h / 2]])
    return np.array([[0, 0], [0, h], [w, h], [w, 0]])

def mergeImage(image, front_img, color, offset_x=0, offset_y=0):
    bg_w, bg_h = image.size
    image = image.convert('RGB')
    image_arr = np.array(image)
    w, h = front_img.size
    assert offset_y+h<=bg_h and offset_x+w<=bg_w, '%s %s %s %s' % (offset_y+h, bg_h, offset_x+w, bg_w)
    image_roi = image_arr[offset_y:offset_y+h, offset_x:offset_x+w, :]
    front_img_arr = np.array(front_img.convert('L'))
    mask = front_img_arr[:, :]  # channel R
    #np.savetxt('alpha_mask' + '.csv', mask, delimiter=',')
    mask_index = mask > 0
    image_roi[mask_index] = color
    image_arr[offset_y:offset_y+h, offset_x:offset_x+w:, :] = image_roi
    res_img = Image.fromarray(image_arr)

    # image = image.convert('RGBA')
    # # convert black to transpant.
    # front_img = front_img.convert('RGBA')
    # x = np.array(front_img)
    # r, g, b, a = np.rollaxis(x, axis=-1)
    # # mask = ((r <= 20) & (g <= 20) & (b <= 20))
    # mask = ((r == 0) & (g == 0) & (b == 0))
    # x[mask, 3] = 0
    # front_img = Image.fromarray(x, 'RGBA')
    # image.paste(front_img, (0, 0), front_img)
    # res_img = image.convert('RGB')

    return res_img

def mergeImageByTrans(image, transformation, bg_image, isAug = True):
    if bg_image is None:
        return

    cropped_image = image
    rot_degree = random.randint(-15, 15)
    skew = aug.Skew(1, 'RANDOM', 0.10)
    width, height = cropped_image.size
    corner_points = getPointsOfImageRectangle(width, height)
    if isAug:
        aug_image, corner_points = augmentImage(cropped_image, corner_points, rot_degree, skew, borderValue=(0, 0, 0, 0))
    else:
        aug_image = cropped_image

    bg_width, bg_height = bg_image.size
    aug_width, aug_height = aug_image.size
    if bg_width<aug_width or bg_height<aug_height:
        return

    bg_width, bg_height = bg_image.size
    # offset_x = random.randint(0, bg_width-aug_width)
    # offset_y = random.randint(0, bg_height-aug_height)
    offset_x = transformation[0]
    offset_y = transformation[1]
    box = (offset_x, offset_y, offset_x+aug_width, offset_y+aug_height)
    roi_img = bg_image.crop(box)

    # color = np.zeros(3, dtype=np.int)
    # for _ in range(0, 3):
    #     s = random.randint(0, 2)
    #     if color[s] > 150:
    #         color[s] = random.randint(0, 20)
    #     else:
    #         color[s] = random.randint(230, 255)

    color = np.zeros(3, dtype=np.int)
    for i in range(0, 3):
        color[i] = random.randint(0, 255)

    comp_img = mergeImage(roi_img, aug_image, color).convert('RGBA')
    bg_image.paste(comp_img, (offset_x, offset_y), comp_img)

def mergeAndTransformImage(image, transformation, bg_image, is_aug=True):
    if bg_image is None:
        return None, None, None

    cropped_image = image

    # rot_degree = random.randint(-15, 15)
    # rot_degree = random.randint(-5, 5)
    rot_degree = random.randint(-2, 2)
    # skew = aug.Skew(1, 'RANDOM', 0.10)
    skew = aug.Skew(1, 'RANDOM', 0.05)
    width, height = cropped_image.size
    corner_points = getPointsOfImageRectangle(width, height)
    if is_aug:
        aug_image, corner_points = augmentImage(cropped_image, corner_points, rot_degree, skew, borderValue=(0, 0, 0, 0))
    else:
        aug_image = cropped_image

    bg_width, bg_height = bg_image.size
    aug_width, aug_height = aug_image.size

    offset_x = transformation[0]
    offset_y = transformation[1]
    rand_ex_x = transformation[2]
    rand_ex_y = transformation[3]
    target_x = aug_width+rand_ex_x
    target_y = aug_height+rand_ex_y

    if bg_width < target_x or bg_height < target_y:
        return None, None, None

    box = (offset_x, offset_y, offset_x+target_x, offset_y+target_y)
    roi_img = bg_image.crop(box)

    # color = np.zeros(3, dtype=np.int)
    # for _ in range(0, 3):
    #     s = random.randint(0, 2)
    #     if color[s] > 150:
    #         color[s] = random.randint(0, 20)
    #     else:
    #         color[s] = random.randint(230, 255)

    color = np.zeros(3, dtype=np.int)
    for i in range(0, 3):
        color[i] = random.randint(0, 255)

    off_x = random.randint(0, rand_ex_x)
    off_y = random.randint(0, rand_ex_y)
    comp_img = mergeImage(roi_img, aug_image, color, offset_x=off_x, offset_y=off_y)

    bin_image = Image.new('RGB', (target_x, target_y), (0, 0, 0))
    if bin_image is None or bin_image.size[0] <= 0 or bin_image.size[1] <= 0:
        return None, None, None
    bin_image_arr = np.array(bin_image)
    cropped_image_arr = np.array(cropped_image)
    # print(bin_image_arr.shape, cropped_image_arr.shape, aug_image.size)
    bin_image_arr[off_y:off_y + cropped_image_arr.shape[0], off_x:off_x + cropped_image_arr.shape[1], :] = cropped_image_arr
    bin_image = Image.fromarray(bin_image_arr)

    return comp_img, bin_image, (offset_x, offset_y, target_x, target_y)

def addRandomInROI(roi):
    left, top, right, bottom = (roi[0][0],roi[0][1],roi[2][0],roi[2][1])
    max_randint = (bottom - top) / 6
    new_left = random.randint(int(left-max_randint), int(left+max_randint/2))
    new_top = random.randint(int(top-max_randint), int(top+max_randint/2))
    new_right = random.randint(int(right-max_randint/2), int(right+max_randint))
    new_bottom = random.randint(int(bottom), int(bottom+2*max_randint))
    return (new_left, new_top, new_right, new_bottom)
    
def getOneLineRectanglePoints(one_line_points):
    top = getTopOrBottomLineInPoints(one_line_points, is_top=1)
    bottom = getTopOrBottomLineInPoints(one_line_points, is_top=0)
    left = min(one_line_points[0][0][0], one_line_points[0][3][0])
    right = max(one_line_points[-1][1][0], one_line_points[-1][2][0])
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]])

def findMaxIndex(image_list):
    index_list = []
    #for image_path in image_list:
    #    index_list.append(int(image_path.split(os.path.sep)[-1].split('_')[0]))

    index_list = [int(image_path.split(os.path.sep)[-1].split('_')[0]) for image_path in image_list]
    index_list = sorted(index_list)
    return index_list[-1]

def bb_overlab(x1, y1, w1, h1, x2, y2, w2, h2):
    if(x1>x2+w2):
        return 0
    if(y1>y2+h2):
        return 0
    if(x1+w1<x2):
        return 0
    if(y1+h1<y2):
        return 0
    colInt = abs(min(x1 +w1 ,x2+w2) - max(x1, x2))
    rowInt = abs(min(y1 + h1, y2 +h2) - max(y1, y2))
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    return overlap_area / (area1 + area2 - overlap_area)

def test_collision(marked_boxes, box_x, box_y, box_w, box_h):
    for box in marked_boxes:
        top_x, top_y, width, height = box
        iou = bb_overlab(box_x, box_y, box_w, box_h, top_x, top_y, width, height)
        if iou<0.01:
            continue
        else:
            return 1

    return 0
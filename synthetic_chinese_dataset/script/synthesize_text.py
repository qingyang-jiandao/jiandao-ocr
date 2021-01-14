#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
from PIL import Image, ImageDraw, ImageFont
import utils 
import codecs
import tqdm
import glob
import add_effects as AE
import numpy as np
import Aug_Operations as aug

def main():
    root_dir = './synthetic_text_n10'
    data = OCRData(root_dir)
    data.args['classes_number'] = 6862
    #data.args['id_character_file_path'] = os.path.basename(data.args['characters_file_path']).split('.')[
    #                                          0] + '_top_%d.txt' % data.args['classes_number']

    data.makeNeededDir()
    if os.path.exists(root_dir):
        data.collectFileNum()
    print('The arguments :')
    print(data.args)
    
    #data.saveTopNCharacters2File(data.args['characters_file_path'], data.args['classes_number'], data.args['id_character_file_path'])
    data.args['image_number'] = data.args['classes_number']*1
    data.bg_img_list = utils.getBackgroundListFromDir(data.args['background_image_dir'])
    data.args['image_number'] = 1000 #len(data.bg_img_list)
    data.synthesizeAllImages(data.args['image_number'])

class OCRData(object):
    def __init__(self, root_dir):
        self.args = {}
        self.args['root_dir'] = root_dir
        args = self.setArguments()

        self.wordict = []
        word_list = open(self.args['characters_file_path'], 'r', encoding='utf-8')
        for w in word_list:
            w = w.replace("\n", "")
            self.wordict.append(w)
        word_list.close()
        self.alpha = self.wordict[0:52]
        self.num = self.wordict[52:62]
        self.punt = self.wordict[62:100]
        self.f1chi = self.wordict[100:3855]
        self.f2chi = self.wordict[3855:]
        self.alphanum = self.alpha + self.num
        self.space = self.wordict[62]
        self.id = 0

    def setArguments(self):
        self.args['characters_length_tuple'] = (4, 10)
        self.args['validation_rate'] = 0.0
        self.args['test_rate'] = 0.0
        #self.args['background_image_dir'] = '/home/tgc/workspace/deeplab/database/benchmark/benchmark_RELEASE/dataset/img'
        #self.args['background_image_dir'] = '/home/tgc/workspace/deeplab/database/train2014'
        #self.args['background_image_dir'] = 'D:/tgc/workspace/deeplab/database/train2014'
        self.args['background_image_dir'] = 'F:/database/VOC/benchmark/benchmark_RELEASE/dataset/img'
        self.args['fonts_dir'] = '../fonts_all'
        self.args['characters_file_path'] = '1s2n.list'
        self.args['classes_number'] = 6862
        #self.args['id_character_file_path'] = os.path.basename(self.args['characters_file_path']).split('.')[0] + '_top_%d.txt'%self.args['classes_number']
        self.args['font_size_min'] = 24
        self.args['font_size_max'] = 48
        self.args['image_number'] = 100
        self.args['save_full_image'] = 0
        self.args['add_rectangle'] = 0
        self.args['lines_num'] = 10
        return self.args

    def get_content(self, length_range_tuple):
        rand_len = random.randint(length_range_tuple[0], length_range_tuple[1])
        # print('space', len(self.space[0]))
        content = u''
        while True:
            # if np.random.binomial(1, 0.85):  # 85%汉字
            if np.random.binomial(1, 0.75):  # 50%汉字
                for i in range(rand_len):
                    space_op = random.randint(0, 50)
                    char = u''
                    if np.random.binomial(1, 0.96):  # 96%一级二级字
                        if np.random.binomial(1, 0.7):  # 70%一级字
                            index = np.random.randint(0, len(self.f1chi))
                            char = self.f1chi[index]
                            if space_op == 0:
                                char = self.space[0]
                                # print(char, 'char', len(self.space[0]), len(char))
                            content = content + char
                            #print("1", char, index)
                        else:  # 30%二级字
                            index = np.random.randint(0, len(self.f2chi))
                            char = self.f2chi[index]
                            if space_op == 0:
                                char = self.space[0]
                            content = content + char
                            #print("2", char, index)
                    else:  # 4%字母数字标点符号
                        if np.random.binomial(1, 0.5):  # 50%字母数字
                            index = np.random.randint(0, len(self.alphanum))
                            char = self.alphanum[index]
                            if space_op == 0:
                                char = self.space[0]
                            content = content + char
                            #print("3", char, index)
                        else:  # 50%标点符号
                            index = np.random.randint(0, len(self.punt))
                            char = self.punt[index]
                            content = content + char
                            #print("4", char, index)
                #print(content, len(content), rand_len, "xx")
            else:
                if np.random.binomial(1, 0.70):
                    for i in range(rand_len):
                        space_op = random.randint(0, 100)
                        if np.random.binomial(1, 0.99):
                            index = np.random.randint(0, len(self.alphanum))
                            char = self.alphanum[index]
                            if space_op == 0:
                                char = self.space[0]
                            content = content + char
                        else:
                            index = np.random.randint(0, len(self.punt))
                            content = content + self.punt[index]
                else:
                    length_range_tuple[0]=2
                    length_range_tuple[1]=8
                    rand_len = random.randint(length_range_tuple[0], length_range_tuple[1])
                    for i in range(rand_len):
                        space_op = random.randint(0, 100)
                        index = np.random.randint(0, len(self.num))
                        char = self.num[index]
                        if space_op == 0:
                            char = self.space[0]
                        content = content + char
                #print(content, len(content), rand_len, "x")
            if len(content) < length_range_tuple[0] or len(content) > length_range_tuple[1]:
                print(content, len(content))
            assert len(content)>=length_range_tuple[0], '%d %d' % (len(content), length_range_tuple[0])
            assert len(content) <= length_range_tuple[1], '%d %d' % (len(content), length_range_tuple[1])
            if len(content)>=length_range_tuple[0]:
                break
        return content

    def get_contents(self, length_range_tuple, line_number=2):
        contents, contents_index = [], []
        for i in range(line_number):
            content = self.get_content(list(length_range_tuple))
            contents.append(content)
        return contents, contents_index

    def makeNeededDir(self):
        utils.makeDirectory(self.args['root_dir'])
        self.makePartDirs('train')
        # if self.args['validation_rate'] > 0:
        self.makePartDirs('validation')
        # if self.args['test_rate'] > 0:
        self.makePartDirs('test')
        self.args['annotations_dir'] = os.path.join(self.args['root_dir'], 'annotations')
        utils.makeDirectory(self.args['annotations_dir'])
        self.args['label_dir'] = os.path.join(self.args['root_dir'], 'labels')
        utils.makeDirectory(self.args['label_dir'])
        return

    def makePartDirs(self, part_role):
        if self.args['save_full_image']:
            self.args[''.join([part_role,'_image_dir'])] = os.path.join(self.args['root_dir'], part_role+'_image')
            utils.makeDirectory(self.args[part_role+'_image_dir'])
        self.args[part_role+'_part_image_dir'] = os.path.join(self.args['root_dir'], part_role+'_part_image')
        utils.makeDirectory(self.args[part_role+'_part_image_dir'])
        return

    def generator(self, sample, thread_id):
        contents, content_indexs, background_image_path, font_path, img_number = sample
        background_image = Image.open(background_image_path).convert('RGBA')
        bg_width, bg_height = background_image.size
        roi_images = []
        roi_boxes = []
        for i, content in enumerate(contents):
            # curr_img_number = img_number*self.args['lines_num'] + i
            image, points = self.putContent2Image(content, font_path, self.args['add_rectangle'])
            if image == None or points == None or len(points) == 0:
                return
            if self.args['save_full_image']:
                self.saveImage(image, img_number)

            text_img_width, text_img_height = image.size
            if bg_width < text_img_width or bg_height < text_img_height:
                continue

            transform = []
            for loop_number in range(10):
                offset_x = random.randint(0, bg_width - text_img_width)
                offset_y = random.randint(0, bg_height - text_img_height)
                if utils.test_collision(roi_boxes, offset_x, offset_y, text_img_width,
                                        text_img_height) == 0:
                    roi_box = (offset_x, offset_y, text_img_width, text_img_height)
                    transform = [offset_x, offset_y, text_img_width, text_img_height]
                    break

            if len(transform) > 0:
                utils.mergeImageByTrans(image, transform, background_image)
                roi_boxes.append(roi_box)
            # self.saveImage(roi_images, img_number, is_part=1)
        roi_images.append(background_image.convert('RGB'))
        self.saveImage(roi_images, img_number, is_part=1)

    def synthesizeAllImages(self, image_number):
        self.font_list = utils.getFontListFromDir(self.args['fonts_dir'])
        start_index = self.restoreFromPartImageDir()
        for i in tqdm.tqdm(range(start_index, image_number)):
            #contents, content_indexs = utils.get_contents(self.id_character_dict, self.args['characters_length_tuple'], self.args['lines_num'])
            contents, content_indexs = self.get_contents(self.args['characters_length_tuple'], self.args['lines_num'])
            background_image_path, font_path = map(utils.getRandomOneFromList, [self.bg_img_list, self.font_list])
            samples.append((contents, content_indexs, background_image_path, font_path, i))

        assert (image_number-start_index)==len(samples)
        from generate_image_base import Process
        num_workers = 4
        Process(samples, inEnlarge=False, num_image=image_number-start_index, callback=self.generator, epochs=1,
                input_workers=num_workers)
        return
    
    
    def saveImage(self, images, image_index, is_part=0):
        image_save_dir = self.chooseSaveDirByIndex(image_index, is_part)
        utils.saveImage2Dir(images, image_save_dir, image_name=str(image_index))

    
    def saveAnnotation(self, contents, image_index):
        #for index, one_content in enumerate(content_indexs):
        for index, one_content in enumerate(contents):
            ann_name = ''.join([str(image_index), '_', str(index), '.txt'])
            ann_path = os.path.join(self.args['annotations_dir'], ann_name)
            with codecs.open(ann_path, 'w', encoding='utf-8') as file:
                #file.write(' '.join([ann_name.split('.')[0], font_path, str(rectangle_points.tolist()), str(one_content)]))
                file.write(' '.join([one_content]))

    def saveLabel(self, images, image_index):
        for index, image in enumerate(images):
            label_name = ''.join([str(image_index), '_', str(index), '.png'])
            label_path = os.path.join(self.args['label_dir'], label_name)
            seg_array = np.array(image.convert("L"), dtype=np.float32)
            #np.savetxt('image_np' + '.csv', seg_array, delimiter=',')
            mask = seg_array > 0
            seg_array[mask] = 1
            label_image = Image.fromarray(seg_array.astype(np.uint8))
            label_image.save(label_path)
            
    def putContent2Image(self, content, font_path, add_rectangle=0, resize_rate=2):
        #image = Image.open(background_image_path)
        #image = Image.new('RGBA', (512, 512), (255, 255, 255, 0))

        image = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
        if image==None or image.size[0]<=0 or image.size[1]<=0:
            return None, None
        font_size_max = image.size[0]/self.args['characters_length_tuple'][1]
        while font_size_max < self.args['font_size_min']:
            resize_rate = resize_rate * 2
            image = image.resize((image.size[0]*resize_rate, image.size[1]*resize_rate))
        font_size_max = self.args['font_size_max']
        if image == None or image.size[0] <= 0 or image.size[1] <= 0:
            return None, None

        # print('font_size_max', font_size_max)

        font_size = random.randint(self.args['font_size_min'], int(font_size_max))
        if int(image.size[0] - font_size * len(content)) < 0 or \
                int(image.size[1] - font_size) < font_size:
            return None, None

        offset_x = random.randint(0, int(image.size[0] - font_size * len(content)))
        offset_y = random.randint(font_size, int(image.size[1] - font_size))
        offset_xy = (offset_x, offset_y)

        color = np.zeros(3, dtype=np.int)
        for _ in range(0, 3):
            s = random.randint(0, 2)
            if color[s] > 150:
                color[s] = random.randint(0, 20)
            else:
                color[s] = random.randint(230, 255)

        content_points = []
        for character in content:
            image, points = self.putOneCharacter2Image(character, image, font_path, font_size, offset_xy, color)
            if image is None:
                print('####error: there are not  the charcter in dict:', character, content)
                break
            content_points.append(points)
            offset_xy = (max(points[1][0], points[2][0]), offset_xy[1])

        roi = utils.getOneLineRectanglePoints(content_points)
        roi_tuple = (int(roi[0][0]), int(roi[0][1]), int(roi[2][0]), int(roi[2][1]))
        image_out = image.crop(roi_tuple)

        return image_out, content_points

    def putOneCharacter2Image(self, character, background_image, font_path, font_size, offset_xy, color=None):
        background = background_image.convert('RGBA')
        try:
            font = ImageFont.truetype(font_path, font_size)
        except AssertionError:
            print('open font file failed.', font_path)
        width, height = font.getsize(character)
        #print('width height ', width, height)

        txt_canvas = Image.new('RGBA', (width, height), (255,255,255,0))
        draw = ImageDraw.Draw(txt_canvas)
        draw.text((0, 0), character, font=font, fill=(color[0],color[1],color[2],255))  # draw text, full opacity
        #if os.path.exists('tmp') is False:
        #    os.mkdir('tmp')
        #filename = 'tmp/2019_%06d.png' % (self.id)
        #txt_canvas.save(filename)
        self.id = self.id + 1

        # transform_matrix = (1, 0, 0,
        #                     0, 1, 0)
        # txt_canvas = txt_canvas.transform((width, height),
        #                         Image.AFFINE,
        #                         transform_matrix,
        #                         Image.BICUBIC)
        w_x, h_x = txt_canvas.size
        if txt_canvas is None or w_x==0 or h_x==0:
            print(type(txt_canvas), id(txt_canvas), character, width, height, txt_canvas.size)
            return None, None
        corner_points = utils.getPointsOfImageRectangle(width, height)
        absolute_corner_points = utils.mergeBgimgAndTxtimgPoints(offset_xy, corner_points)
        absolute_corner_points = absolute_corner_points.astype(np.int32)
        absolute_corner_points[absolute_corner_points < 0] = 0
        absolute_corner_points[:, 0][absolute_corner_points[:, 0] >= background.size[0]] = background.size[0]-1
        absolute_corner_points[:, 1][absolute_corner_points[:, 1] >= background.size[1]] = background.size[1]-1
        #print('absolute_corner_points', absolute_corner_points)

        left, top = absolute_corner_points[0]
        new_width, mew_height = txt_canvas.size
        if (left + new_width) > background.size[0]:
            absolute_corner_points[0][0] = absolute_corner_points[0][0] - (left + new_width) + background.size[0]
        if (top + mew_height) > background.size[1]:
            absolute_corner_points[0][1] = absolute_corner_points[0][1] - (top + mew_height) + background.size[1]

        #out_image = Image.alpha_composite(background, txt_canvas)
        out_image = utils.mergeImageAtPoint(background, txt_canvas, tuple(absolute_corner_points[0]))
        out_image = out_image.convert('RGB')        
        return out_image, absolute_corner_points

    def saveTopNCharacters2File(self, characters_file_path, top_n, save_path):
        if 'id_character_dict' not in dict():
            self.id_character_dict = utils.getTopNCharacters2Dict(characters_file_path, top_n)
        #utils.saveIdCharacterDict2File(self.id_character_dict, save_path)
        return


    def chooseSaveDirByIndex(self, image_number_index, is_part_img=0):
        train_rate = 1 - self.args['test_rate'] - self.args['validation_rate']
        if image_number_index < int(train_rate * self.args['image_number']):
            if not is_part_img:
                image_save_dir = self.args['train_image_dir']
            else:
                image_save_dir = self.args['train_part_image_dir']
        elif image_number_index < int((train_rate + self.args['validation_rate'])* self.args['image_number']):
            if not is_part_img:
                image_save_dir = self.args['validation_image_dir']
            else:
                image_save_dir = self.args['validation_part_image_dir']
        else:
            if not is_part_img:
                image_save_dir = self.args['test_image_dir']
            else:
                image_save_dir = self.args['test_part_image_dir']
        return image_save_dir

    def collectFileNum(self):
        train_image_list = glob.glob(os.path.join(self.args['train_part_image_dir'], '*.jpg'))
        validation_image_list = glob.glob(os.path.join(self.args['validation_part_image_dir'], '*.jpg'))
        test_image_list = glob.glob(os.path.join(self.args['test_part_image_dir'], '*.jpg'))
        annotations_list = glob.glob(os.path.join(self.args['annotations_dir'], '*.txt'))
        if len(test_image_list) != 0:
            max_index = utils.findMaxIndex(test_image_list)
            print('test image num:', max_index)
        elif len(validation_image_list) != 0:
            max_index = utils.findMaxIndex(validation_image_list)
            print('validation image num:', max_index)
        elif len(train_image_list) != 0:
            max_index = utils.findMaxIndex(train_image_list)
            print('train image num:', max_index)
        else:
            max_index = 0

        if len(annotations_list) != 0:
            annotations_index = utils.findMaxIndex(annotations_list)
            print('annotations_list num:', annotations_index)
        return max_index

    def restoreFromPartImageDir(self):
        train_image_list = glob.glob(os.path.join(self.args['train_part_image_dir'], '*.jpg'))
        validation_image_list = glob.glob(os.path.join(self.args['validation_part_image_dir'], '*.jpg'))
        test_image_list = glob.glob(os.path.join(self.args['test_part_image_dir'], '*.jpg'))
        if len(test_image_list) != 0:
            max_index = utils.findMaxIndex(test_image_list)
        elif len(validation_image_list) != 0:
            max_index = utils.findMaxIndex(validation_image_list)
        elif len(train_image_list) != 0:
            max_index = utils.findMaxIndex(train_image_list)
        else:
            max_index = 0
        if max_index != 0:
            print('There are %d images had been generated before. Do you want to continue from that? "y" will continue or "n" will recover.'%max_index)
            #choose = input()
            #while choose not in ['y', 'n']:
            #    print('Input error, please choose "y" or "n".')
            #    choose = input()
            #if choose == 'n':
            #    max_index = 0
        return max_index

    def add_effect(self, images):
        blur_imgs = []
        for index, img in enumerate(images):
            effects = AE.Effects(img)
            size = np.random.randint(5, 7)
            #print(size)
            blur_img = effects.motion_blur(size)
            blur_imgs.append(blur_img)
        return blur_imgs

if __name__ == '__main__':
    samples = []
    main()

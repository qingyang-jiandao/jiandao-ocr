# -*- coding:utf-8 -*-

import os
import random
from PIL import Image, ImageDraw, ImageFont
import utils 
import codecs
import tqdm
import glob
import numpy as np
import add_effects as AE
import Aug_Operations as aug
from generate_image_base import Process

def main():
    root_dir = '../synthetic_data_6862_n10_coco'
    data = OCRData(root_dir)
    data.args['classes_number'] = 6862

    data.makeNeededDir()
    if os.path.exists(root_dir):
        data.collectFileNum()
    print('The arguments :\n', data.args)

    data.args['image_number'] = data.args['classes_number']*1
    data.bg_img_list = utils.getBackgroundListFromDir(data.args['background_image_dir'])
    data.args['image_number'] = 1000
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
        # self.space = self.wordict[80]
        self.id = 0

    def setArguments(self):
        self.args['characters_length_tuple'] = (4, 10)
        self.args['validation_rate'] = 0.0
        self.args['test_rate'] = 0.0
        # self.args['background_image_dir'] = 'G:/deeplab/dataset/ms-coco/images/train2014'
        # self.args['background_image_dir'] = 'G:/deeplab/dataset/ms-coco/images/test2014'
        self.args['background_image_dir'] = 'G:/deeplab/dataset/ms-coco/images/val2014'
        self.args['fonts_dir'] = '../fonts_all'
        self.args['characters_file_path'] = '1s2n.list'
        self.args['classes_number'] = 6862
        self.args['font_size_min'] = 16
        self.args['font_size_max'] = 32
        self.args['image_number'] = 100
        self.args['save_full_image'] = 0
        self.args['add_rectangle'] = 0
        self.args['lines_num'] = 2
        return self.args

    def get_content(self, length_range_tuple):
        rand_len = random.randint(length_range_tuple[0], length_range_tuple[1])
        # print('space', len(self.space[0]))
        content = u''
        while True:
            if np.random.binomial(1, 0.85):  # 85% chinese character
                for i in range(rand_len):
                    space_op = random.randint(0, 50)
                    char = u''
                    if np.random.binomial(1, 0.96):  # 96% first-level & second-level chinese character
                        if np.random.binomial(1, 0.7):  # 70% first-level character
                            index = np.random.randint(0, len(self.f1chi))
                            char = self.f1chi[index]
                            if space_op == 0:
                                char = self.space[0]
                            content = content + char
                            #print("1", char, index)
                        else:  # 30% second-level character
                            index = np.random.randint(0, len(self.f2chi))
                            char = self.f2chi[index]
                            if space_op == 0:
                                char = self.space[0]
                            content = content + char
                            #print("2", char, index)
                    else:  # 4% alpha,num,punt
                        if np.random.binomial(1, 0.5):  # 50% alpha,num
                            index = np.random.randint(0, len(self.alphanum))
                            char = self.alphanum[index]
                            if space_op == 0:
                                char = self.space[0]
                            content = content + char
                            #print("3", char, index)
                        else:  # 50% punt
                            index = np.random.randint(0, len(self.punt))
                            char = self.punt[index]
                            content = content + char
                            #print("4", char, index)
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
                    length_range_tuple[0] = 2
                    length_range_tuple[1] = 8
                    rand_len = random.randint(length_range_tuple[0], length_range_tuple[1])
                    for i in range(rand_len):
                        space_op = random.randint(0, 100)
                        index = np.random.randint(0, len(self.num))
                        char = self.num[index]
                        if space_op == 0:
                            char = self.space[0]
                        content = content + char
            if len(content) < length_range_tuple[0] or len(content) > length_range_tuple[1]:
                print(content, len(content))
            assert len(content) >= length_range_tuple[0], '%d %d' % (len(content), length_range_tuple[0])
            assert len(content) <= length_range_tuple[1], '%d %d' % (len(content), length_range_tuple[1])
            if len(content) >= length_range_tuple[0]:
                break
        return content

    #
    # def get_content(self, length_range_tuple):
    #     rand_len = random.randint(length_range_tuple[0], length_range_tuple[1])
    #     # print('space', len(self.space[0]))
    #     content = u''
    #     while True:
    #         if np.random.binomial(1, 0.70):  # 70% chinese character
    #             for i in range(rand_len):
    #                 space_op = random.randint(0, 50)
    #                 char = u''
    #                 if np.random.binomial(1, 0.90):  # 96% first-level & second-level chinese character
    #                     if np.random.binomial(1, 0.8):  # 70% first-level chinese character
    #                         index = np.random.randint(0, len(self.f1chi))
    #                         char = self.f1chi[index]
    #                         if space_op == 0:
    #                             char = self.space[0]
    #                             # print(char, 'char', len(self.space[0]), len(char))
    #                         content = content + char
    #                         #print("1", char, index)
    #                     else:  # 30% second-level chinese character
    #                         index = np.random.randint(0, len(self.f2chi))
    #                         char = self.f2chi[index]
    #                         if space_op == 0:
    #                             char = self.space[0]
    #                         content = content + char
    #                         #print("2", char, index)
    #                 else:  # 4% alpha,num,punt
    #                     if np.random.binomial(1, 0.95):  # 80% alpha,num
    #                         index = np.random.randint(0, len(self.alphanum))
    #                         char = self.alphanum[index]
    #                         if space_op == 0:
    #                             char = self.space[0]
    #                         content = content + char
    #                         #print("3", char, index)
    #                     else:  # 20% punt
    #                         index = np.random.randint(0, len(self.punt))
    #                         char = self.punt[index]
    #                         content = content + char
    #                         #print("4", char, index)
    #         else:
    #             if np.random.binomial(1, 0.70):
    #                 head_type = random.randint(0, 1)
    #                 for i in range(rand_len):
    #                     space_op = random.randint(0, 50)
    #                     char = u''
    #                     if head_type == 0:
    #                         if i < rand_len // 2:
    #                             if np.random.binomial(1, 0.8):  # 80% first-level chinese character
    #                                 index = np.random.randint(0, len(self.f1chi))
    #                                 char = self.f1chi[index]
    #                                 if space_op == 0:
    #                                     char = self.space[0]
    #                                     # print(char, 'char', len(self.space[0]), len(char))
    #                                 content = content + char
    #                                 # print("1", char, index)
    #                             else:  # 20% second-level chinese character
    #                                 index = np.random.randint(0, len(self.f2chi))
    #                                 char = self.f2chi[index]
    #                                 if space_op == 0:
    #                                     char = self.space[0]
    #                                 content = content + char
    #                                 # print("2", char, index)
    #                         else:  # 50% alpha,num,punt
    #                             if np.random.binomial(1, 0.98):  # 95% alpha,num
    #                                 index = np.random.randint(0, len(self.alphanum))
    #                                 char = self.alphanum[index]
    #                                 if space_op == 0:
    #                                     char = self.space[0]
    #                                 content = content + char
    #                                 # print("3", char, index)
    #                             else:  # 2% punt
    #                                 index = np.random.randint(0, len(self.punt))
    #                                 char = self.punt[index]
    #                                 content = content + char
    #                                 # print("4", char, index)
    #                     else:
    #                         if i < rand_len // 2:
    #                             if np.random.binomial(1, 0.98):  # 98% alpha,num
    #                                 index = np.random.randint(0, len(self.alphanum))
    #                                 char = self.alphanum[index]
    #                                 if space_op == 0:
    #                                     char = self.space[0]
    #                                 content = content + char
    #                                 # print("3", char, index)
    #                             else:  # 2% punt
    #                                 index = np.random.randint(0, len(self.punt))
    #                                 char = self.punt[index]
    #                                 content = content + char
    #                                 # print("4", char, index)
    #                         else:
    #                             if np.random.binomial(1, 0.8):  # 80% first-level chinese character
    #                                 index = np.random.randint(0, len(self.f1chi))
    #                                 char = self.f1chi[index]
    #                                 if space_op == 0:
    #                                     char = self.space[0]
    #                                     # print(char, 'char', len(self.space[0]), len(char))
    #                                 content = content + char
    #                                 # print("1", char, index)
    #                             else:  # 20% second-level chinese character
    #                                 index = np.random.randint(0, len(self.f2chi))
    #                                 char = self.f2chi[index]
    #                                 if space_op == 0:
    #                                     char = self.space[0]
    #                                 content = content + char
    #                                 # print("2", char, index)
    #             else:
    #                 if np.random.binomial(1, 0.90):
    #                     for i in range(rand_len):
    #                         char = u''
    #                         space_op = random.randint(0, 100)
    #                         if np.random.binomial(1, 0.99):
    #                             index = np.random.randint(0, len(self.alphanum))
    #                             char = self.alphanum[index]
    #                             if space_op == 0:
    #                                 char = self.space[0]
    #                             content = content + char
    #                         else:
    #                             index = np.random.randint(0, len(self.punt))
    #                             content = content + self.punt[index]
    #                 else:
    #                     length_range_tuple[0]=2
    #                     length_range_tuple[1]=8
    #                     rand_len = random.randint(length_range_tuple[0], length_range_tuple[1])
    #                     for i in range(rand_len):
    #                         char = u''
    #                         space_op = random.randint(0, 100)
    #                         index = np.random.randint(0, len(self.num))
    #                         char = self.num[index]
    #                         if space_op == 0:
    #                             char = self.space[0]
    #                         content = content + char
    #
    #         if len(content) < length_range_tuple[0] or len(content) > length_range_tuple[1]:
    #             print(content, len(content))
    #         assert len(content)>=length_range_tuple[0], '%d %d' % (len(content), length_range_tuple[0])
    #         assert len(content) <= length_range_tuple[1], '%d %d' % (len(content), length_range_tuple[1])
    #         if len(content)>=length_range_tuple[0]:
    #             break
    #     return content
    
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
        self.args['segObjects_dir'] = os.path.join(self.args['root_dir'], 'segObjects')
        utils.makeDirectory(self.args['segObjects_dir'])
        return

    def makePartDirs(self, part_role):
        if self.args['save_full_image']:
            self.args[''.join([part_role,'_image_dir'])] = os.path.join(self.args['root_dir'], part_role+'_images')
            utils.makeDirectory(self.args[part_role+'_image_dir'])
        self.args[part_role+'_part_image_dir'] = os.path.join(self.args['root_dir'], part_role+'_part_images')
        utils.makeDirectory(self.args[part_role+'_part_image_dir'])
        return

    def generator(self, sample, thread_id):
        contents, content_indexs, background_image_path, font_path, img_number = sample
        background_image = Image.open(background_image_path).convert('RGBA')
        # color = np.zeros(3, dtype=np.int)
        # color[0] = random.randint(0, 255)
        # color[1] = random.randint(0, 255)
        # color[2] = random.randint(0, 255)
        # background_image = Image.new('RGBA', (600, 600), (color[0], color[1], color[2], 255))
        if background_image is None:
            return
        bg_width, bg_height = background_image.size
        roi_images = []
        part_images = []
        roi_boxes = []
        annotation = []
        font_size = 0
        for i, content in enumerate(contents):
            font_size = random.randint(self.args['font_size_min'], self.args['font_size_max'])
            image, points = self.putContent2Image(content, font_path, font_size=font_size)
            if image is None or points is None or len(points) == 0:
                continue
            if self.args['save_full_image']:
                self.saveImage(image, img_number)

            text_img_width, text_img_height = image.size
            if bg_width < text_img_width + 40 or bg_height < text_img_height + 12:
                continue

            transform = []
            for loop_number in range(10):
                # random choice a position at image.
                offset_x = random.randint(0, bg_width - text_img_width - 24)
                offset_y = random.randint(0, bg_height - text_img_height - 8)
                rand_ex_x = random.randint(0, 24)
                rand_ex_y = random.randint(0, 8)
                if utils.test_collision(roi_boxes, offset_x, offset_y, text_img_width + rand_ex_x, text_img_height + rand_ex_y) == 0:
                    roi_box = (offset_x, offset_y, text_img_width + rand_ex_x, text_img_height + rand_ex_y)
                    transform = [offset_x, offset_y, rand_ex_x, rand_ex_y]
                    break

            if len(transform) > 0:
                roi_image, part_image, roi_box = utils.mergeAndTransformImage(image, transform, background_image)
                if roi_image is not None:
                    roi_images.append(roi_image)
                    part_images.append(part_image)
                    roi_boxes.append(roi_box)
                    annotation.append(content)
        if font_size > 24:
            roi_images = self.add_effect(roi_images)
        self.saveSegmentationObject(part_images, img_number)
        self.saveImage(roi_images, img_number, is_part=1)
        self.saveAnnotation(annotation, img_number)

    def synthesizeAllImages(self, image_number):
        self.font_list = utils.getFontListFromDir(self.args['fonts_dir'])
        start_index = self.restoreFromPartImageDir()
        for i in tqdm.tqdm(range(start_index, image_number)):
            contents, content_indexs = self.get_contents(self.args['characters_length_tuple'], self.args['lines_num'])
            background_image_path, font_path = map(utils.getRandomOneFromList, [self.bg_img_list, self.font_list])
            samples.append((contents, content_indexs, background_image_path, font_path, i))

        assert (image_number-start_index)==len(samples)
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

    def saveSegmentationObject(self, images, image_index):
        for index, image in enumerate(images):
            label_name = ''.join([str(image_index), '_', str(index), '.png'])
            label_path = os.path.join(self.args['segObjects_dir'], label_name)
            seg_array = np.array(image.convert("L"), dtype=np.float32)
            #np.savetxt('image_np' + '.csv', seg_array, delimiter=',')
            mask = seg_array > 0
            seg_array[mask] = 1
            label_image = Image.fromarray(seg_array.astype(np.uint8))
            label_image.save(label_path)
            
    def putContent2Image(self, content, font_path, font_size=0, resize_rate=2):
        #image = Image.open(background_image_path)
        #image = Image.new('RGBA', (512, 512), (255, 255, 255, 0))

        image = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
        if image is None or image.size[0] <= 0 or image.size[1] <= 0:
            return None, None
        font_size_max = image.size[0]/self.args['characters_length_tuple'][1]
        while font_size_max < self.args['font_size_min']:
            resize_rate = resize_rate * 2
            image = image.resize((image.size[0]*resize_rate, image.size[1]*resize_rate))
        font_size_max = self.args['font_size_max']
        if image is None or image.size[0] <= 0 or image.size[1] <= 0:
            return None, None

        if font_size == 0:
            font_size = random.randint(self.args['font_size_min'], int(font_size_max))
            if int(image.size[0] - font_size * len(content)) < 0 or \
                    int(image.size[1] - font_size) < font_size:
                return None, None

        offset_x = random.randint(0, int(image.size[0] - font_size * len(content)))
        offset_y = random.randint(font_size, int(image.size[1] - font_size))
        offset_xy = (offset_x, offset_y)

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

        content_points = []
        for character in content:
            image, points = self.putOneCharacter2Image(character, image, font_path, font_size, offset_xy, color)
            if image is None:
                print('####error: the character is no in dict:', character, content)
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
            size = np.random.randint(3, 5)
            blur_img = effects.motion_blur(size)
            # type = np.random.randint(0, 2)
            # if type == 0:
            #     blur_img = effects.motion_blur(size)
            # else:
            #     blur_img = effects.motion_blur(size, type=0)
            blur_img = blur_img.convert("RGB")
            blur_imgs.append(blur_img)
        return blur_imgs

if __name__ == '__main__':
    samples = []
    main()

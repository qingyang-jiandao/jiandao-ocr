'''
date: 2018.07.12
author: lyfee
des: some noise function&effects
'''

import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from skimage import util
from ConvertRGBA2RGB import alpha_composite, alpha_to_color

class Effects(object):
    def __init__(self, image):
        self.image = image.convert("RGBA")
        self.img_arr = np.array(image)
        # print(self.img_arr.shape)

    def motion_blur(self, size = 5, type=-1):
        if type == -1:
            type = np.random.randint(0, 10)
            # type = 0

        if type == 0 or type == 3:
            self.image = alpha_to_color(self.image)
            self.img_arr = np.array(self.image)
            self.img_arr = np.transpose(self.img_arr, (2, 0, 1))
            # generating the kernel
            kernel_motion_blur = np.zeros((size, size))
            kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
            kernel_motion_blur /= size
            for i in range(self.img_arr.shape[0]):
                map = self.img_arr[i]
                self.img_arr[i] = cv2.filter2D(map, -1, kernel_motion_blur)
                # matplotlib.image.imsave("data/stn_out_" + str(i) +".jpg", map, cmap = cm.gray)
                #np.savetxt('data/moran_out_' + str(i) + '.csv', map, delimiter=',')

            self.img_arr = np.transpose(self.img_arr, (1, 2, 0))
            img_blur = Image.fromarray(self.img_arr)

        elif type==1:
            noise_img = util.random_noise(self.img_arr, mode='gaussian')
            noise_img = noise_img * 255
            img_blur = Image.fromarray(noise_img.astype(np.uint8))

        # elif type==2:
        #     noise_img = util.random_noise(self.img_arr, mode='salt')
        #     noise_img = noise_img * 255
        #     img_blur = Image.fromarray(noise_img.astype(np.uint8))

        elif type==2:
            noise_img = util.random_noise(self.img_arr, mode='pepper')
            noise_img = noise_img * 255
            img_blur = Image.fromarray(noise_img.astype(np.uint8))

        # elif type==3:
        #     noise_img = util.random_noise(self.img_arr, mode='s&p')
        #     noise_img = noise_img * 255
        #     img_blur = Image.fromarray(noise_img.astype(np.uint8))

        elif type==4:
            enh_bri = ImageEnhance.Brightness(self.image)
            brightness = np.random.uniform(0.2, 3.0)
            img_blur = enh_bri.enhance(brightness)

        elif type==5:
            enh_con = ImageEnhance.Contrast(self.image)
            contrast = np.random.uniform(0.2, 0.5)
            img_blur = enh_con.enhance(contrast)

        elif type==6:
            self.image = alpha_to_color(self.image)
            self.img_arr = np.array(self.image)
            enh_sha = ImageEnhance.Sharpness(self.image)
            sharpness = np.random.uniform(5, 10)
            img_blur = enh_sha.enhance(sharpness)

        else:
            img_blur = self.image

        return img_blur


#image = Image.open("data_6_7.jpg")
# # effects = Effects(image)
# # size = np.random.randint(3, 9)
# # print(size)
# # blur_img = effects.motion_blur(size)
# # blur_img.save("data_6_7_blur.jpg")
#
# image = np.array(image)
# noise_gs_img = util.random_noise(image, mode='gaussian')
# print(type(noise_gs_img), noise_gs_img.shape)
# noise_gs_img = noise_gs_img*255
# noise_gs_img = Image.fromarray(noise_gs_img.astype(np.uint8))
# noise_gs_img.save("noise_gs_img.jpg")
#
# noise_salt_img = util.random_noise(image,mode='salt')
# noise_salt_img = noise_salt_img*255
# noise_salt_img = Image.fromarray(noise_salt_img.astype(np.uint8))
# noise_salt_img.save("noise_salt_img.jpg")
#
# noise_pepper_img = util.random_noise(image,mode='pepper')
# noise_pepper_img = noise_pepper_img*255
# noise_pepper_img = Image.fromarray(noise_pepper_img.astype(np.uint8))
# noise_pepper_img.save("noise_pepper_img.jpg")
#
# noise_sp_img = util.random_noise(image,mode='s&p')
# noise_sp_img = noise_sp_img*255
# noise_sp_img = Image.fromarray(noise_sp_img.astype(np.uint8))
# noise_sp_img.save("noise_sp_img.jpg")


# #亮度增强
# enh_bri = ImageEnhance.Brightness(image)
# brightness = 3.0 #(0.2-3.0)
# image_brightened = enh_bri.enhance(brightness)
# image_brightened.show()

# # 色度增强
# enh_col = ImageEnhance.Color(image)
# color = 2.5 #0.2-0.5
# image_colored = enh_col.enhance(color)
# image_colored.show()

# # 对比度增强
# enh_con = ImageEnhance.Contrast(image)
# contrast = 0.2  #0.2-0.5
# image_contrasted = enh_con.enhance(contrast)
# image_contrasted.show()

# # 锐度增强
# enh_sha = ImageEnhance.Sharpness(image)
# sharpness = 5.0  #5-10
# image_sharped = enh_sha.enhance(sharpness)
# image_sharped.show()
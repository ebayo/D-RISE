# Functions to generate masks and operate with them??
import cv2
import numpy as np


class MaskGenerator:
    def __init__(self, size=(16, 16), prob=0.5):
        # prob is the probability of 1 in binary mask generator (~ how much the masks cover the original image)
        # TODO: change prints for asserts
        if 0 <= prob <= 1:
            self.p = prob
        else:
            print(f'Probability of 1 should be in range [0,1]. Is {prob}')

        if type(size) is int:
            self.h = size
            self.w = size

        elif type(size) is tuple and len(size) == 2:
            self.h, self.w = size
            if type(self.h) is not int or type(self.w) is not int:
                print(f'Size values in tuple {size} should be integers')
        else:
            print(f'Size {size} is not a valid value. Should be int or tuple')

        self.rnd_gen = np.random.default_rng()

    def get_mask(self, H, W):
        ch = H // self.h
        cw = W // self.w
        resize = ((self.w+1)*cw, (self.h+1)*ch)
        # print(f'H={H}, W={W}, resize = {resize}')
        binmask = self.rnd_gen.choice([0, 1], size=(self.h, self.w), replace=True, p=[1 - self.p, self.p])
        mask = cv2.resize(binmask.astype('float32'), resize, interpolation=cv2.INTER_LINEAR)
        # print(f'mask post resize {mask.shape}')
        # oh = self.rnd_gen.integers(ch, endpoint=True)     # vertical offset
        # ow = self.rnd_gen.integers(cw, endpoint=True)     # horizontal offset
        # print(f'oh = {oh}, ow= {ow}')
        mask = mask[:H, :W]
        # print(f'Cropped mask: {mask.shape}')
        return mask

    def mask_image(self, image):
        H, W = image.shape[:2]
        mask = self.get_mask(H, W)
        return ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) * 255).astype(np.uint8)

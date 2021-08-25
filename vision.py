import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import random

# Code author: Girish Hegde
class MixUp(nn.Module):
    def __init__(self,
        p = 0.5,
        ):
        '''
        Batch level augmentation
        '"" """" " " "'
        '''
        super().__init__()
        self.p = p

    def forward(self, x, y=None):
        '''
        x: [bs, ch, h, w] not [ch, h, w] bcoz it is batch level augmentation
        y: labels if available - [bs, k]
        '"" """" " " "'
        '''
        bs, ch, h, w = x.shape
        if random.random() < self.p:
            shuffled = torch.randperm(bs)
            x_ = x[shuffled, ]
            lmda = random.random()**0.5
            x = lmda*x + (1 - lmda)*x_
            if y is not None:
                y_ = t[shuffled, ]
                y = lmda*y + (1 - lmda)*y_
        if y is not None:
            return x, y
        return x



class CutOut(nn.Module):
    def __init__(self,
        num_holes = 1,
        max_h = 8,
        max_w = 8,
        fill_value = 0,
        p = 0.5,
        ):
        super().__init__()
        self.num_holes = num_holes
        self.max_h = max_h
        self.max_w = max_w 
        self.fill_value = fill_value
        self.p = p

    def forward(self, x):
        c, h, w = x.shape
        for _ in range(self.num_holes):
            if random.random() < self.p:
                i = random.randint(0, h - 1)
                j = random.randint(0, w - 1)
                x[..., i: i + random.randint(0, self.max_h - 1), j: j + random.randint(0, self.max_w - 1)] = self.fill_value
        return x


class CutMix(nn.Module):
    def __init__(self,
        p = 0.5,
        ):
        '''
        Batch level augmentation
        '"" """" " " "'
        '''
        super().__init__()
        self.p = p

    def forward(self, x, y=None):
        '''
        x: [bs, ch, h, w] not [ch, h, w] bcoz it is batch level augmentation
        y: labels if available - [bs, k]
        '"" """" " " "'
        '''
        bs, ch, h, w = x.shape
        if random.random() < self.p:
            shuffled = torch.randperm(bs)
            x_ = x[shuffled, ]
            lmda = random.random()**0.5
            i = random.randint(0, h - 1)
            j = random.randint(0, w - 1)
            i_ = int(i + lmda*h)
            j_ = int(j + lmda*w)
            x[..., i: i_, j: j_] = x_[..., i: i_, j: j_]
            if y is not None:
                area = (torch.clip(i_, 0, h) - i)*(torch.clip(j_, 0, w) - j)
                lmda = area/(w*h) 
                y_ = y[shuffled, ]
                y = lmda*y_ + (1 - lmda)*y
        if y is not None:
            return x, y
        return x


# Unit test
if __name__ == '__main__':
    import numpy as np
    import cv2

    def show(img, save_path=None):
        img =  (img.permute(1, 2, 0).numpy()*255).astype(np.uint8)[:, :, ::-1]
        cv2.imshow('Output', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if save_path is not None:
            cv2.imwrite(save_path, img)

    # # CutOut
    # img = cv2.imread('./data/lena.tif')
    # img = torch.tensor(img[:, :, ::-1].copy()).permute(2, 0, 1)/255.
    
    # cutout = CutOut(5, 100, 100, 0., 1.0)
    # cutouted = cutout(img)
    
    # show(cutouted, './output/cutout.png')

    # CutMix
    # imgs = cv2.imread('./data/lena.tif')
    # imgs = torch.tensor(imgs[:, :, ::-1].copy()).permute(2, 0, 1)/255.

    # img = cv2.imread('./data/lena2.tiff')
    # img = torch.tensor(img[:, :, ::-1].copy()).permute(2, 0, 1)/255.

    # imgs = torch.cat([imgs[None, :, :, :], img[None, :, :, :]], dim=0)
    
    # cutmix = CutMix(1.) 
    # cutmixed = cutmix(imgs)
    
    # show(cutmixed[0], './output/cutmix.png')

    # MixUp
    # imgs = cv2.imread('./data/lena.tif')
    # imgs = torch.tensor(imgs[:, :, ::-1].copy()).permute(2, 0, 1)/255.

    # img = cv2.imread('./data/lena2.tiff')
    # img = torch.tensor(img[:, :, ::-1].copy()).permute(2, 0, 1)/255.

    # imgs = torch.cat([imgs[None, :, :, :], img[None, :, :, :]], dim=0)
    
    # mixup = MixUp(1.) 
    # mixedup = mixup(imgs)
    
    # show(mixedup[0], './output/mixup.png')



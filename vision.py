import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import random


# Helper functions taken from torchvision
def to_gray(img, keep_dims=True):
    '''
    Args:
        img: image tensor of shape [ch, h, w]
    '''
    gray = (0.2989 * img[0] + 0.587 * img[1] + 0.114 * img[2])
    return torch.stack([gray, gray, gray]) if keep_dims else gray

def _rgb2hsv(img):
    r, g, b = img.unbind(dim=-3)

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = (hr + hg + hb)
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv2rgb(img):
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)

    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)

    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)

    return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)


class ColorJitter(nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        '''
        Args:
            brightness: float or tuple(min, max)
            contrast: float or tuple(min, max)
            saturation: float or tuple(min, max)
            huw: float or tuple(min, max) - [0.5, 0.5]
        '''
        if type(brightness) not in {tuple, list}:
            self.brightness = (max(0, 1 - brightness), 1 + brightness)
        else:
            self.brightness = brightness
        if type(contrast) not in {tuple, list}:
            self.contrast = (max(0, 1 - contrast), 1 + contrast)
        else:
            self.contrast = contrast        
        if type(saturation) not in {tuple, list}:
            self.saturation = (max(0, 1 - saturation), 1 + saturation)
        else:
            self.saturation = saturation   
        if type(hue) not in {tuple, list}:
            self.hue = (-hue, hue)
        else:
            self.hue = hue

    def forward(self, x):
        '''
        Args:
            x: Image tensor of dimension [..., h, w]            
        Returns:
            jitter(x)
        '''
        brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
        x = (x*brightness_factor).clip(0., 1.)

        contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
        gray = to_gray(x).mean()
        x = (contrast_factor * x + (1.0 - contrast_factor) * gray).clip(0., 1.)
        
        saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
        gray = to_gray(x)
        x = (contrast_factor * x + (1.0 - contrast_factor) * gray).clip(0., 1.)
        
        hue_factor = random.uniform(self.hue[0], self.hue[1])
        x = _rgb2hsv(x)
        h, s, v = x.unbind(dim=-3)
        h = (h + hue_factor) % 1.0
        x = torch.stack((h, s, v), dim=-3)
        x = _hsv2rgb(x)
        
        return x



class MixUp(nn.Module):
    def __init__(self,
        p = 0.5,
        ):
        '''
        Batch level augmentation
        '''
        super().__init__()
        self.p = p

    def forward(self, x, y=None):
        '''
        x: [bs, ch, h, w] not [ch, h, w] bcoz it is batch level augmentation
        y: labels if available - [bs, k]
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
        '''
        super().__init__()
        self.p = p

    def forward(self, x, y=None):
        '''
        x: [bs, ch, h, w] not [ch, h, w] bcoz it is batch level augmentation
        y: labels if available - [bs, k]
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

    # # Jitter
    # img = cv2.imread('./data/einstein.jpg')
    # img = torch.tensor(img[:, :, ::-1].copy()).permute(2, 0, 1)/255.
    
    # jitter = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    # jittered = jitter(img)
    
    # show(jittered, './output/jitter.png')

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



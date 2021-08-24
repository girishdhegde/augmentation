import glob

import torch 
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import cv2


class VisionDataset(Dataset):
    def __init__(self,
    img_path = './data/',
    ):
        super().__init__()
        self.images = glob.glob(f'{img_path}/*')
        self.len = len(self.images)
        
        # BYOL Augmenetations: https://arxiv.org/pdf/2006.07733.pdf
        self.tranforms = T.Compose([
            T.ToTensor(),
            T.CenterCrop((250, 250)),
            T.Pad((3, 3, 3, 3), fill=0, padding_mode='constant'),
            
            T.RandomResizedCrop(size=(256, 256), interpolation=TF.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),

            T.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),

            T.RandomApply(torch.nn.ModuleList([
                T.Grayscale(num_output_channels=3), 
                ]),
                p=0.5
            ),

            T.GaussianBlur(23, sigma=(0.1, 2.0)),
            T.RandomSolarize(0.5, p=0.5),

            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Read image and BGR to RGB(For opencv to PIL format conversion)
        # copy() is required because -ve stride not supported
        img = cv2.imread(self.images[idx])[..., ::-1].copy()
        img = self.tranforms(img)

        return img


if __name__ == '__main__':

    trainset = VisionDataset('./data/')
    data = next(iter(trainset))

    img = data.permute(1, 2, 0).numpy()[..., ::-1]
    img = (img*0.5) + 0.5
    print(img.max(), img.min())
    cv2.imshow('img', img.clip(0., 1.))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


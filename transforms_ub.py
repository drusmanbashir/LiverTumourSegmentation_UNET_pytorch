
import numpy as np
import random
import torch
from torch.nn import functional as F

class RandomFlip(object):
    def __init__ (self,model_type,num_channels=1): #if none , random plane and angle are used):
        assert model_type in ['unet','classification'],"model_type can either be 'unet' or 'classification!"
        self.model_type=model_type
    def __call__(self, sample):
        if self.model_type=='unet':
           self.axis = random.sample([1,2,3],k=2)  #note 0 is channel dim so not included
           image,label = sample
           img, label = image.transpose(self.axis[0],self.axis[1]), label.transpose(self.axis[0],self.axis[1])
           return img,label


class RandomFlipHorizontal(object):
    def __call__(self,sample):
        heads = random.randint(0, 1)
        if heads ==1:
            im,mask = sample
            im_flipped = im.flip(2)  # assuming im is channelxheightxwidthxslices
            mask_flipped = mask.flip(2)
            sample = im_flipped,mask_flipped
        return sample


class Rotate3D(object):  #on an numpy dataset   #NOTE: HAVENT CONFIRMED IT WORKING WITH 2 CHANNEL DATA I.E., T1CE + FLAIR
    def __init__ (self,model_type,num_channels=1): #if none , random plane and angle are used):
        assert model_type in ['unet','classification'],"model_type can either be 'unet' or 'classification!"
        self.model_type=model_type


    def __call__(self,sample,plane=None, angle=None):
        if not plane:
            self.plane =  np.random.choice([1, 2, 3], size=2, replace=False)  # axis 0 is channel the next 3 are 3d axes of data
        if not angle:
            self.angle = np.random.randint(0, 360)
        image, label = sample
        rotMatrix = torch.tensor([torch.cos(self.angle)])
        image = R(image,axes=self.plane, angle=self.angle,reshape=False)
        if self.model_type=='unet':
            label= R(label.int(),axes=self.plane, angle=self.angle,reshape=False)
        return([image,label])

class Rotate3D_tensor(object):
    def __init__ (self,model_type,num_channels=1): #if none , random plane and angle are used):
        assert model_type in ['unet','classification'],"model_type can either be 'unet' or 'classification!"
        self.model_type=model_type

    def __call__(self,sample,axis=None, angle=None):
        if not axis:
            self.axis =  np.random.choice([1, 2, 3], size=1, replace=False)  # axis 0 is channel the next 3 are 3d axes of data
        if not angle:
            self.angle = np.random.randint(0, 360)


        image, label = sample
        image = R(image,axes=self.plane, angle=self.angle,reshape=False)
        if self.model_type=='unet':
            label= R(label.int(),axes=self.plane, angle=self.angle,reshape=False)
        return([image,label])



class Normalize(object):
    # def __init__(self):
    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W, D) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        image,label = sample
        tensor = F.normalize(image, p=2,dim=(1,2))
        return ([tensor,label])

def normalize_numpy(a):
    b = (a - np.min(a)) / (np.max(a) - np.min(a))
    return b


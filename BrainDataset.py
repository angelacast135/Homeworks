

import time
import torch
import torch.nn as nn
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import torchvision.datasets.folder as fl

import pdb

# pdb.set_trace()
# root = '/media/SSD3/acastillo/database_brains/Brats17PNG/NT_WT_images.txt'

#-------------------------------------------------------------------------
def txt_sloss(route):
    obj = open(route,'r')
    lins = obj.readlines(); i_rs = []; l_im = []
    obj.close()
    cls_l = lins[0]
    clss = cls_l.strip().split(',')
    for ligne in lins[1::]:
        [r_t, lb] = ligne.split(',')
        i_rs.append(r_t)
        l_c = lb.strip()
        l_im.append(l_c)
    return i_rs,l_im,clss

#-------------------------------------------------------------------------

class Singlab_dataset(data.Dataset):
    """ 
    Class Singlab_dataset
    Class implemented to generate a dataset for single category image 
    classification
 
    """
    def __init__(self,txt_r,transform = None, loader = fl.default_loader,
        txt_fun = txt_sloss):
        [iroutes,ilabs,iclass] = txt_fun(txt_r)
        self.impath = iroutes
        self.imlabs = ilabs
        self.loader = loader
        self.transform = transform
        self.classes = iclass

    def __getitem__(self,index):
        img = self.loader(self.impath[index])
        if self.transform is not None:
           img =  self.transform(img)
        kek = self.imlabs[index].strip()
        label = int(kek)
        return img,label

    def __len__(self):
        return len(self.imlabs) 
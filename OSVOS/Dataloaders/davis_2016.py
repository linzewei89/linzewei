import os
import numpy as np
import cv2

from Dataloaders.helpers import *
from torch.utils.data import Dataset
from PIL import Image

class DAVIS2016(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='./DAVIS/ImageSets/480p',
                 root = './DAVIS',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None):
        """Loads image to label pairs for tool pose estimation
        """
        self.train = train
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name
        
        if self.train:
            fname = 'train'
        else:
            fname = 'val'

        if self.seq_name is None:
            # Initialize the original DAVIS splits for training the parent network
            print(db_root_dir + '/' + fname +'.txt')
            with open(db_root_dir + '/' + fname +'.txt') as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                for seq in seqs:
                    Words = seq.strip().split() 
                    images_path = root + Words[0]
    #               images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images))
                    img_list.append(images_path)
                    lab_path = root + Words[1]
    #               lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
                    labels.append(lab_path)

            assert (len(labels) == len(img_list))
        
        else:
            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(root + '/JPEGImages/480p/' + str(seq_name)))
            img_list = []
            for item in names_img:
                img_list.append(root + '/JPEGImages/480p/' + str(seq_name) + '/' + item.strip())
                
            name_label = np.sort(os.listdir(root + '/Annotations/480p/' + str(seq_name)))
            labels = []
            for item in name_label:
                labels.append(root + '/Annotations/480p/' + str(seq_name) + '/' + item)
#             labels = [root + '/Annotations/480p/' + str(seq_name) + '/' + name_label[0].strip()]
#             labels.extend([None]*(len(names_img)-1))
            if self.train:
                img_list = [img_list[0]]
                labels = [labels[0]]
    #       print(self.img_list)
#           print(self.labels)
        
        self.img_list = img_list
        self.labels = labels
#         print('img_list',self.img_list)
#         print('labels',self.labels)
        
        print('Done initializing ')

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)
        
        sample = {'image': img, 'gt': gt}
        
        if self.seq_name is not None:
            fname = self.seq_name + str("%05d" % idx) 
            sample['fname'] = fname
            
        if self.transform is not None:
            sample = self.transform(sample)

        return sample 
    
    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(self.img_list[idx])  #shape(480, 854, 3)
        if self.labels[idx] is not None:
            label = cv2.imread(self.labels[idx])
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)
            
        if self.inputRes is not None:
            height = img.shape[1]
            width = img.shape[0]
            size = (int(width * inputRes), int(height * inputRes))  
            img = np.array(Image.fromarray(img).resize(size))
            if self.labels[idx] is not None:
                label = np.array(Image.fromarray(label).resize(size))
       
        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))
        
        if self.labels[idx] is not None:
            gt = np.array(label, dtype=np.float32)
            gt = gt/np.max([gt.max(), 1e-8])

        return img, gt
    
    def get_img_size(self):
        img = cv2.imread(root + self.img_list[0])
        
        return list(img.shape[:2])
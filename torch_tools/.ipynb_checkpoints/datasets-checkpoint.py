import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import os
import SimpleITK as sitk
import PIL as Image

def get_data_paths(data_dir):
    # load each type individually
    flair, t1w, t1wce, t2, labels = np.loadtxt(data_dir,
                                      dtype=np.str_,
                                      unpack=True,
                                      delimiter=",")
    # convert labels to ints
    labels = labels.astype(np.int_)
    return flair, t1w, t1wce, t2, labels

# very quickly made and tested dataloader. It's not very robust rn but its functional
# instancing the class: foo = flair_dataset("./data/train/", some_transform)
# get data by indexing: foo[[1,2,3]] or foo[8] 
# returns [ [imglist],  [labellist] ] 
class flair_dataset(Dataset):
    def __init__(self, data_table, transform=None):
        self.paths, self.labels = data_table
        self.transform=transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        
        # get list if multiple indeces are passed
        if isinstance(item, list):
            result = [sitk.GetArrayFromImage(sitk.ReadImage(self.paths[i])) for i in item]
        else:
            result = [sitk.GetArrayFromImage(sitk.ReadImage(self.paths[item]))]
        
        # convert to tensor to support batch transforms
        result = torch.FloatTensor(result)
            
        if self.transform is not None:
            result = self.transform(result)
            
        return result, self.labels[item]
        
        
import glob
import os.path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def data_normalization(imgs):
    sample_mean = [0 for z in range(1)]
    sample_std = [0 for z in range(1)]
    sample_num = len(imgs)
    for img in imgs:
        for i in range(1):
            sample_mean[i] += img[i, :, :].mean()
            sample_std[i] += img[i, :, :].std()
    sample_mean = np.asarray(sample_mean) / sample_num
    sample_std = np.asarray(sample_std) / sample_num
    transform_Nor = transforms.Normalize(
        mean=sample_mean,  # 取决于数据集
        std=sample_std
    )
    Nor_imgs = imgs
    for i in range(sample_num):
        Nor_imgs[i] = transform_Nor(imgs[i])
    return Nor_imgs


class MyData(Dataset):
    def __init__(self, data_location,cls):
        super(MyData, self).__init__()
        dataClses = glob.glob(os.path.join(data_location,"*"))

        dbz_inputs = []
        zdr_inputs = []
        labels = []

        dirs0 = glob.glob(os.path.join(dataClses[0],cls+"\\*"))
        for d in dirs0:
            sample = torch.unsqueeze(torch.tensor(np.load(d)),dim=0)
            dbz_inputs.append(sample)

        dbz_inputs = data_normalization(dbz_inputs)


        dirs1 = glob.glob(os.path.join(dataClses[1],cls+"\\*"))
        for d in dirs1:
            sample = torch.unsqueeze(torch.tensor(np.load(d)),dim=0)
            labels.append(sample)
        labels = data_normalization(labels)


        dirs2 = glob.glob(os.path.join(dataClses[2],cls+"\\*"))
        for d in dirs2:
            sample = torch.unsqueeze(torch.tensor(np.load(d)),dim=0)
            zdr_inputs.append(sample)

        zdr_inputs = data_normalization(zdr_inputs)


        self.dbz_inputs = dbz_inputs


        self.zdr_inputs = zdr_inputs

        self.labels = labels


    def __getitem__(self, index):
        dbz = self.dbz_inputs[index]
        zdr = self.zdr_inputs[index]

        label = self.labels[index]
        return dbz,zdr,label

    def __len__(self):
        return len(self.dbz_inputs)
import glob
import os.path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def data_normalization(imgs):
    sample_mean = [0 for z in range(10)]
    sample_std = [0 for z in range(10)]
    sample_num = len(imgs)
    for img in imgs:
        for i in range(10):
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
        kdp_inputs = []
        zdr_inputs = []

        dbz_labels = []
        kdp_labels = []
        zdr_labels = []


        dirs0 = glob.glob(os.path.join(dataClses[0],"3.0km/"+cls+"/*"))
        for d in dirs0:
            input_temp = torch.zeros((1, 256, 256))
            label_temp = torch.zeros((1, 256, 256))
            files = glob.glob(os.path.join(d, "*"))
            for i in range(10):
                sample = torch.unsqueeze(torch.tensor(np.load(files[i])), dim=0)
                input_temp = torch.cat([input_temp, sample], dim=0)
            input_temp = input_temp[1:, :, :]
            dbz_inputs.append(input_temp)
            for i in range(10, 20):
                sample = torch.unsqueeze(torch.tensor(np.load(files[i])), dim=0)
                label_temp = torch.cat([label_temp, sample], dim=0)
            label_temp = label_temp[1:, :, :]
            dbz_labels.append(label_temp)
        dbz_inputs = data_normalization(dbz_inputs)
        dbz_labels = data_normalization(dbz_labels)

        dirs1 = glob.glob(os.path.join(dataClses[1], "3.0km/"+cls+"/*"))
        for d in dirs1:
            input_temp = torch.zeros((1, 256, 256))
            label_temp = torch.zeros((1, 256, 256))
            files = glob.glob(os.path.join(d, "*"))
            for i in range(10):
                sample = torch.unsqueeze(torch.tensor(np.load(files[i])), dim=0)
                input_temp = torch.cat([input_temp, sample], dim=0)
            input_temp = input_temp[1:, :, :]
            kdp_inputs.append(input_temp)
            for i in range(10, 20):
                sample = torch.unsqueeze(torch.tensor(np.load(files[i])), dim=0)
                label_temp = torch.cat([label_temp, sample], dim=0)
            label_temp = label_temp[1:, :, :]
            kdp_labels.append(label_temp)
        kdp_inputs = data_normalization(kdp_inputs)
        kdp_labels = data_normalization(kdp_labels)

        dirs2 = glob.glob(os.path.join(dataClses[2], "3.0km/"+cls+"/*"))
        for d in dirs2:
            input_temp = torch.zeros((1, 256, 256))
            label_temp = torch.zeros((1, 256, 256))
            files = glob.glob(os.path.join(d, "*"))
            for i in range(10):
                sample = torch.unsqueeze(torch.tensor(np.load(files[i])), dim=0)
                input_temp = torch.cat([input_temp, sample], dim=0)
            input_temp = input_temp[1:, :, :]
            zdr_inputs.append(input_temp)
            for i in range(10, 20):
                sample = torch.unsqueeze(torch.tensor(np.load(files[i])), dim=0)
                label_temp = torch.cat([label_temp, sample], dim=0)
            label_temp = label_temp[1:, :, :]
            zdr_labels.append(label_temp)
        zdr_inputs = data_normalization(zdr_inputs)
        zdr_labels = data_normalization(zdr_labels)


        self.dbz_inputs = dbz_inputs
        self.dbz_labels = dbz_labels

        self.kdp_inputs = kdp_inputs
        self.kdp_labels = kdp_labels

        self.zdr_inputs = zdr_inputs
        self.zdr_labels = zdr_labels

    def __getitem__(self, index):
        dbz = self.dbz_inputs[index]
        kdp = self.dbz_inputs[index]
        zdr = self.dbz_inputs[index]

        label = self.dbz_labels[index]
        return dbz,kdp,zdr,label

    def __len__(self):
        return len(self.dbz_inputs)
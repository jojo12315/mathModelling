import math
import os
import shutil
from glob import glob

import numpy
import torch
import torch.nn as nn

from dataUtil import MyData
from torch.utils.data import DataLoader

if __name__ == '__main__':
    train_path = "SampleData\\"
    train_data = MyData(train_path, "validate")
    all_label = torch.Tensor([item.detach().numpy() for item in train_data.dbz_labels])
    train_label = torch.unsqueeze(all_label, dim=4)
    train_label = train_label.permute(0, 1, 4, 3, 2)

    path = "ConvLSTM_pred_validate"
    files = glob(os.path.join(path,"*"))
    pred = torch.zeros(train_label.shape)
    c = 0
    for f in files:
        sample = torch.squeeze(torch.Tensor(numpy.load(f)),0)
        pred[c] = sample
        c+=1
    c1 = nn.L1Loss()
    c2 = nn.MSELoss()

    print("MAE:{:.5f}".format(c1(pred,train_label)))
    print("MSE:{:.5f}".format(c2(pred, train_label)))
    print("RMSE:{:.5f}".format(torch.sqrt(c2(pred, train_label))))









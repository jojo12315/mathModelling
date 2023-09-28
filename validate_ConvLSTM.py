import numpy

from dataUtil import MyData
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from model import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    device = "cuda:0"
    train_path = "SampleData\\"

    train_data= MyData(train_path,"validate")

    train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
    model = ConvLSTM(3, 1 ,(3,3), 1, True, False, False)
    UNet3D_param = torch.load("Conv_LSTM.pt")
    model.load_state_dict(UNet3D_param)
    model = model.to(device)
    count = 0
    for dbz, kdp, zdr, label in train_dataloader:

        dbz = torch.unsqueeze(dbz, dim=4)
        kdp = torch.unsqueeze(kdp, dim=4)
        zdr = torch.unsqueeze(zdr, dim=4)

        train_input = torch.cat([dbz, kdp, zdr], dim=4)

        train_input = train_input.permute(0, 1, 4, 3, 2)

        train_input = train_input.to(device)
        numpy.save("ConvLSTM_pred_validate/pred"+str(count),model(train_input)[0][0].cpu().data.numpy())
        count+=1




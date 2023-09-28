import numpy

from rainfall_dataUtil import MyData
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from rainfall_unet import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    device = "cuda:0"
    train_path = "rainfal_SampleData\\"

    train_data= MyData(train_path,"test")

    train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)

    model = UNet(2)
    model_param = torch.load("rainfall_Unet.pt")
    model.load_state_dict(model_param)
    model = model.to(device)
    count = 0
    for dbz,zdr,label in train_dataloader:

        train_input = torch.cat([dbz,zdr],dim=1)

        train_input = train_input.to(device)
        result = model(train_input)
        numpy.save("test_rainfall_pred/pred" + str(count), model(train_input)[0][0].cpu().data.numpy())
        count += 1







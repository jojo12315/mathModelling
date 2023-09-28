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

    train_data= MyData(train_path)

    train_dataloader = DataLoader(dataset=train_data, batch_size=4, shuffle=False)

    epoch = 100
    model = UNet(1)
    model = model.to(device)
    c = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    auc_sum = []
    for i in range(epoch):
        all_result = []
        all_label = []
        l = 0
        for dbz,zdr,label in train_dataloader:

            train_input = zdr
            train_label = label


            train_input = train_input.to(device)
            train_label = train_label.to(device)
            result = model(train_input)
            loss = c(result, train_label)
            l += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch[{}/{},Loss:{:.5f}]".format(i + 1, epoch, l/len(train_dataloader)))
    torch.save(model.state_dict(), "rainfall_Unet_zdr.pt")





